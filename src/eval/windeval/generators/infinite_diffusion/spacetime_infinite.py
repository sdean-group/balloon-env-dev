"""InfiniteDiffusion over conditional space-time wind blocks.

This is the four-dimensional counterpart of :mod:`sampler`: channels are finite,
while physical time and both horizontal axes are lazily tiled.  It supports a
black-box ``T=1`` baseline and paper-style ``T=2`` inference, where the deterministic
EDM trajectory is split and overlapping intermediate states are blended before the
remaining denoising steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow

try:
    from .spacetime import SpaceTimeSampler
except ImportError:  # pragma: no cover - standalone cluster use
    from spacetime import SpaceTimeSampler


def _linear_weight_1d(n: int, eps: float, *, device, dtype) -> torch.Tensor:
    if n == 1:
        return torch.ones(1, device=device, dtype=dtype)
    center = (n - 1) / 2.0
    i = torch.arange(n, device=device, dtype=dtype)
    return eps + (1.0 - eps) * (1.0 - (i - center).abs() / center)


def _coordinate_noise(
    channels: int,
    tau: int,
    height: int,
    width: int,
    *,
    t0: int,
    y0: int,
    x0: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coordinate-keyed Gaussian noise, generated on CPU for MPS compatibility."""
    cpu = torch.device("cpu")
    c = torch.arange(channels, dtype=torch.int64, device=cpu)[:, None, None, None]
    t = torch.arange(t0, t0 + tau, dtype=torch.int64, device=cpu)[None, :, None, None]
    y = torch.arange(y0, y0 + height, dtype=torch.int64, device=cpu)[None, None, :, None]
    x = torch.arange(x0, x0 + width, dtype=torch.int64, device=cpu)[None, None, None, :]
    h = torch.full((1, 1, 1, 1), int(seed), dtype=torch.int64, device=cpu)
    h = h ^ (x * 6364136223846793005)
    h = h ^ (y * 1442695040888963407)
    h = h ^ (t * 22695477)
    h = h ^ (c * 1103515245)
    h = h ^ (h >> 33)
    h = h * 2862933555777941757 + 3037000493
    h = h ^ (h >> 29)
    uniform = (h & ((1 << 53) - 1)).to(torch.float64) / float(1 << 53)
    uniform = uniform.clamp(1e-12, 1.0 - 1e-12)
    noise = 2.0**0.5 * torch.erfinv(2.0 * uniform - 1.0)
    return noise.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SpaceTimeGrid:
    """Map integer tensor coordinates to the checkpoint's physical conditioning."""

    lat_origin: float = 25.0
    lon_origin: float = 225.0
    dlat: float = 0.25
    dlon: float = 0.25
    time_origin: str = "2023-01-15T00"
    dt_hours: int = 1

    def coordinates(
        self,
        *,
        t0: int,
        y0: int,
        x0: int,
        tau: int,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat = self.lat_origin + self.dlat * np.arange(y0, y0 + height)
        lon = self.lon_origin + self.dlon * np.arange(x0, x0 + width)
        offsets = np.arange(t0, t0 + tau) * np.timedelta64(self.dt_hours, "h")
        times = np.datetime64(self.time_origin) + offsets
        return lat, lon, times


class InfiniteSpaceTimeDiffusion:
    """Lazy, overlapping InfiniteDiffusion for ``(channel, time, y, x)`` wind.

    The generated tensor is normalized internally.  Public wind queries are
    de-normalized with the checkpoint statistics and returned as ``u`` and ``v``
    arrays of shape ``(time, level, height, width)``.
    """

    def __init__(
        self,
        sampler: SpaceTimeSampler,
        *,
        grid: SpaceTimeGrid | None = None,
        window: int = 64,
        stride: int | None = None,
        time_stride: int | None = None,
        seed: int = 0,
        outer_depth: int = 1,
        split_step: int | None = None,
        split_steps: tuple[int, ...] | list[int] | None = None,
        weight_eps: float = 0.01,
        cache_bytes: int | None = 512 * 1024 * 1024,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not sampler.conditional:
            raise ValueError("the space-time InfiniteDiffusion runner expects a conditional checkpoint")
        self.sampler = sampler
        self.device = sampler.device
        self.dtype = dtype
        self.C = int(sampler.n_channels)
        self.tau = int(sampler.tau)
        self.window = int(window)
        self.stride = int(stride if stride is not None else window // 2)
        self.time_stride = int(time_stride if time_stride is not None else max(1, self.tau // 2))
        self.seed = int(seed)
        self.outer_depth = int(outer_depth)
        self.split_steps = self._resolve_split_steps(split_step, split_steps)
        # Backward-compatible attribute for existing T=2 reports and callers.
        self.split_step = self.split_steps[0] if len(self.split_steps) == 1 else None
        self.grid = grid or SpaceTimeGrid()
        self.model_window_calls = 0
        self.model_forward_evaluations = 0
        self.evaluated_contexts: set[tuple[int, ...]] = set()
        self.phase_window_calls: dict[str, int] = {}
        self.phase_forward_evaluations: dict[str, int] = {}

        if not 0 < self.stride <= self.window:
            raise ValueError("stride must be in (0, window]")
        if not 0 < self.time_stride <= self.tau:
            raise ValueError("time_stride must be in (0, tau]")
        self.store = MemoryTileStore(cache_size_bytes=cache_bytes)
        wt = _linear_weight_1d(self.tau, weight_eps, device=self.device, dtype=dtype)
        wy = _linear_weight_1d(self.window, weight_eps, device=self.device, dtype=dtype)
        wx = _linear_weight_1d(self.window, weight_eps, device=self.device, dtype=dtype)
        self._weight = wt[:, None, None] * wy[None, :, None] * wx[None, None, :]
        self._build()

    def _resolve_split_steps(
        self,
        split_step: int | None,
        split_steps: tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        if self.outer_depth < 1:
            raise ValueError("outer_depth must be positive")
        if self.outer_depth > self.sampler.num_steps:
            raise ValueError("outer_depth cannot exceed the number of denoising steps")
        if split_step is not None and split_steps is not None:
            raise ValueError("provide split_step or split_steps, not both")
        if self.outer_depth == 1:
            if split_steps:
                raise ValueError("T=1 does not accept split_steps")
            return ()
        if split_steps is not None:
            resolved = tuple(int(step) for step in split_steps)
        elif split_step is not None:
            if self.outer_depth != 2:
                raise ValueError("split_step is the T=2 compatibility option; use split_steps")
            resolved = (int(split_step),)
        else:
            resolved = tuple(
                round(index * self.sampler.num_steps / self.outer_depth)
                for index in range(1, self.outer_depth)
            )
        if len(resolved) != self.outer_depth - 1:
            raise ValueError(
                f"T={self.outer_depth} requires {self.outer_depth - 1} split steps; "
                f"received {len(resolved)}"
            )
        if any(left >= right for left, right in zip((0, *resolved), (*resolved, self.sampler.num_steps))):
            raise ValueError(
                f"split_steps must be strictly increasing inside (0, {self.sampler.num_steps})"
            )
        return resolved

    def _overlap_window(self) -> TensorWindow:
        return TensorWindow(
            size=(self.C + 1, self.tau, self.window, self.window),
            stride=(self.C + 1, self.time_stride, self.stride, self.stride),
            offset=(0, 0, 0, 0),
        )

    @torch.no_grad()
    def _sample_segment(
        self,
        state: torch.Tensor,
        ctx: tuple[int, ...],
        *,
        start_step: int,
        end_step: int,
        unit_noise: bool,
        phase: str,
    ) -> torch.Tensor:
        _, wt, wy, wx = ctx
        t0 = wt * self.time_stride
        y0 = wy * self.stride
        x0 = wx * self.stride
        lat, lon, times = self.grid.coordinates(
            t0=t0,
            y0=y0,
            x0=x0,
            tau=self.tau,
            height=self.window,
            width=self.window,
        )
        cond, tfeat = self.sampler._condition((self.window, self.window), lat, lon, times)
        model_input = state.permute(1, 0, 2, 3).unsqueeze(0)
        prediction = self.sampler._heun_segment(
            model_input,
            start_step=start_step,
            end_step=end_step,
            unit_noise=unit_noise,
            cond=cond,
            tfeat=tfeat,
        )[0]
        self.model_window_calls += 1
        self.phase_window_calls[phase] = self.phase_window_calls.get(phase, 0) + 1
        forward_evaluations = 2 * (end_step - start_step) - int(end_step == self.sampler.num_steps)
        self.model_forward_evaluations += forward_evaluations
        self.phase_forward_evaluations[phase] = (
            self.phase_forward_evaluations.get(phase, 0) + forward_evaluations
        )
        self.evaluated_contexts.add(tuple(int(value) for value in ctx))
        return prediction.permute(1, 0, 2, 3).to(self.dtype)

    def _build(self) -> None:
        shape = (self.C + 1, None, None, None)
        C, tau, win = self.C, self.tau, self.window
        weight = self._weight.unsqueeze(0)

        def initial_f(ctx, end_step):
            _, wt, wy, wx = ctx
            t0 = wt * self.time_stride
            y0 = wy * self.stride
            x0 = wx * self.stride
            noise = _coordinate_noise(
                C,
                tau,
                win,
                win,
                t0=t0,
                y0=y0,
                x0=x0,
                seed=self.seed,
                device=self.device,
                dtype=self.dtype,
            )
            prediction = self._sample_segment(
                noise,
                ctx,
                start_step=0,
                end_step=end_step,
                unit_noise=True,
                phase="initial",
            )
            return torch.cat([weight * prediction, weight], dim=0)

        boundaries = (0, *self.split_steps, self.sampler.num_steps)
        schedule_id = "-".join(str(step) for step in self.split_steps) or "full"
        initial_end = boundaries[1]
        initial = InfiniteTensor(
            shape=shape,
            f=lambda ctx: initial_f(ctx, initial_end),
            output_window=self._overlap_window(),
            dtype=self.dtype,
            device=self.device,
            tile_store=self.store,
            tensor_id=(
                f"wind-idiff4d-{self.seed}-T{self.outer_depth}-"
                f"splits{schedule_id}-phase0"
            ),
        )
        self.phases = [initial]
        overlap_window = self._overlap_window()
        for phase_index in range(1, self.outer_depth):
            previous_phase = self.phases[-1]
            start_step = boundaries[phase_index]
            end_step = boundaries[phase_index + 1]
            phase_name = (
                "continuation" if self.outer_depth == 2
                else f"continuation_{phase_index}"
            )

            def continuation_f(
                ctx,
                previous,
                segment_start=start_step,
                segment_end=end_step,
                segment_name=phase_name,
            ):
                intermediate = previous[:C] / previous[C:].clamp_min(1e-12)
                prediction = self._sample_segment(
                    intermediate,
                    ctx,
                    start_step=segment_start,
                    end_step=segment_end,
                    unit_noise=False,
                    phase=segment_name,
                )
                return torch.cat([weight * prediction, weight], dim=0)

            phase = InfiniteTensor(
                shape=shape,
                f=continuation_f,
                output_window=overlap_window,
                args=(previous_phase,),
                args_windows=(overlap_window,),
                dtype=self.dtype,
                device=self.device,
                tile_store=self.store,
                tensor_id=(
                    f"wind-idiff4d-{self.seed}-T{self.outer_depth}-"
                    f"splits{schedule_id}-phase{phase_index}"
                ),
            )
            self.phases.append(phase)
        self.packed = self.phases[-1]

    def materialize(
        self,
        t0: int,
        t1: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> torch.Tensor:
        """Return normalized wind as ``(channel, time, height, width)``."""
        packed = self.packed[0:self.C + 1, int(t0):int(t1), int(y0):int(y1), int(x0):int(x1)]
        return packed[:self.C] / packed[self.C:].clamp_min(1e-12)

    def field_uv(
        self,
        t0: int,
        t1: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.materialize(t0, t1, y0, y1, x0, x1)
        frames = normalized.permute(1, 0, 2, 3)
        wind = self.sampler.stats.denormalize(frames)
        wind = wind.reshape(wind.shape[0], self.sampler.n_levels, 2, wind.shape[2], wind.shape[3])
        array = wind.detach().cpu().numpy()
        return array[:, :, 0], array[:, :, 1]

    def clear_cache(self) -> None:
        for phase in reversed(self.phases):
            phase.clear_cache()
