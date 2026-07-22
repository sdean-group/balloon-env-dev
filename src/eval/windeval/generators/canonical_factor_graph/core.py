"""Canonical Factor-Graph Diffusion (CFGD).

CFGD keeps the trained space-time denoiser frozen.  It replaces recursive lazy diffusion
with a locally finite atlas of canonical charts.  Inside each chart, overlapping model
windows are factors connected through one shared noisy state; their EDM directions are
fused after every denoising evaluation before the chart takes its global Heun step.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
import torch


class SpaceTimeSamplerLike(Protocol):
    device: torch.device
    n_channels: int
    n_levels: int
    tau: int
    num_steps: int
    conditional: bool
    model: torch.nn.Module
    stats: object

    def sigma_schedule(self, *, device=None, dtype=torch.float64) -> torch.Tensor: ...
    def _condition(self, hw, lat, lon, times): ...


@dataclass(frozen=True)
class SpaceTimeGrid:
    """Map integer chart coordinates to physical model conditioning."""

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


@dataclass(frozen=True)
class ChartConfig:
    """Geometry of one canonical chart and its local window factor graph."""

    core_time: int = 2
    core_size: int = 64
    halo_time: int = 1
    halo_size: int = 32
    window_size: int = 64
    window_stride: int = 32
    time_stride: int = 2
    window_weight_eps: float = 0.01
    window_batch_size: int = 1

    @property
    def support_time(self) -> int:
        return self.core_time + 2 * self.halo_time

    @property
    def support_size(self) -> int:
        return self.core_size + 2 * self.halo_size


def _linear_window_weight(n: int, eps: float, *, device, dtype) -> torch.Tensor:
    if n == 1:
        return torch.ones(1, device=device, dtype=dtype)
    center = (n - 1) / 2.0
    index = torch.arange(n, device=device, dtype=dtype)
    return eps + (1.0 - eps) * (1.0 - (index - center).abs() / center)


def _chart_weight_1d(core: int, halo: int, *, device, dtype) -> torch.Tensor:
    weight = torch.ones(core + 2 * halo, device=device, dtype=dtype)
    if halo:
        ramp = torch.arange(1, halo + 1, device=device, dtype=dtype) / (halo + 1)
        weight[:halo] = ramp
        weight[-halo:] = ramp.flip(0)
    return weight


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
    hashed = torch.full((1, 1, 1, 1), int(seed), dtype=torch.int64, device=cpu)
    hashed = hashed ^ (x * 6364136223846793005)
    hashed = hashed ^ (y * 1442695040888963407)
    hashed = hashed ^ (t * 22695477)
    hashed = hashed ^ (c * 1103515245)
    hashed = hashed ^ (hashed >> 33)
    hashed = hashed * 2862933555777941757 + 3037000493
    hashed = hashed ^ (hashed >> 29)
    uniform = (hashed & ((1 << 53) - 1)).to(torch.float64) / float(1 << 53)
    uniform = uniform.clamp(1e-12, 1.0 - 1e-12)
    noise = 2.0**0.5 * torch.erfinv(2.0 * uniform - 1.0)
    return noise.to(device=device, dtype=dtype)


def _covering_chart_indices(start: int, stop: int, core: int, halo: int) -> list[int]:
    if stop <= start:
        return []
    low = (start - halo) // core - 1
    high = (stop + halo - 1) // core + 1
    return [
        index
        for index in range(low, high + 1)
        if index * core - halo < stop and (index + 1) * core + halo > start
    ]


class CanonicalFactorGraphField:
    """Lazy infinite atlas of fixed, per-step-consensus diffusion charts.

    Internal tensors use ``(channel, time, y, x)``.  A chart is generated independently
    from coordinate-keyed noise, but neighboring chart outputs are assembled through a
    deterministic partition of unity.  Recomputing an evicted chart is therefore exact.
    """

    def __init__(
        self,
        sampler: SpaceTimeSamplerLike,
        *,
        config: ChartConfig | None = None,
        grid: SpaceTimeGrid | None = None,
        seed: int = 0,
        max_cached_charts: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.sampler = sampler
        self.device = torch.device(sampler.device)
        self.dtype = dtype
        self.config = config or ChartConfig()
        self.grid = grid or SpaceTimeGrid()
        self.seed = int(seed)
        self.max_cached_charts = int(max_cached_charts)
        self.C = int(sampler.n_channels)
        self.tau = int(sampler.tau)

        if not sampler.conditional:
            raise ValueError("CFGD currently requires a location/time-conditional checkpoint")
        if self.max_cached_charts < 1:
            raise ValueError("max_cached_charts must be positive")
        self._validate_geometry()

        cfg = self.config
        wt = _linear_window_weight(self.tau, cfg.window_weight_eps,
                                   device=self.device, dtype=dtype)
        wy = _linear_window_weight(cfg.window_size, cfg.window_weight_eps,
                                   device=self.device, dtype=dtype)
        self._window_weight = wt[:, None, None] * wy[None, :, None] * wy[None, None, :]

        ct = _chart_weight_1d(cfg.core_time, cfg.halo_time,
                              device=self.device, dtype=dtype)
        cy = _chart_weight_1d(cfg.core_size, cfg.halo_size,
                              device=self.device, dtype=dtype)
        self._chart_weight = ct[:, None, None] * cy[None, :, None] * cy[None, None, :]

        self._factor_offsets = list(product(
            range(0, cfg.support_time - self.tau + 1, cfg.time_stride),
            range(0, cfg.support_size - cfg.window_size + 1, cfg.window_stride),
            range(0, cfg.support_size - cfg.window_size + 1, cfg.window_stride),
        ))
        self._cache: OrderedDict[tuple[int, int, int], torch.Tensor] = OrderedDict()
        self.charts_generated = 0
        self.chart_cache_hits = 0
        self.model_window_evaluations = 0
        self.model_batch_calls = 0

    def _validate_geometry(self) -> None:
        cfg = self.config
        if cfg.support_time < self.tau or cfg.support_size < cfg.window_size:
            raise ValueError("chart support must be at least one model window in every axis")
        if (cfg.support_time - self.tau) % cfg.time_stride:
            raise ValueError("time support must be exactly coverable by time-strided windows")
        if (cfg.support_size - cfg.window_size) % cfg.window_stride:
            raise ValueError("spatial support must be exactly coverable by strided windows")
        if cfg.window_batch_size < 1:
            raise ValueError("window_batch_size must be positive")

    def _chart_origin(self, key: tuple[int, int, int]) -> tuple[int, int, int]:
        kt, ky, kx = key
        cfg = self.config
        return (
            kt * cfg.core_time - cfg.halo_time,
            ky * cfg.core_size - cfg.halo_size,
            kx * cfg.core_size - cfg.halo_size,
        )

    def _factor_condition(
        self, chart_origin: tuple[int, int, int], offset: tuple[int, int, int]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        t0, y0, x0 = (origin + delta for origin, delta in zip(chart_origin, offset))
        lat, lon, times = self.grid.coordinates(
            t0=t0, y0=y0, x0=x0, tau=self.tau,
            height=self.config.window_size, width=self.config.window_size,
        )
        return self.sampler._condition(
            (self.config.window_size, self.config.window_size), lat, lon, times
        )

    @torch.no_grad()
    def _consensus_direction(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        chart_origin: tuple[int, int, int],
    ) -> torch.Tensor:
        cfg = self.config
        accumulated = torch.zeros_like(state)
        normalizer = torch.zeros((1, *state.shape[1:]), device=self.device, dtype=self.dtype)

        for start in range(0, len(self._factor_offsets), cfg.window_batch_size):
            offsets = self._factor_offsets[start:start + cfg.window_batch_size]
            model_inputs = []
            conditions = []
            time_features = []
            for dt, dy, dx in offsets:
                window = state[:, dt:dt + self.tau, dy:dy + cfg.window_size,
                               dx:dx + cfg.window_size]
                model_inputs.append(window.permute(1, 0, 2, 3))
                cond, tfeat = self._factor_condition(chart_origin, (dt, dy, dx))
                conditions.append(cond)
                time_features.append(tfeat)

            batch = torch.stack(model_inputs, dim=0)
            cond_batch = torch.cat(conditions, dim=0) if conditions[0] is not None else None
            time_batch = torch.cat(time_features, dim=0) if time_features[0] is not None else None
            sigmas = sigma.expand(len(offsets))
            denoised = self.sampler.model(batch, sigmas, cond=cond_batch, tfeat=time_batch)
            directions = (batch - denoised) / sigma
            self.model_batch_calls += 1
            self.model_window_evaluations += len(offsets)

            for direction, (dt, dy, dx) in zip(directions, offsets):
                placed = direction.permute(1, 0, 2, 3)
                accumulated[:, dt:dt + self.tau, dy:dy + cfg.window_size,
                            dx:dx + cfg.window_size] += self._window_weight.unsqueeze(0) * placed
                normalizer[:, dt:dt + self.tau, dy:dy + cfg.window_size,
                           dx:dx + cfg.window_size] += self._window_weight.unsqueeze(0)

        if torch.any(normalizer <= 0):
            raise RuntimeError("factor windows did not cover the complete canonical chart")
        return accumulated / normalizer

    @torch.no_grad()
    def _generate_chart(self, key: tuple[int, int, int]) -> torch.Tensor:
        cfg = self.config
        origin = self._chart_origin(key)
        state = _coordinate_noise(
            self.C, cfg.support_time, cfg.support_size, cfg.support_size,
            t0=origin[0], y0=origin[1], x0=origin[2], seed=self.seed,
            device=self.device, dtype=self.dtype,
        )
        schedule = self.sampler.sigma_schedule(device=self.device, dtype=self.dtype)
        state = state * schedule[0]
        for step in range(self.sampler.num_steps):
            sigma, sigma_next = schedule[step], schedule[step + 1]
            direction = self._consensus_direction(state, sigma, origin)
            proposal = state + (sigma_next - sigma) * direction
            if step + 1 < self.sampler.num_steps:
                corrected = self._consensus_direction(proposal, sigma_next, origin)
                state = state + 0.5 * (sigma_next - sigma) * (direction + corrected)
            else:
                state = proposal
        self.charts_generated += 1
        return state

    def _get_chart(self, key: tuple[int, int, int]) -> torch.Tensor:
        chart = self._cache.get(key)
        if chart is not None:
            self._cache.move_to_end(key)
            self.chart_cache_hits += 1
            return chart
        chart = self._generate_chart(key)
        self._cache[key] = chart
        while len(self._cache) > self.max_cached_charts:
            self._cache.popitem(last=False)
        return chart

    def chart_keys_for_query(
        self, t0: int, t1: int, y0: int, y1: int, x0: int, x1: int
    ) -> list[tuple[int, int, int]]:
        cfg = self.config
        axes = (
            _covering_chart_indices(t0, t1, cfg.core_time, cfg.halo_time),
            _covering_chart_indices(y0, y1, cfg.core_size, cfg.halo_size),
            _covering_chart_indices(x0, x1, cfg.core_size, cfg.halo_size),
        )
        return list(product(*axes))

    @torch.no_grad()
    def materialize(
        self, t0: int, t1: int, y0: int, y1: int, x0: int, x1: int
    ) -> torch.Tensor:
        """Return normalized wind with shape ``(channel, time, y, x)``."""
        if t1 <= t0 or y1 <= y0 or x1 <= x0:
            raise ValueError("query bounds must have positive extent")
        output = torch.zeros((self.C, t1 - t0, y1 - y0, x1 - x0),
                             device=self.device, dtype=self.dtype)
        normalizer = torch.zeros((1, t1 - t0, y1 - y0, x1 - x0),
                                 device=self.device, dtype=self.dtype)
        cfg = self.config
        for key in self.chart_keys_for_query(t0, t1, y0, y1, x0, x1):
            chart = self._get_chart(key)
            ct0, cy0, cx0 = self._chart_origin(key)
            ct1, cy1, cx1 = ct0 + cfg.support_time, cy0 + cfg.support_size, cx0 + cfg.support_size
            it0, it1 = max(t0, ct0), min(t1, ct1)
            iy0, iy1 = max(y0, cy0), min(y1, cy1)
            ix0, ix1 = max(x0, cx0), min(x1, cx1)
            if it1 <= it0 or iy1 <= iy0 or ix1 <= ix0:
                continue
            chart_slice = (
                slice(it0 - ct0, it1 - ct0),
                slice(iy0 - cy0, iy1 - cy0),
                slice(ix0 - cx0, ix1 - cx0),
            )
            output_slice = (
                slice(it0 - t0, it1 - t0),
                slice(iy0 - y0, iy1 - y0),
                slice(ix0 - x0, ix1 - x0),
            )
            weight = self._chart_weight[chart_slice]
            output[(slice(None), *output_slice)] += chart[(slice(None), *chart_slice)] * weight
            normalizer[(slice(None), *output_slice)] += weight.unsqueeze(0)
        if torch.any(normalizer <= 0):
            raise RuntimeError("canonical charts did not cover the complete query")
        return output / normalizer

    def field_uv(
        self, t0: int, t1: int, y0: int, y1: int, x0: int, x1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.materialize(t0, t1, y0, y1, x0, x1)
        frames = normalized.permute(1, 0, 2, 3)
        wind = self.sampler.stats.denormalize(frames)
        wind = wind.reshape(wind.shape[0], self.sampler.n_levels, 2,
                            wind.shape[2], wind.shape[3])
        array = wind.detach().cpu().numpy()
        return array[:, :, 0], array[:, :, 1]

    def clear_cache(self) -> None:
        self._cache.clear()
