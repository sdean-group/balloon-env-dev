"""InfiniteDiffusion — lazy MultiDiffusion on an infinite lattice (the *machinery*).

This is the blending wrapper from Goslin, *InfiniteDiffusion / Terrain Diffusion*
(arXiv 2512.08309), specialised to wind fields and built on the `infinite-tensor`
library. It is denoiser-agnostic: pass any `WindowDenoiser` (the toy now, a trained
model later) and it inherits the Axis-2 properties — seamless tiling, seed-consistency,
constant-time random access, constant memory.

The math (paper Eq. 2 / Algorithm 1)
------------------------------------
Each diffusion step is a weighted average of overlapping window denoiser predictions::

    J_t[R] = A_t[R] / B[R],
        A_t = sum over windows i overlapping R of  U_i( W_i ⊗ Phi(J_{t+1}[R_i]) )   (numerator)
        B   = sum over windows i overlapping R of  U_i( W_i )                       (denominator)

generated lazily: only the windows intersecting a queried region are evaluated.

Mapping onto `infinite-tensor`
------------------------------
- `A_t`  : an InfiniteTensor whose `f` returns ``W ⊗ Phi(J_{t+1}[R_i])`` per window;
           overlapping windows are **summed** by the store (its default blend) → numerator.
- `B`    : an InfiniteTensor whose `f` returns the window weight map ``W``; summed → denominator.
           B is pure geometry (no noise, no step) so it is identical at every t → built ONCE.
- `J_t`  : an InfiniteTensor that divides ``A_t / B`` pointwise (non-overlapping windows).
- noise  : the base ``J_T``; seeded per-tile from the global seed → seed-consistency.
- recursion `J_t depends on J_{t+1}` : dependency chaining (`args` / `args_windows`).
- truncation: ``T`` is small (default 2). Each query of `J_t[R]` reads a slightly larger
  region of `J_{t+1}`, so cost grows ~geometrically in T; the paper shows T=2 ≈ full quality.
- LRU cache + recompute-on-miss: handled by `MemoryTileStore` given a deterministic `f`.
"""
from __future__ import annotations

import numpy as np
import torch
from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow

from .denoiser import WindowDenoiser


def _linear_weight(win: int, eps: float, *, device, dtype) -> torch.Tensor:
    """Separable linear window weight: 1 at the centre decaying to ``eps`` at the edge.

    The paper finds this beats a constant map (FID 14.78 vs 19.32) — the taper is what
    makes overlapping windows fuse without a visible step.
    """
    c = (win - 1) / 2.0
    i = torch.arange(win, device=device, dtype=dtype)
    w1 = eps + (1.0 - eps) * (1.0 - (i - c).abs() / c)
    return torch.outer(w1, w1)  # (win, win)


def _tile_seed(seed: int, wy: int, wx: int) -> int:
    """Deterministic 63-bit per-tile seed from the global seed and (signed) tile index.

    A splitmix64-style mix in masked 64-bit Python integer arithmetic (no numpy overflow).
    """
    M = (1 << 64) - 1
    h = ((seed & M) * 0x9E3779B97F4A7C15) & M
    h ^= ((wy & 0xFFFFFFFF) * 0xBF58476D1CE4E5B9) & M
    h ^= ((wx & 0xFFFFFFFF) * 0x94D049BB133111EB) & M
    h &= M
    h ^= h >> 31
    return int(h & 0x7FFFFFFFFFFFFFFF)


class InfiniteDiffusion:
    """A seed-consistent, infinitely-extensible wind field.

    Args:
        denoiser: the window denoiser Phi (channels must be ``2 * n_levels``).
        window: side length of each square denoiser window (pixels).
        stride: spacing between overlapping windows. ``< window`` gives the overlap that
            the weighted average uses to blend; ``window // 2`` (50% overlap) is typical.
        T: number of blend steps (truncated diffusion). 2 ≈ full quality per the paper.
        seed: global seed; the entire infinite field is a deterministic function of it.
        weight_eps: edge value of the linear window weight (small > 0 keeps B strictly positive).
        cache_bytes: MemoryTileStore cache cap (constant-memory knob). None = unbounded.
        device, dtype: torch placement.
    """

    def __init__(
        self,
        denoiser: WindowDenoiser,
        *,
        window: int = 64,
        stride: int | None = None,
        T: int = 2,
        seed: int = 0,
        weight_eps: float = 0.01,
        cache_bytes: int | None = 256 * 1024 * 1024,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(denoiser, WindowDenoiser):
            raise TypeError("denoiser must implement the WindowDenoiser protocol")
        self.denoiser = denoiser
        self.C = int(denoiser.n_channels)
        self.window = int(window)
        self.stride = int(stride if stride is not None else window // 2)
        self.T = int(T)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype

        if self.stride <= 0 or self.stride > self.window:
            raise ValueError("stride must be in (0, window]")
        if self.T < 1:
            raise ValueError("T must be >= 1")

        self.store = MemoryTileStore(cache_size_bytes=cache_bytes)
        self._W = _linear_weight(self.window, weight_eps, device=self.device, dtype=self.dtype)

        self._build()

    # ----- window specs -----
    def _out_win(self) -> TensorWindow:
        """Overlapping window over the (C, y, x) grid; C dim is finite (single index 0)."""
        return TensorWindow(
            size=(self.C, self.window, self.window),
            stride=(self.C, self.stride, self.stride),
            offset=(0, 0, 0),
        )

    def _tile_win(self) -> TensorWindow:
        """Non-overlapping tile window (for the pointwise division tensor / noise base)."""
        return TensorWindow(
            size=(self.C, self.window, self.window),
            stride=(self.C, self.window, self.window),
            offset=(0, 0, 0),
        )

    def _build(self) -> None:
        C, win = self.C, self.window
        shape = (C, None, None)

        # --- base noise J_T: per-tile seeded gaussian, deterministic in (seed, tile idx) ---
        def noise_f(ctx):
            _, wy, wx = ctx
            g = torch.Generator(device="cpu").manual_seed(_tile_seed(self.seed, wy, wx))
            n = torch.randn(C, win, win, generator=g, dtype=self.dtype)
            return n.to(self.device)

        noise = InfiniteTensor(
            shape=shape, f=noise_f, output_window=self._tile_win(),
            dtype=self.dtype, device=self.device, tile_store=self.store,
            tensor_id=f"idiff-{self.seed}-noise",
        )

        # --- denominator B = sum of overlapping window weights (geometry only, all t) ---
        Wfull = self._W.unsqueeze(0).expand(C, win, win).contiguous()

        def weight_f(ctx):  # noqa: ARG001
            return Wfull.clone()

        B = InfiniteTensor(
            shape=shape, f=weight_f, output_window=self._out_win(),
            dtype=self.dtype, device=self.device, tile_store=self.store,
            tensor_id=f"idiff-{self.seed}-B",
        )

        # --- recursion: for t = T-1 .. 0 build A_t (numerator) then J_t = A_t / B ---
        Wb = self._W.unsqueeze(0)  # (1, win, win) broadcasts over channels

        J_next = noise
        for t in range(self.T - 1, -1, -1):
            def num_f(ctx, jnext, _t=t):  # noqa: ARG001
                phi = self.denoiser(jnext, _t)            # (C, win, win)
                return Wb * phi                            # weighted prediction

            A_t = InfiniteTensor(
                shape=shape, f=num_f, output_window=self._out_win(),
                args=(J_next,), args_windows=(self._out_win(),),
                dtype=self.dtype, device=self.device, tile_store=self.store,
                tensor_id=f"idiff-{self.seed}-A{t}",
            )

            def div_f(ctx, a, b):  # noqa: ARG001
                return a / b

            J_t = InfiniteTensor(
                shape=shape, f=div_f, output_window=self._tile_win(),
                args=(A_t, B), args_windows=(self._tile_win(), self._tile_win()),
                dtype=self.dtype, device=self.device, tile_store=self.store,
                tensor_id=f"idiff-{self.seed}-J{t}",
            )
            J_next = J_t

        self.noise, self.B, self.J0 = noise, B, J_next

    # ----- queries -----
    def materialize(self, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
        """Return the final field J_0 over pixel region [y0,y1) x [x0,x1) as (C, H, W)."""
        return self.J0[0 : self.C, int(y0):int(y1), int(x0):int(x1)]

    def field_uv(self, y0: int, y1: int, x0: int, x1: int) -> tuple[np.ndarray, np.ndarray]:
        """Materialize and split into (u, v) numpy arrays of shape (n_levels, H, W)."""
        f = self.materialize(y0, y1, x0, x1).cpu().numpy()
        L = self.C // 2
        f = f.reshape(L, 2, *f.shape[1:])
        return f[:, 0], f[:, 1]

    def seam_lines(self, y0: int, y1: int, x0: int, x1: int) -> dict[str, list[int]]:
        """Window-stride stitch lines inside a region (targets for the Axis-2 seam metric)."""
        ys = [y for y in range(0, y1, self.stride) if y0 < y < y1]
        ys += [y for y in range(-self.stride, y0 - 1, -self.stride) if y0 < y < y1]
        xs = [x for x in range(0, x1, self.stride) if x0 < x < x1]
        xs += [x for x in range(-self.stride, x0 - 1, -self.stride) if x0 < x < x1]
        return {"y": sorted(set(ys)), "x": sorted(set(xs))}

    def clear_cache(self) -> None:
        self.J0.clear_cache()
