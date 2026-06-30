"""The window-denoiser contract (Phi) and an analytic toy that satisfies it.

InfiniteDiffusion is a blending *wrapper* around a window denoiser Phi. The wrapper
(sampler.py) owns the Axis-2 claims (seamless / seed-consistent / O(1) / constant-mem);
Phi owns Axis-1 field quality. They are built and validated independently. This file is
the Phi side: an interface + an ML-free stand-in good enough to exercise the machinery.

Phi contract
------------
A denoiser maps a noisy window to a denoised window of the *same* shape::

    phi(x, t) -> y          x, y : torch.Tensor of shape (C, H, W),  C = 2 * n_levels

Channel layout is interleaved per level: channel ``2*l`` is u at level l, ``2*l+1`` is v.
Phi MUST be deterministic in ``x`` (the library caches outputs and assumes purity). ``t``
is the (truncated) diffusion step index, smaller = cleaner; the toy is stateless and
ignores it. In the paper's truncated-T formulation Phi is "an arbitrary denoising
function" — a full few-step sampler, not one atomic step — so this signature is faithful.

ToyDivFreeDenoiser
------------------
Maps any window to a **divergence-free** field with a **controlled radial spectrum**:

    (u,v) --FFT--> vorticity zeta --solve--> streamfunction psi
    impose |psi(k)| = target(k) (keep phase)            # pins the spectrum, T-independent
    (u,v) = (-d psi/dy, d psi/dx)                        # curl of psi => exactly div-free

The *phase* of psi carries the input noise, so different windows differ and the sampler's
blending has real seams to remove; the *amplitude* is pinned, so the output spectrum is
stationary and independent of how many blend steps T are taken. This is deliberately a
stand-in: real stratospheric wind is not perfectly non-divergent, but div-free + a -3-ish
slope is the right zeroth-order shape and makes the toy a sane Axis-1 baseline too.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class WindowDenoiser(Protocol):
    """A fixed-size window denoiser Phi. See module docstring for the contract."""

    n_channels: int  # = 2 * n_levels

    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor: ...


def _freq_grids(h: int, w: int, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Angular spatial frequencies (kx, ky) on an (h, w) grid, unit pixel spacing."""
    ky = 2.0 * torch.pi * torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
    kx = 2.0 * torch.pi * torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    return KX, KY


class ToyDivFreeDenoiser:
    """Analytic divergence-free denoiser with a pinned radial spectrum (no ML).

    Args:
        n_levels: number of vertical levels; ``n_channels = 2 * n_levels``.
        slope: target *power* spectral slope of u, v vs wavenumber (≈ -3 ~ synoptic).
        k0: low-wavenumber softening (in cycles·2π / pixel) so the DC neighbourhood is
            finite; sets the dominant eddy scale.
        rms: target root-mean-square wind speed (m/s) the output is normalised to.
        vertical_corr: 0 → levels independent; >0 blends adjacent levels (exp kernel,
            scale = this many levels) so vertical shear is not pathologically random.
    """

    def __init__(
        self,
        n_levels: int,
        *,
        slope: float = -3.0,
        k0: float = 0.05,
        rms: float = 15.0,
        vertical_corr: float = 1.0,
    ) -> None:
        self.n_levels = int(n_levels)
        self.n_channels = 2 * self.n_levels
        self.slope = float(slope)
        self.k0 = float(k0)
        self.rms = float(rms)
        self.vertical_corr = float(vertical_corr)

    def __call__(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:  # noqa: ARG002
        if x.shape[0] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[0]}")
        C, H, W = x.shape
        dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        x = x.to(dtype)

        # (C,H,W) -> (L, 2, H, W): channel 2l = u_l, 2l+1 = v_l
        f = x.reshape(self.n_levels, 2, H, W)
        u, v = f[:, 0], f[:, 1]  # (L,H,W)

        KX, KY = _freq_grids(H, W, device=x.device, dtype=dtype)
        k2 = KX * KX + KY * KY
        kr = torch.sqrt(k2)
        dc = k2 == 0

        U, V = torch.fft.fft2(u), torch.fft.fft2(v)  # (L,H,W) complex
        # vorticity zeta = dv/dx - du/dy  (in Fourier: i*kx*V - i*ky*U)
        zeta = 1j * KX * V - 1j * KY * U
        # streamfunction from Laplacian(psi) = zeta  ->  psi_hat = -zeta / k^2
        k2_safe = torch.where(dc, torch.ones_like(k2), k2)
        psi = -zeta / k2_safe
        psi = torch.where(dc, torch.zeros_like(psi), psi)

        # pin the amplitude to the target radial profile, keep the (noise-carrying) phase.
        # u,v = grad(psi) multiplies the psi power by k^2, so to get a u,v power slope
        # `slope` we want psi power slope `slope - 2`  => |psi(k)| ~ (k0 + k)^((slope-2)/2).
        amp = (self.k0 + kr) ** ((self.slope - 2.0) / 2.0)
        amp = torch.where(dc, torch.zeros_like(amp), amp)
        psi = amp * torch.exp(1j * torch.angle(psi))

        # curl of psi: u = -d psi/dy, v = d psi/dx  -> divergence-free by construction
        Uout = -1j * KY * psi
        Vout = 1j * KX * psi
        u_out = torch.fft.ifft2(Uout).real
        v_out = torch.fft.ifft2(Vout).real

        if self.vertical_corr > 0 and self.n_levels > 1:
            u_out = self._smooth_levels(u_out)
            v_out = self._smooth_levels(v_out)

        out = torch.stack([u_out, v_out], dim=1).reshape(C, H, W)
        # normalise to target RMS speed
        speed_rms = torch.sqrt((u_out**2 + v_out**2).mean()).clamp_min(1e-12)
        out = out * (self.rms / speed_rms)
        return out.to(x.dtype)

    def _smooth_levels(self, a: torch.Tensor) -> torch.Tensor:
        """Normalised exponential blur across the level axis of an (L,H,W) tensor."""
        L = a.shape[0]
        idx = torch.arange(L, device=a.device, dtype=a.dtype)
        ker = torch.exp(-torch.abs(idx[:, None] - idx[None, :]) / self.vertical_corr)
        ker = ker / ker.sum(dim=1, keepdim=True)  # (L,L) row-normalised
        return torch.einsum("lm,mhw->lhw", ker, a)
