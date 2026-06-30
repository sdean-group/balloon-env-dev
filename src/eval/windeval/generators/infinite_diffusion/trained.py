"""TrainedWindowDenoiser — wrap a trained EDM model as the sampler's Phi(window, t).

This is where the *model* half plugs back into the *machinery* half with zero changes to
``sampler.py``. The trained network is a denoiser D(x; sigma); the InfiniteDiffusion
machinery wants a ``WindowDenoiser`` Phi that maps a noise window to a field window. We
bridge them with a **full internal deterministic EDM (Heun) sampler**: each Phi call runs
the complete reverse ODE (noise -> clean) on its window.

The Phi <-> machinery mapping (the design detail the handoff flagged)
--------------------------------------------------------------------
- The machinery's base ``J_T`` is unit Gaussian noise per tile; Phi lifts it to the EDM
  top noise level (``x_init = x * sigma_max``) and integrates the probability-flow ODE
  down to ``sigma_min`` with Heun's 2nd-order steps.
- We therefore run the machinery at **outer T=1**: one MultiDiffusion blend of *complete*
  samples. This is the cleanest faithful mapping and is deterministic in the input window
  (no fresh randomness inside Phi), so the sampler's tile-cache purity assumption holds.
- If the Axis-2 *seam* metric later degrades vs the toy, the principled next step is to
  raise outer T and re-noise between blends (a Restart/SDEdit refinement) — added only
  when that metric demands it, per the project's metric-driven methodology.

Phi returns fields in **m/s** (it denormalises the model's standardised output). Because
denormalisation is a per-channel affine map and the blend weights are shared across
channels, blending in m/s and blending-then-denormalising are identical — so this is
consistent with the machinery's weighted average.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

# Standalone-import-safe (spacetime.py imports edm_sigma_schedule from here on the cluster).
try:
    from .data import NormStats
    from .net import EDMPrecond
except ImportError:  # pragma: no cover - standalone script path
    from data import NormStats
    from net import EDMPrecond


def _window_seed(x: torch.Tensor) -> int:
    """A deterministic 63-bit seed derived from a window's content.

    Stochastic sampling needs per-step noise, but Phi must stay *deterministic in its input
    window* or the machinery's tile cache and Axis-2 exact revisit-determinism break. We
    therefore seed the churn RNG from the window's own bytes: identical tiles (cache hits,
    revisits) get identical churn -> identical output. (Same-device/dtype assumption holds
    within a run, which is all the cache + revisit guarantees require.)
    """
    b = x.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little") >> 1


def edm_sigma_schedule(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0,
                       *, device, dtype) -> torch.Tensor:
    """Karras EDM noise schedule: ``n`` levels sigma_max..sigma_min, with a trailing 0."""
    i = torch.arange(n, device=device, dtype=dtype)
    a = sigma_max ** (1.0 / rho)
    b = sigma_min ** (1.0 / rho)
    sig = (a + i / max(1, n - 1) * (b - a)) ** rho
    return torch.cat([sig, sig.new_zeros(1)])  # (n+1,)


class TrainedWindowDenoiser:
    """A trained EDM model presented as the InfiniteDiffusion Phi (WindowDenoiser).

    Args:
        ckpt_path: a checkpoint written by ``train.save_ckpt``.
        num_steps: internal EDM sampler steps (Heun). 18 is a sane default; fewer for smoke.
        device: torch device for the model + sampling.
        use_ema: load the EMA weights (recommended for inference) vs the raw model.
        s_churn: EDM stochasticity (Karras Alg. 2). 0 = the deterministic ODE (default).
            >0 injects+removes noise each step, counteracting the deterministic sampler's
            variance under-dispersion (low wind amplitude). Noise is seeded from the window
            (see ``_window_seed``) so Phi stays deterministic-in-window. ``s_min``/``s_max``
            gate which noise levels get churn; ``s_noise`` slightly inflates it (EDM default).
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        num_steps: int = 18,
        device: str | torch.device = "cpu",
        use_ema: bool = True,
        s_churn: float = 0.0,
        s_min: float = 0.05,
        s_max: float = 50.0,
        s_noise: float = 1.003,
    ) -> None:
        self.device = torch.device(device)
        # weights_only=False: our own checkpoint embeds numpy NormStats (trusted source).
        ck = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ck["cfg"]
        st = ck["stats"]
        self.stats = NormStats(st["mean_u"], st["std_u"], st["mean_v"], st["std_v"], st["levels"])
        self.n_levels = self.stats.n_levels
        self.n_channels = 2 * self.n_levels

        self.model = EDMPrecond(
            self.n_channels,
            sigma_data=cfg["sigma_data"],
            net_kwargs=dict(
                model_channels=cfg["model_channels"],
                channel_mult=tuple(cfg["channel_mult"]),
                num_res_blocks=cfg["num_res_blocks"],
                attn_resolutions=tuple(cfg["attn_resolutions"]),
            ),
        ).to(self.device)
        self.model.load_state_dict(ck["ema"] if use_ema else ck["model"])
        self.model.eval()

        self.num_steps = int(num_steps)
        self.sigma_min = self.model.sigma_min
        self.sigma_max = self.model.sigma_max
        self.s_churn = float(s_churn)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.s_noise = float(s_noise)
        self.step = int(ck.get("step", -1))

    @torch.no_grad()
    def _heun_sample(self, x_unit: torch.Tensor) -> torch.Tensor:
        """EDM Heun sampler. x_unit: (B,C,H,W) ~ N(0,1) -> clean (norm).

        ``s_churn == 0`` is the deterministic probability-flow ODE (Karras Alg. 1). With
        ``s_churn > 0`` it becomes the stochastic sampler (Alg. 2): each eligible step bumps
        sigma up by gamma and injects matching noise before the Heun step, which lets the
        trajectory recover marginal variance the ODE otherwise loses. The injected noise is
        drawn from a generator seeded by the window, so the map stays deterministic-in-window.
        """
        sig = edm_sigma_schedule(self.num_steps, self.sigma_min, self.sigma_max,
                                 device=x_unit.device, dtype=x_unit.dtype)
        x = x_unit * sig[0]
        gen = None
        if self.s_churn > 0:
            gen = torch.Generator(device="cpu").manual_seed(_window_seed(x_unit))
        gamma_base = min(self.s_churn / self.num_steps, 2.0 ** 0.5 - 1.0)
        for i in range(self.num_steps):
            s_cur, s_next = sig[i], sig[i + 1]
            sc = float(s_cur)
            gamma = gamma_base if (gen is not None and self.s_min <= sc <= self.s_max) else 0.0
            if gamma > 0:
                s_hat = s_cur * (1.0 + gamma)
                eps = torch.randn(x.shape, generator=gen, dtype=torch.float32)
                eps = eps.to(x.device, x.dtype) * self.s_noise
                x = x + (s_hat ** 2 - s_cur ** 2).clamp_min(0).sqrt() * eps
            else:
                s_hat = s_cur
            d = (x - self.model(x, s_hat.expand(x.shape[0]))) / s_hat
            x_next = x + (s_next - s_hat) * d
            if s_next > 0:  # 2nd-order correction
                d2 = (x_next - self.model(x_next, s_next.expand(x.shape[0]))) / s_next
                x_next = x + (s_next - s_hat) * 0.5 * (d + d2)
            x = x_next
        return x

    def __call__(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:  # noqa: ARG002
        """Phi: map a unit-noise window (C,H,W) to a denoised wind field (C,H,W) in m/s."""
        if x.shape[0] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[0]}")
        x = x.to(self.device)
        sample_norm = self._heun_sample(x[None])[0]      # (C,H,W) normalised
        return self.stats.denormalize(sample_norm).to(x.dtype)

    @torch.no_grad()
    def sample_crops(self, n: int, size: int, *, seed: int = 0) -> torch.Tensor:
        """Standalone (no machinery) sampler for the Axis-1 gate. Returns (n,C,size,size) m/s."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(n, self.n_channels, size, size, generator=g).to(self.device)
        norm = self._heun_sample(z)
        return self.stats.denormalize(norm)


def build_sampler(ckpt_path, *, num_steps=18, window=64, seed=0, device="cpu",
                  s_churn=0.0, s_min=0.05, s_max=50.0, s_noise=1.003, **kwargs):
    """Convenience: a Phase-3-ready InfiniteDiffusion driven by the trained denoiser.

    Outer ``T=1`` per the Phi mapping above (full internal sampler per window). ``s_churn>0``
    enables the stochastic internal sampler (window-seeded, so Axis-2 determinism survives).
    """
    from .sampler import InfiniteDiffusion

    phi = TrainedWindowDenoiser(ckpt_path, num_steps=num_steps, device=device,
                                s_churn=s_churn, s_min=s_min, s_max=s_max, s_noise=s_noise)
    return InfiniteDiffusion(phi, window=window, T=1, seed=seed, device=device, **kwargs)
