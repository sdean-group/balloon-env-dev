"""Sampling from a Stage 1 checkpoint: whole-sphere coarse wind blocks in m/s, NEST order.

Deterministic EDM Heun ODE (18 steps by default), exposed as ``heun_segment(start, end)`` so
that a later time-tiling wrapper can split the trajectory the way the regional 4-D wrapper
does. One block = ``n_frames`` global frames at ``stride_hours`` spacing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import time_features  # noqa: E402
from layout import FaceLayout, coord_channels  # noqa: E402
from net import EDMPrecondHpx  # noqa: E402


def edm_sigma_schedule(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0, *, device, dtype):
    i = torch.arange(n, device=device, dtype=dtype)
    a, b = sigma_max ** (1.0 / rho), sigma_min ** (1.0 / rho)
    sig = (a + i / max(1, n - 1) * (b - a)) ** rho
    return torch.cat([sig, sig.new_zeros(1)])


class HpxSampler:
    def __init__(self, ckpt_path: str | Path, layout_dir: str | Path, *, num_steps: int = 18,
                 device: str = "cpu", use_ema: bool = True) -> None:
        self.device = torch.device(device)
        ck = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ck["cfg"]
        self.cfg, self.tau, self.nside = cfg, int(cfg["n_frames"]), int(ck["nside"])
        self.C = int(ck["n_channels"])
        self.stride_hours = int(cfg["stride_hours"])
        self.mean = torch.as_tensor(ck["stats"]["mean"], device=self.device)[:, None]
        self.std = torch.as_tensor(ck["stats"]["std"], device=self.device)[:, None]
        self.slow_mean, self.slow_std = ck.get("slow_norm", (0.0, 1.0))
        self.layout = FaceLayout.load(self.nside, layout_dir)
        self.coords = torch.from_numpy(coord_channels(self.nside, self.layout.perm))[None].to(self.device)
        self.model = EDMPrecondHpx(self.C, tau=self.tau, sigma_data=cfg["sigma_data"],
                                   net_kwargs=dict(model_channels=cfg["model_channels"], channel_mult=tuple(cfg["channel_mult"]),
                                                   num_res_blocks=cfg["num_res_blocks"], attn_resolutions=tuple(cfg["attn_resolutions"]),
                                                   temporal_kernel=cfg["temporal_kernel"], slow_features=2)).to(self.device)
        self.model.load_state_dict(ck["ema"] if use_ema else ck["model"]); self.model.eval()
        self.num_steps, self.sigma_min, self.sigma_max = int(num_steps), self.model.sigma_min, self.model.sigma_max
        self.step = int(ck.get("step", -1))

    def conditioning(self, hours: np.ndarray, slow_value: float | None = None):
        """hours: ARCO hour indices (τ,). slow_value: raw QBO-style index in m/s, or None (absent)."""
        tfeat = torch.from_numpy(time_features(np.asarray(hours)))[None].to(self.device)
        slow = torch.zeros(1, self.tau, 2, device=self.device)
        if slow_value is not None:
            slow[..., 0] = (float(slow_value) - self.slow_mean) / self.slow_std; slow[..., 1] = 1.0
        return tfeat, slow

    @torch.no_grad()
    def heun_segment(self, x: torch.Tensor, *, start_step: int, end_step: int, unit_noise: bool,
                     tfeat: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        sig = edm_sigma_schedule(self.num_steps, self.sigma_min, self.sigma_max, device=x.device, dtype=x.dtype)
        if unit_noise:
            x = x * sig[0]
        B = x.shape[0]
        cond = self.coords.expand(B, -1, -1, -1, -1)
        for i in range(start_step, end_step):
            s_cur, s_next = sig[i], sig[i + 1]
            d = (x - self.model(x, s_cur.expand(B), cond, tfeat, slow)) / s_cur
            x_next = x + (s_next - s_cur) * d
            if s_next > 0:
                d2 = (x_next - self.model(x_next, s_next.expand(B), cond, tfeat, slow)) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
            x = x_next
        return x

    @torch.no_grad()
    def sample_block(self, hours: np.ndarray, *, seed: int = 0, slow_value: float | None = None) -> np.ndarray:
        """One global block ``(τ, C, npix)`` in m/s, NEST order, for the given hour stamps."""
        tfeat, slow = self.conditioning(hours, slow_value)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(1, self.tau, self.C, 12, self.nside, self.nside, generator=g).to(self.device)
        x = self.heun_segment(z, start_step=0, end_step=self.num_steps, unit_noise=True, tfeat=tfeat, slow=slow)[0]
        x = self.layout.from_faces(x)                                    # (τ, C, npix) normalised
        return (x * self.std + self.mean).cpu().numpy()
