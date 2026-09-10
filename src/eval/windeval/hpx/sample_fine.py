"""Stage 2 sampler: the fine residual on the whole sphere by MultiDiffusion over patches.

Given the coarse frames of a block (hours 0, 6, 12 at nside 32, m/s, NEST) and a seed, produce
the fine field (13 hourly frames at nside 256, m/s, NEST). The diffusion state is one global
residual tensor; at every sampler step each coarse-aligned 64 px patch of the padded faces
(stride 32, i.e. 50% overlap) is denoised by the network with its own conditioning and the
estimates are blended back with a Kaiser-Bessel-derived window (cBottle's scheme, S7/S8).
Because patches are gathers from the NEST sphere through the padded index map, a patch that
straddles a face edge blends into the neighbouring face's pixels too; the fabricated corner
squares of the padding get zero weight. At the coarse hours (frames 0, 6, 12) the denoised
residual is projected so that its block means vanish, which makes block_mean(x) = C exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset import time_features  # noqa: E402
from layout import FaceLayout, coord_channels  # noqa: E402
from patches import PatchGeometry, SmoothLift  # noqa: E402
from sample import edm_sigma_schedule  # noqa: E402
from train_fine import build_model  # noqa: E402


def kbd_window(n: int, beta: float) -> np.ndarray:
    try:
        from scipy.signal.windows import kaiser_bessel_derived
        w = kaiser_bessel_derived(n, beta)
    except Exception:                                  # noqa: BLE001 - fallback: sine window (also Princen-Bradley)
        w = np.sin(np.pi * (np.arange(n) + 0.5) / n)
    return np.outer(w, w).astype(np.float32)


class HpxFineSampler:
    def __init__(self, ckpt_path: str | Path, layout_dir: str | Path, *, num_steps: int = 18, device: str = "cuda",
                 use_ema: bool = True, sigma_max: float | None = None, batch_patches: int = 32, stride: int | None = None,
                 window_beta: float = 6.0, guidance: float = 1.0, amp: bool = True,
                 sigma_min: float | None = None) -> None:
        """``sigma_min``/``sigma_max`` override the schedule ends (default: the network's; note the
        training floor ``train_sigma_min`` -- stepping below it asks the net to denoise where it was
        never trained, which leaves pixel-scale noise in the output)."""
        self.device = torch.device(device)
        ck = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        cfg = SimpleNamespace(**ck["cfg"]); self.cfg = cfg
        self.C, self.nside, self.nside_c = int(ck["n_channels"]), int(ck["nside"]), int(ck["nside_coarse"])
        self.tau, self.coarse_stride = int(cfg.n_frames), int(cfg.coarse_stride)
        self.n_int = (self.tau - 1) // self.coarse_stride
        self.mean = torch.as_tensor(ck["stats"]["mean"], device=self.device)[:, None]
        self.std = torch.as_tensor(ck["stats"]["std"], device=self.device)[:, None]
        self.scale = torch.as_tensor(np.asarray(ck["scale"], np.float32), device=self.device)[:, None]
        self.model = build_model(cfg, self.C).to(self.device)
        self.model.load_state_dict(ck["ema"] if use_ema else ck["model"]); self.model.eval()
        self.step = int(ck.get("step", -1))
        self.num_steps = int(num_steps)
        self.sigma_min = float(sigma_min) if sigma_min else self.model.sigma_min
        self.sigma_max = float(sigma_max) if sigma_max else self.model.sigma_max
        self.guidance, self.amp, self.batch = float(guidance), bool(amp), int(batch_patches)

        # ---- geometry: patch positions, gathers, blend weights
        self.geom = PatchGeometry(self.nside, self.nside_c, patch=int(cfg.patch), pad=int(cfg.pad), cache_dir=layout_dir)
        P, S, pad = self.geom.patch, self.geom.size, self.geom.pad
        self.stride = int(stride or P // 2)
        layout = FaceLayout.load(self.nside, layout_dir)
        self.coords = torch.from_numpy(layout.from_faces(coord_channels(self.nside, layout.perm)).astype(np.float32)).to(self.device)
        offs = list(range(0, S - P + 1, self.stride))
        if offs[-1] != S - P:
            offs.append(S - P)
        pos = [(f, i, j) for f in range(12) for i in offs for j in offs]
        fidx = np.stack([self.geom.fine_index(f, i, j) for f, i, j in pos])                 # (n, P, P)
        valid = np.ones((12, S, S), bool)
        for a in (slice(0, pad), slice(S - pad, S)):
            for b in (slice(0, pad), slice(S - pad, S)):
                valid[:, a, b] = False                                                     # padding corner squares
        vmask = np.stack([valid[f, i:i + P, j:j + P] for f, i, j in pos])                  # (n, P, P)
        win = kbd_window(P, window_beta)[None] * vmask                                     # (n, P, P)
        self.fidx = torch.from_numpy(fidx).to(self.device)
        self.cidx = self.fidx // (self.geom.ratio ** 2)
        self.lift_mode = getattr(cfg, "lift", "nearest")
        if self.lift_mode == "bilinear":
            sm = SmoothLift(self.nside, self.nside_c, layout_dir)
            self.sm_idx = torch.from_numpy(sm.idx).to(self.device)            # (npix, 4)
            self.sm_w = torch.from_numpy(sm.w).to(self.device)                # (npix, 4)
        self.win = torch.from_numpy(win.astype(np.float32)).to(self.device)
        wsum = torch.zeros(12 * self.nside * self.nside, device=self.device)
        wsum.index_add_(0, self.fidx.flatten(), self.win.flatten())
        assert float(wsum.min()) > 0, "some fine pixel receives no patch"
        self.wsum = wsum
        self.n_patches = len(pos)
        q = np.minimum(np.arange(self.tau) // self.coarse_stride, self.n_int - 1)
        self.q = torch.from_numpy(q).to(self.device)
        self.w = torch.from_numpy(((np.arange(self.tau) - q * self.coarse_stride) / self.coarse_stride).astype(np.float32)).to(self.device)

    # ---- helpers on NEST tensors ------------------------------------------------------
    def block_mean(self, x: torch.Tensor) -> torch.Tensor:          # (..., npix) -> (..., npix_c)
        return x.reshape(*x.shape[:-1], -1, self.geom.ratio ** 2).mean(-1)

    def lift(self, c: torch.Tensor) -> torch.Tensor:                # (..., npix_c) -> (..., npix), nearest (exact)
        return c.repeat_interleave(self.geom.ratio ** 2, dim=-1)

    def lift_cond(self, c: torch.Tensor, fi: torch.Tensor | None = None) -> torch.Tensor:
        """The lift the model was trained with, on the whole sphere (fi None) or gathered at fine
        indices ``fi`` of any shape: ``(..., npix_c)`` -> ``(..., npix)`` or ``(..., *fi.shape)``."""
        if self.lift_mode != "bilinear":
            return self.lift(c) if fi is None else c[..., fi // (self.geom.ratio ** 2)]
        idx = self.sm_idx if fi is None else self.sm_idx[fi]; w = self.sm_w if fi is None else self.sm_w[fi]
        out = torch.zeros(c.shape[:-1] + idx.shape[:-1], device=c.device, dtype=c.dtype)
        for k in range(4):
            out += c[..., idx[..., k]] * w[..., k]
        return out

    def normalise_coarse(self, coarse_ms: np.ndarray) -> torch.Tensor:
        c = torch.as_tensor(np.asarray(coarse_ms, np.float32), device=self.device)   # (n_int+1, C, npix_c)
        return (c - self.mean) / self.std

    def baseline(self, coarse_n: torch.Tensor) -> torch.Tensor:
        """Linear-in-time interpolation of the coarse frames, lifted: ``(τ, C, npix)`` normalised."""
        prev, nxt = coarse_n[self.q], coarse_n[self.q + 1]                                # (τ, C, npix_c)
        lin = (1 - self.w)[:, None, None] * prev + self.w[:, None, None] * nxt
        return self.lift_cond(lin)

    def tfeat(self, hours: np.ndarray) -> torch.Tensor:
        tf = np.concatenate([time_features(np.asarray(hours)), self.w.cpu().numpy()[:, None]], axis=1)
        return torch.from_numpy(tf.astype(np.float32)).to(self.device)                   # (τ, 7)

    def project(self, r: torch.Tensor, coarse_n: torch.Tensor) -> torch.Tensor:
        """Exact block-mean consistency at the coarse hours (frames 0, 6, 12): set the residual's
        block means to what makes block_mean(baseline + scale * r) == coarse. With the nearest lift
        the target is zero; with the bilinear lift it is (coarse - block_mean(baseline)) / scale,
        which the model has learned to produce -- zeroing it there was the 2026-09-10 bug."""
        r = r.clone(); base = self.baseline(coarse_n)
        for k, f in enumerate(range(0, self.tau, self.coarse_stride)):
            target = (coarse_n[k] - self.block_mean(base[f])) / self.scale                # (C, npix_c)
            r[f] = r[f] - self.lift(self.block_mean(r[f]) - target)
        return r

    # ---- one denoiser evaluation over the sphere ------------------------------------
    @torch.no_grad()
    def denoise(self, r: torch.Tensor, sigma: torch.Tensor, coarse_n: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
        """r: (τ, C, npix) noisy residual state -> denoised estimate, blended over patches."""
        tau, C, npix = r.shape
        out = torch.zeros(tau * C, npix, device=self.device)
        rflat = r.reshape(tau * C, npix)
        for a in range(0, self.n_patches, self.batch):
            fi, wn = self.fidx[a:a + self.batch], self.win[a:a + self.batch]
            b = fi.shape[0]
            xp = rflat[:, fi.reshape(-1)].reshape(tau, C, b, *fi.shape[1:]).permute(2, 0, 1, 3, 4)          # (b, τ, C, P, P)
            cc = self.lift_cond(coarse_n, fi).permute(2, 0, 1, 3, 4)                                          # (b, n_int+1, C, P, P)
            prev, nxt = cc[:, self.q], cc[:, self.q + 1]                                                    # (b, τ, C, P, P)
            lin = (1 - self.w)[None, :, None, None, None] * prev + self.w[None, :, None, None, None] * nxt
            cond = torch.cat([lin, prev, nxt], dim=2)                                                       # (b, τ, 3C, P, P)
            co = self.coords[:, fi.reshape(-1)].reshape(3, b, *fi.shape[1:]).permute(1, 0, 2, 3)            # (b, 3, P, P)
            tf = tfeat[None].expand(b, -1, -1)
            sig = sigma.expand(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
                D = self.model(xp, sig, cond=co, tfeat=tf, coarse=cond)
                if self.guidance != 1.0:
                    D0 = self.model(xp, sig, cond=co, tfeat=tf, coarse=torch.zeros_like(cond),
                                    coarse_mask=torch.zeros(b, device=self.device))
                    D = D0 + self.guidance * (D - D0)
            D = D.float() * wn[:, None, None]                                                               # window
            out.index_add_(1, fi.reshape(-1), D.permute(1, 2, 0, 3, 4).reshape(tau * C, -1))
        out = (out / self.wsum).reshape(tau, C, npix)
        return self.project(out)

    @torch.no_grad()
    def heun(self, r: torch.Tensor, coarse_n: torch.Tensor, tfeat: torch.Tensor, *, unit_noise: bool = True,
             log=print) -> torch.Tensor:
        sig = edm_sigma_schedule(self.num_steps, self.sigma_min, self.sigma_max, device=self.device, dtype=r.dtype)
        if unit_noise:
            r = r * sig[0]
        for i in range(self.num_steps):
            s_cur, s_next = sig[i], sig[i + 1]
            d = (r - self.denoise(r, s_cur, coarse_n, tfeat)) / s_cur
            r_next = r + (s_next - s_cur) * d
            if s_next > 0:
                d2 = (r_next - self.denoise(r_next, s_next, coarse_n, tfeat)) / s_next
                r_next = r + (s_next - s_cur) * 0.5 * (d + d2)
            r = r_next
            if log is not None and (i % 6 == 0 or i == self.num_steps - 1):
                log(f"[sample] step {i + 1}/{self.num_steps} sigma {float(s_cur):.3f}")
        return self.project(r, coarse_n)

    @torch.no_grad()
    def sample_block(self, coarse_ms: np.ndarray, hours: np.ndarray, *, seed: int = 0, log=print) -> np.ndarray:
        """coarse_ms: (n_int+1, C, npix_c) m/s NEST at hours 0/6/12; hours: (τ,) ARCO hour indices.
        Returns the fine field (τ, C, npix) in m/s, NEST, with block_mean = coarse at frames 0/6/12."""
        coarse_n = self.normalise_coarse(coarse_ms)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(self.tau, self.C, 12 * self.nside * self.nside, generator=g).to(self.device)
        r = self.heun(z, coarse_n, self.tfeat(hours), log=log)
        x = self.baseline(coarse_n) + self.scale[None] * r                                # (τ, C, npix) normalised
        return (x * self.std[None] + self.mean[None]).cpu().numpy()
