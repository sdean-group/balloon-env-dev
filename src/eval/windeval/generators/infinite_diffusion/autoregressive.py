"""M3 — autoregressive temporal rollout for the InfiniteDiffusion wind field.

The static EDM denoiser learns p(field); the *paired* checkpoint (trained with
``cfg.paired=True``, see train.py / data.WindPairDataset) learns the **transition**
``p(frame_{t+1} | frame_t)`` by taking the previous frame as clean conditioning channels
(``net.WindUNet(cond_channels=2L)``). This module runs that transition forward in time:

    frame_{t+1} = EDM_sample(noise | frame_t)

starting from a seed frame (a real ERA5 frame, or a sample from the static generator), so
the *spatial* realism is inherited from the seed/static model and M3 owns only the
**temporal dynamics** — which is exactly what the temporal benchmark scores.

Scope / relation to the machinery
----------------------------------
The window net is fully convolutional, so a rollout over a fixed (H, W) benchmark crop runs
the net directly — no spatial tiling needed for a *finite* crop. Unbounded-extent rollout
(tiling each frame through ``InfiniteDiffusion`` with per-tile conditioning) is the natural
extension once the model is trained; it reuses the frozen ``sampler.py`` per frame exactly as
``AdvectedField`` wraps it, and is deferred until the learned transition is validated. This
keeps M3's first cut honest about its cost: O(t) roll, drift accumulates (its known failure
mode — measured by ``metrics.temporal.drift``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Standalone-import-safe (mirrors train.py), in case M3 inference is ever run off-package.
try:
    from .data import NormStats
    from .net import EDMPrecond
    from .trained import edm_sigma_schedule
except ImportError:  # pragma: no cover - standalone script path
    from data import NormStats
    from net import EDMPrecond
    from trained import edm_sigma_schedule


class ConditionedDenoiser:
    """A paired (autoregressive) EDM checkpoint presented as a one-step transition sampler.

    Args:
        ckpt_path: a checkpoint from a ``cfg.paired=True`` run (its ``cfg.cond_channels`` /
            ``cfg.paired`` flag selects the conditioned net). Raises if the checkpoint is
            unconditional.
        num_steps: internal Heun steps per frame.
        device, use_ema: as in ``TrainedWindowDenoiser``.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        num_steps: int = 18,
        device: str | torch.device = "cpu",
        use_ema: bool = True,
    ) -> None:
        self.device = torch.device(device)
        ck = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ck["cfg"]
        st = ck["stats"]
        self.stats = NormStats(st["mean_u"], st["std_u"], st["mean_v"], st["std_v"], st["levels"])
        self.n_levels = self.stats.n_levels
        self.n_channels = 2 * self.n_levels

        cond_channels = int(cfg.get("cond_channels", self.n_channels if cfg.get("paired") else 0))
        if cond_channels == 0:
            raise ValueError("ConditionedDenoiser needs a PAIRED checkpoint (cond_channels>0); "
                             "this checkpoint is unconditional — use TrainedWindowDenoiser.")
        self.model = EDMPrecond(
            self.n_channels,
            sigma_data=cfg["sigma_data"],
            cond_channels=cond_channels,
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
        self.step = int(ck.get("step", -1))

    @torch.no_grad()
    def _heun_sample(self, x_unit: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Deterministic EDM Heun ODE (Karras Alg. 1), conditioned on ``cond`` (clean, norm).

        x_unit: (B,C,H,W) ~ N(0,1); cond: (B,C,H,W) the previous frame (normalised). The same
        condition is fed at every solver step (it is a clean input, not part of the diffused
        state). Returns the denoised next frame (normalised).
        """
        sig = edm_sigma_schedule(self.num_steps, self.sigma_min, self.sigma_max,
                                 device=x_unit.device, dtype=x_unit.dtype)
        x = x_unit * sig[0]
        for i in range(self.num_steps):
            s_cur, s_next = sig[i], sig[i + 1]
            d = (x - self.model(x, s_cur.expand(x.shape[0]), cond=cond)) / s_cur
            x_next = x + (s_next - s_cur) * d
            if s_next > 0:
                d2 = (x_next - self.model(x_next, s_next.expand(x.shape[0]), cond=cond)) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
            x = x_next
        return x

    @torch.no_grad()
    def step_frame(self, prev_norm: torch.Tensor, *, seed: int) -> torch.Tensor:
        """One transition: previous frame (C,H,W norm) -> next frame (C,H,W norm)."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(prev_norm.shape, generator=g).to(self.device, prev_norm.dtype)
        return self._heun_sample(z[None], prev_norm[None].to(self.device))[0]

    @torch.no_grad()
    def rollout(
        self,
        seed_uv: tuple[np.ndarray, np.ndarray],
        *,
        n_times: int,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the learned transition forward from an initial (u0, v0) frame.

        Args:
            seed_uv: ``(u0, v0)``, each ``(L, H, W)`` in m/s — the t=0 frame (e.g. a real ERA5
                frame or a static-generator sample). Spatial realism comes from this seed.
            n_times: total frames to emit (including the seed at index 0).
            seed: base RNG seed; per-frame noise is seeded ``seed + frame_index`` for
                reproducibility.

        Returns ``(us, vs)`` each ``(n_times, L, H, W)`` in m/s.
        """
        u0, v0 = np.asarray(seed_uv[0], np.float32), np.asarray(seed_uv[1], np.float32)
        L, H, W = u0.shape
        if L != self.n_levels:
            raise ValueError(f"seed has {L} levels, model has {self.n_levels}")
        f0 = np.stack([u0, v0], axis=1).reshape(self.n_channels, H, W)
        cur = self.stats.normalize(torch.from_numpy(f0).to(self.device))   # (C,H,W) norm

        frames_norm = [cur]
        for k in range(1, n_times):
            cur = self.step_frame(cur, seed=seed + k)
            frames_norm.append(cur)

        out = torch.stack(frames_norm, 0)                                  # (n,C,H,W) norm
        out = self.stats.denormalize(out).cpu().numpy()                    # m/s
        out = out.reshape(n_times, self.n_levels, 2, H, W)
        return out[:, :, 0], out[:, :, 1]                                  # us, vs
