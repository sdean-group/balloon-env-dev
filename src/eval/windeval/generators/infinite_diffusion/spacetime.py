"""M2 — joint spacetime denoiser for the InfiniteDiffusion wind field.

Where M3 (autoregressive) learns a *transition* p(frame_{t+1} | frame_t) and rolls it
forward, M2 denoises a whole **H×W×τ block** of consecutive frames jointly — the diffusion
target is the entire short space-time chunk, with a single EDM noise level per block (as in
video diffusion). Long sequences then come from **tiling in time** the same way
InfiniteDiffusion tiles in space (overlap-blend successive blocks), which is what preserves
the O(1) / seamless / lazy guarantees *in time* — M2's reason to exist. That temporal-tiling
layer wraps the frozen ``sampler.py`` (mirroring ``AdvectedField``) and is built only once the
block denoiser is trained; this module is the **block denoiser + its dataset + a block
sampler**, validated on a single block first (exactly how the static spatial model was).

Factorization (NOT full 3D conv)
--------------------------------
The three axes are wildly anisotropic (horizontal ~28 km, vertical ~380 m carried as
channels, time ~1 h), so an isotropic 3D kernel is wrong. We factorize: the proven 2D spatial
``ResBlock`` runs per-frame (fold ``B·τ`` into the batch), and a lightweight ``TemporalConv``
(1D conv along τ, zero-init residual so training starts at the per-frame static model) mixes
frames at each resolution. This reuses the spatial machinery and adds the minimum temporal
coupling — one trick at a time, the project's methodology.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Work both as a package module and as a standalone cluster script (train.py imports this);
# the absolute fallback avoids importing src/eval/__init__ (the jax/gym stack). Mirrors train.py.
try:
    from .data import CoordNorm, NormStats, time_features
    from .net import AttnBlock, FourierEmbedding, ResBlock, _num_groups
    from .trained import edm_sigma_schedule
except ImportError:  # pragma: no cover - standalone script path
    from data import CoordNorm, NormStats, time_features
    from net import AttnBlock, FourierEmbedding, ResBlock, _num_groups
    from trained import edm_sigma_schedule


class TemporalConv(nn.Module):
    """Mix τ frames at each spatial location with a 1D conv along time (zero-init residual).

    Input/return ``(B*τ, C, H, W)``; the residual starts at zero so an untrained block model
    is exactly the per-frame static model and temporal coupling is *learned* on top.
    """

    def __init__(self, ch: int, *, kernel: int = 3, n_groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(ch, n_groups), ch)
        self.conv = nn.Conv1d(ch, ch, kernel, padding=kernel // 2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, tau: int) -> torch.Tensor:
        BT, C, H, W = x.shape
        B = BT // tau
        h = self.norm(x)
        # (B*τ,C,H,W) -> (B*H*W, C, τ): conv mixes along the time axis per (b, h, w, channel)
        h = h.reshape(B, tau, C, H, W).permute(0, 3, 4, 2, 1).reshape(B * H * W, C, tau)
        h = self.conv(torch.nn.functional.silu(h))
        h = h.reshape(B, H, W, C, tau).permute(0, 4, 3, 1, 2).reshape(BT, C, H, W)
        return x + h


class SpaceTimeUNet(nn.Module):
    """Factorized space-time U-Net F over (B, τ, C, H, W). Predicts the EDM-preconditioned residual.

    Spatial path mirrors :class:`net.WindUNet` (ResBlocks + coarse attention); a ``TemporalConv``
    follows each spatial ResBlock to couple the τ frames. The noise embedding is per-block,
    broadcast across τ. Output channels == ``in_channels`` (= 2L per frame).

    Conditioning (Phase 5)
    ----------------------
    ``cond_channels`` clean per-pixel channels (normalized lat/lon — constant across τ) are
    concatenated to the input before ``in_conv``, exactly like M3's previous-frame channels.
    ``time_features`` per-frame scalars (cyclic harmonics) enter through a **zero-init**
    linear added to the per-frame noise embedding — an untrained conditional path is
    exactly the unconditional model, matching the TemporalConv zero-init pattern.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        model_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (4,),
        temporal_kernel: int = 3,
        cond_channels: int = 0,
        time_features: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.time_features = int(time_features)
        self.num_res_blocks = int(num_res_blocks)
        emb_ch = model_channels * 4

        self.map_noise = FourierEmbedding(model_channels)
        self.map_layer = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )
        if self.time_features:
            self.map_tfeat = nn.Linear(self.time_features, emb_ch)
            nn.init.zeros_(self.map_tfeat.weight)
            nn.init.zeros_(self.map_tfeat.bias)
        self.in_conv = nn.Conv2d(in_channels + self.cond_channels, model_channels, 3, padding=1)

        # encoder
        self.down = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_temporal = nn.ModuleList()
        self.downsample = nn.ModuleList()
        chs = [model_channels]
        ch = model_channels
        ds = 1
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, out_ch, emb_ch))
                self.down_attn.append(AttnBlock(out_ch) if ds in attn_resolutions else nn.Identity())
                self.down_temporal.append(TemporalConv(out_ch, kernel=temporal_kernel))
                ch = out_ch
                chs.append(ch)
            if i != len(channel_mult) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chs.append(ch)
                ds *= 2
            else:
                self.downsample.append(None)

        # bottleneck
        self.mid1 = ResBlock(ch, ch, emb_ch)
        self.mid_attn = AttnBlock(ch)
        self.mid_temporal = TemporalConv(ch, kernel=temporal_kernel)
        self.mid2 = ResBlock(ch, ch, emb_ch)

        # decoder
        self.up = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.up_temporal = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(ch + chs.pop(), out_ch, emb_ch))
                self.up_attn.append(AttnBlock(out_ch) if ds in attn_resolutions else nn.Identity())
                self.up_temporal.append(TemporalConv(out_ch, kernel=temporal_kernel))
                ch = out_ch
            if i != 0:
                self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                ds //= 2
            else:
                self.upsample.append(None)

        self.out_norm = nn.GroupNorm(_num_groups(ch, 32), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor,
                cond: torch.Tensor | None = None,
                tfeat: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, τ, C, H, W); c_noise: (B,) per-block. Fold τ into batch for spatial ops,
        # repeat the noise embedding across τ, and run TemporalConv with the real τ.
        # cond: (B, cond_channels, H, W) clean coord channels, constant across τ.
        # tfeat: (B, τ, time_features) per-frame cyclic time harmonics.
        B, tau, C, H, W = x.shape
        emb = self.map_layer(self.map_noise(c_noise))            # (B, emb)
        emb = emb.repeat_interleave(tau, dim=0)                  # (B*τ, emb)
        if self.time_features:
            if tfeat is None:
                raise ValueError(f"net has time_features={self.time_features} but tfeat is None")
            emb = emb + self.map_tfeat(tfeat.reshape(B * tau, -1))
        xf = x.reshape(B * tau, C, H, W)
        if self.cond_channels:
            if cond is None:
                raise ValueError(f"net has cond_channels={self.cond_channels} but cond is None")
            xf = torch.cat([xf, cond.repeat_interleave(tau, dim=0)], dim=1)

        h = self.in_conv(xf)
        hs = [h]
        bi = 0
        for i in range(len(self.downsample)):
            for _ in range(self.num_res_blocks):
                h = self.down[bi](h, emb)
                h = self.down_attn[bi](h)
                h = self.down_temporal[bi](h, tau)
                hs.append(h)
                bi += 1
            if self.downsample[i] is not None:
                h = self.downsample[i](h)
                hs.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_temporal(h, tau)
        h = self.mid2(h, emb)

        ui = 0
        for i in range(len(self.upsample)):
            for _ in range(self.num_res_blocks + 1):
                h = self.up[ui](torch.cat([h, hs.pop()], dim=1), emb)
                h = self.up_attn[ui](h)
                h = self.up_temporal[ui](h, tau)
                ui += 1
            if self.upsample[i] is not None:
                h = self.upsample[i](h)

        out = self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
        return out.reshape(B, tau, C, H, W)


class EDMPrecondSpaceTime(nn.Module):
    """Karras EDM preconditioning around :class:`SpaceTimeUNet`, on 5D blocks (B,τ,C,H,W).

    Identical math to :class:`net.EDMPrecond`; one sigma per block, broadcast over τ (and C,H,W).
    """

    def __init__(
        self,
        n_channels: int,
        *,
        tau: int,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        cond_channels: int = 0,
        time_features: int = 0,
        net_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.tau = int(tau)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.cond_channels = int(cond_channels)
        self.time_features = int(time_features)
        self.net = SpaceTimeUNet(n_channels, cond_channels=self.cond_channels,
                                 time_features=self.time_features, **dict(net_kwargs or {}))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                cond: torch.Tensor | None = None,
                tfeat: torch.Tensor | None = None) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4.0)
        # Conditioning inputs are clean (coord channels already ~[-1,1], harmonics in
        # [-1,1]); only the noisy target gets the EDM c_in scaling — same convention as
        # net.EDMPrecond's M3 conditioning.
        F_x = self.net(c_in * x, c_noise, cond=cond, tfeat=tfeat)
        return c_skip * x + c_out * F_x

    def loss(self, x0: torch.Tensor, *, cond: torch.Tensor | None = None,
             tfeat: torch.Tensor | None = None,
             P_mean: float = -1.2, P_std: float = 1.2) -> torch.Tensor:
        rnd = torch.randn(x0.shape[0], device=x0.device)
        sigma = (rnd * P_std + P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x0) * sigma.reshape(-1, 1, 1, 1, 1)
        D = self(x0 + n, sigma, cond=cond, tfeat=tfeat)
        return (weight.reshape(-1, 1, 1, 1, 1) * (D - x0) ** 2).mean()


class SpaceTimeSampler:
    """A trained spacetime block model presented as a block sampler (Heun ODE).

    ``.sample_block((H,W), seed)`` generates one τ-frame block in m/s; temporal tiling for
    long sequences is the deferred extension (wraps the frozen machinery in time).

    Conditional checkpoints (``cfg["conditional"]``) additionally require **where and
    when**: ``sample_block(..., lat=, lon=, times=)`` with per-pixel coordinate vectors
    (degrees; ``lat`` length H, ``lon`` length W) and a ``(τ,)`` datetime64 array. The
    checkpoint's stored :class:`data.CoordNorm` normalizes coordinates identically to
    training (including the 0–360 vs ±180 longitude branch guard).
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
        self.tau = int(cfg.get("tau") or cfg["n_frames"])
        self.conditional = bool(cfg.get("conditional", False))
        self.coord_norm = CoordNorm(**ck["coord_norm"]) if self.conditional else None
        cond_ch = 2 if self.conditional else 0
        n_tfeat = int(cfg.get("n_time_features", 6)) if self.conditional else 0

        self.model = EDMPrecondSpaceTime(
            self.n_channels, tau=self.tau, sigma_data=cfg["sigma_data"],
            cond_channels=cond_ch, time_features=n_tfeat,
            net_kwargs=dict(
                model_channels=cfg["model_channels"],
                channel_mult=tuple(cfg["channel_mult"]),
                num_res_blocks=cfg["num_res_blocks"],
                attn_resolutions=tuple(cfg["attn_resolutions"]),
                temporal_kernel=int(cfg.get("temporal_kernel", 3)),
            ),
        ).to(self.device)
        self.model.load_state_dict(ck["ema"] if use_ema else ck["model"])
        self.model.eval()
        self.num_steps = int(num_steps)
        self.sigma_min = self.model.sigma_min
        self.sigma_max = self.model.sigma_max
        self.step = int(ck.get("step", -1))

    def sigma_schedule(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        return edm_sigma_schedule(
            self.num_steps,
            self.sigma_min,
            self.sigma_max,
            device=device or self.device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _heun_segment(self, x: torch.Tensor, *, start_step: int, end_step: int,
                      unit_noise: bool = False,
                      cond: torch.Tensor | None = None,
                      tfeat: torch.Tensor | None = None) -> torch.Tensor:
        """Advance a contiguous segment of the deterministic EDM trajectory."""
        if not 0 <= start_step < end_step <= self.num_steps:
            raise ValueError(
                f"expected 0 <= start_step < end_step <= {self.num_steps}; "
                f"got {start_step}, {end_step}"
            )
        sig = self.sigma_schedule(device=x.device, dtype=x.dtype)
        if unit_noise:
            if start_step != 0:
                raise ValueError("unit_noise is only valid for a segment starting at step 0")
            x = x * sig[0]
        B = x.shape[0]
        for i in range(start_step, end_step):
            s_cur, s_next = sig[i], sig[i + 1]
            d = (x - self.model(x, s_cur.expand(B), cond=cond, tfeat=tfeat)) / s_cur
            x_next = x + (s_next - s_cur) * d
            if s_next > 0:
                d2 = (x_next - self.model(x_next, s_next.expand(B), cond=cond, tfeat=tfeat)) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
            x = x_next
        return x

    @torch.no_grad()
    def _heun_block(self, x_unit: torch.Tensor,
                    cond: torch.Tensor | None = None,
                    tfeat: torch.Tensor | None = None) -> torch.Tensor:
        return self._heun_segment(
            x_unit,
            start_step=0,
            end_step=self.num_steps,
            unit_noise=True,
            cond=cond,
            tfeat=tfeat,
        )

    def _condition(self, hw: tuple[int, int], lat, lon, times
                   ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.conditional:
            return None, None
        if lat is None or lon is None or times is None:
            raise ValueError("conditional checkpoint: sample_block needs lat=, lon=, times=")
        H, W = hw
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        times = np.asarray(times)
        if lat.shape != (H,) or lon.shape != (W,) or times.shape != (self.tau,):
            raise ValueError(f"expected lat ({H},), lon ({W},), times ({self.tau},); "
                             f"got {lat.shape}, {lon.shape}, {times.shape}")
        cond = torch.from_numpy(self.coord_norm.channels(lat, lon))[None].to(self.device)
        tfeat = torch.from_numpy(time_features(times))[None].to(self.device)
        return cond, tfeat

    @torch.no_grad()
    def sample_block(self, hw: tuple[int, int], *, seed: int = 0,
                     lat=None, lon=None, times=None) -> tuple[np.ndarray, np.ndarray]:
        """Generate one τ-frame block. Returns ``(us, vs)`` each ``(τ, L, H, W)`` in m/s."""
        H, W = hw
        cond, tfeat = self._condition(hw, lat, lon, times)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(1, self.tau, self.n_channels, H, W, generator=g).to(self.device)
        block = self._heun_block(z, cond=cond, tfeat=tfeat)          # (1,τ,C,H,W) norm
        block = self.stats.denormalize(block.reshape(self.tau, self.n_channels, H, W))
        block = block.reshape(self.tau, self.n_levels, 2, H, W).cpu().numpy()
        return block[:, :, 0], block[:, :, 1]
