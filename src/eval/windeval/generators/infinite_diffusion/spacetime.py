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
    from .data import CoordNorm, NormStats, coarsen, time_features
    from .net import AttnBlock, FourierEmbedding, ResBlock, _num_groups
    from .trained import edm_sigma_schedule
except ImportError:  # pragma: no cover - standalone script path
    from data import CoordNorm, NormStats, coarsen, time_features
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

    ``coarse_channels`` (Phase 5b, optional) is a PER-FRAME conditioning field — a
    horizontally block-meaned copy of the target — given at low resolution and bilinearly
    upsampled to (H, W) here, then concatenated. It is deliberately a separate argument
    from ``cond``: ``cond`` is τ-constant (coords) and broadcast, whereas the coarse field
    evolves frame to frame and so carries temporal information too. Bilinear (not nearest)
    upsampling avoids injecting artificial block edges the network would have to learn to
    ignore. Channel order into ``in_conv`` is [x, cond, coarse, (flag)]. With
    ``coarse_channels=0`` this class is numerically identical to the pre-Phase-5b model,
    so existing checkpoints keep loading.

    ``coarse_flag`` (Phase 5b-2) adds ONE constant plane carrying 1.0 when the coarse
    field is real and 0.0 when it has been dropped, which is what makes classifier-free
    guidance well posed: an all-zero coarse field in normalized units is a *legal* field
    (climatological mean everywhere), so without the flag the network cannot tell "no
    conditioning" from "a genuinely calm synoptic state" and the unconditional branch is
    contaminated. Enabled iff the run trains with ``coarse_dropout > 0``.
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
        coarse_channels: int = 0,
        coarse_flag: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.time_features = int(time_features)
        self.coarse_channels = int(coarse_channels)
        self.coarse_flag = bool(coarse_flag and coarse_channels)
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
        self.in_conv = nn.Conv2d(in_channels + self.cond_channels + self.coarse_channels
                                 + int(self.coarse_flag),
                                 model_channels, 3, padding=1)

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
                tfeat: torch.Tensor | None = None,
                coarse: torch.Tensor | None = None,
                coarse_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, τ, C, H, W); c_noise: (B,) per-block. Fold τ into batch for spatial ops,
        # repeat the noise embedding across τ, and run TemporalConv with the real τ.
        # cond: (B, cond_channels, H, W) clean coord channels, constant across τ.
        # tfeat: (B, τ, time_features) per-frame cyclic time harmonics.
        # coarse: (B, τ, coarse_channels, h, w) low-res per-frame field, upsampled here.
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
        if self.coarse_channels:
            if coarse is None:
                raise ValueError(f"net has coarse_channels={self.coarse_channels} "
                                 f"but coarse is None")
            cf = coarse.reshape(B * tau, self.coarse_channels, *coarse.shape[-2:])
            cf = torch.nn.functional.interpolate(cf, size=(H, W), mode="bilinear",
                                                 align_corners=False)
            xf = torch.cat([xf, cf], dim=1)
            if self.coarse_flag:
                m = (torch.ones(B, device=x.device, dtype=x.dtype)
                     if coarse_mask is None else coarse_mask.to(x.dtype).reshape(B))
                mf = m.repeat_interleave(tau)[:, None, None, None].expand(B * tau, 1, H, W)
                xf = torch.cat([xf, mf], dim=1)

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

    Residual parameterization (Phase 5b-2, ``coarse_residual``)
    ----------------------------------------------------------
    When on, the diffusion does NOT model the field. It models
    ``r = (x - U(coarse)) / coarse_scale`` — the part of the field bilinear upsampling of
    the conditioner does not already give away, rescaled to unit variance.

    The measurement that forces this (2026-07-30, held-out reference, 8x/2deg): bilinear
    upsampling of the coarse field already explains **97.4%** of the field variance; the
    residual has std **0.139** in normalized units. Trained on the raw field with
    ``sigma_data = 1`` and EDM's ``log sigma ~ N(-1.2, 1.2^2)``, ~74% of training steps
    draw ``sigma > 0.139`` — noise that completely swamps everything the conditioner has
    not already supplied, so at those steps the optimal denoiser is just "return the
    upsampled coarse field" and the step exerts no pressure toward fine structure. Three
    quarters of the training budget went to an already-solved problem.

    Rescaling the residual to unit variance (rather than lowering ``sigma_data``) is the
    equivalent fix with the smaller blast radius: the noise schedule, the loss weight,
    and the sampler's ``sigma_min/max`` all keep their tuned EDM values and simply now
    act on the band that is actually unknown. ``coarse_scale`` is measured from the
    training set at run start (:func:`data.measure_residual_scale`) and stored in the
    checkpoint, so sampling reproduces the training transform exactly.

    Side benefit that makes the experiment safer: "predict zero" is now *exactly* the
    ``coarse upsampled`` baseline, so the model starts at the control it must beat
    instead of having to rediscover it.
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
        coarse_channels: int = 0,
        coarse_residual: bool = False,
        coarse_scale: float = 1.0,
        coarse_flag: bool = False,
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
        self.coarse_channels = int(coarse_channels)
        self.coarse_residual = bool(coarse_residual and coarse_channels)
        self.coarse_scale = float(coarse_scale)
        self.coarse_flag = bool(coarse_flag and coarse_channels)
        self.net = SpaceTimeUNet(n_channels, cond_channels=self.cond_channels,
                                 time_features=self.time_features,
                                 coarse_channels=self.coarse_channels,
                                 coarse_flag=self.coarse_flag,
                                 **dict(net_kwargs or {}))

    # ---- residual space -------------------------------------------------------
    def coarse_base(self, coarse: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Bilinear lift of the coarse field to (H,W): the deterministic part of the field.

        This is *the same operator* :func:`benchmark._coarse_upsample_artifacts` scores as
        the `coarse upsampled` baseline, so "residual = 0" and "the baseline" are the same
        field by construction, not by approximation.
        """
        B, tau, C = coarse.shape[:3]
        cf = coarse.reshape(B * tau, C, *coarse.shape[-2:])
        cf = torch.nn.functional.interpolate(cf, size=hw, mode="bilinear", align_corners=False)
        return cf.reshape(B, tau, C, *hw)

    def to_residual(self, x0: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return (x0 - self.coarse_base(coarse, x0.shape[-2:])) / self.coarse_scale

    def from_residual(self, r: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return r * self.coarse_scale + self.coarse_base(coarse, r.shape[-2:])

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                cond: torch.Tensor | None = None,
                tfeat: torch.Tensor | None = None,
                coarse: torch.Tensor | None = None,
                coarse_mask: torch.Tensor | None = None) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4.0)
        # Conditioning inputs are clean (coord channels already ~[-1,1], harmonics in
        # [-1,1], the coarse field is in the same normalised units as the target); only
        # the noisy target gets the EDM c_in scaling — same convention as
        # net.EDMPrecond's M3 conditioning.
        F_x = self.net(c_in * x, c_noise, cond=cond, tfeat=tfeat, coarse=coarse,
                       coarse_mask=coarse_mask)
        return c_skip * x + c_out * F_x

    def loss(self, x0: torch.Tensor, *, cond: torch.Tensor | None = None,
             tfeat: torch.Tensor | None = None, coarse: torch.Tensor | None = None,
             P_mean: float = -1.2, P_std: float = 1.2,
             coarse_dropout: float = 0.0) -> torch.Tensor:
        B = x0.shape[0]
        mask = None
        if self.coarse_residual:
            # The TARGET is always the true residual — only the network's VIEW of the
            # conditioner is dropped below. So the unconditional branch learns
            # p(residual), marginalised over coarse, which is what CFG extrapolates from.
            x0 = self.to_residual(x0, coarse)
        if coarse_dropout > 0 and self.coarse_flag:
            keep = (torch.rand(B, device=x0.device) >= coarse_dropout).to(x0.dtype)
            coarse = coarse * keep.reshape(B, 1, 1, 1, 1)
            mask = keep
        rnd = torch.randn(B, device=x0.device)
        sigma = (rnd * P_std + P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x0) * sigma.reshape(-1, 1, 1, 1, 1)
        D = self(x0 + n, sigma, cond=cond, tfeat=tfeat, coarse=coarse, coarse_mask=mask)
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

    # Defaults for every sampler knob (a sampler built without __init__, e.g. a bare
    # ``object.__new__(SpaceTimeSampler)`` in tests, is the deterministic ODE sampler with
    # no guidance and no coarse conditioning — exactly the pre-Phase-5b behaviour).
    s_churn: float = 0.0
    s_min: float = 0.05
    s_max: float = 50.0
    s_noise: float = 1.003
    guidance: float = 1.0
    coarse_project: bool = True
    coarse_factor: int = 0
    coarse_flag: bool = False
    coarse_residual: bool = False
    coarse_scale: float = 1.0

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
        guidance: float = 1.0,
        coarse_project: bool = True,
    ) -> None:
        self.device = torch.device(device)
        ck = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        cfg = ck["cfg"]
        self.cfg = dict(cfg)
        st = ck["stats"]
        self.stats = NormStats(st["mean_u"], st["std_u"], st["mean_v"], st["std_v"], st["levels"])
        self.n_levels = self.stats.n_levels
        self.n_channels = 2 * self.n_levels
        self.tau = int(cfg.get("tau") or cfg["n_frames"])
        self.conditional = bool(cfg.get("conditional", False))
        self.coord_norm = CoordNorm(**ck["coord_norm"]) if self.conditional else None
        cond_ch = 2 if self.conditional else 0
        n_tfeat = int(cfg.get("n_time_features", 6)) if self.conditional else 0
        # coarse_factor absent/0 (every pre-Phase-5b checkpoint) ⇒ coarse_ch 0 ⇒ the model
        # is byte-identical to before, so old checkpoints keep loading unchanged.
        self.coarse_factor = int(cfg.get("coarse_factor", 0) or 0)
        coarse_ch = self.n_channels if self.coarse_factor else 0
        # Phase 5b-2 fields; absent in every earlier checkpoint ⇒ False/1.0 ⇒ the older
        # coarse model and every pre-Phase-5b model load and sample exactly as before.
        self.coarse_residual = bool(cfg.get("coarse_residual", False)) and bool(coarse_ch)
        self.coarse_scale = float(cfg.get("coarse_scale", 1.0) or 1.0)
        self.coarse_flag = bool(cfg.get("coarse_dropout", 0.0)) and bool(coarse_ch)

        self.model = EDMPrecondSpaceTime(
            self.n_channels, tau=self.tau, sigma_data=cfg["sigma_data"],
            cond_channels=cond_ch, time_features=n_tfeat, coarse_channels=coarse_ch,
            coarse_residual=self.coarse_residual, coarse_scale=self.coarse_scale,
            coarse_flag=self.coarse_flag,
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
        self.s_churn = float(s_churn)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.s_noise = float(s_noise)
        self.guidance = float(guidance)
        self.coarse_project = bool(coarse_project)
        self.step = int(ck.get("step", -1))
        if self.guidance != 1.0 and not self.coarse_flag:
            raise ValueError("guidance != 1 requires a checkpoint trained with "
                             "coarse_dropout > 0 (no unconditional branch otherwise)")

    def _denoise(self, x, s, *, cond, tfeat, coarse):
        """One denoiser evaluation, with CFG and the coarse consistency projection.

        **Classifier-free guidance** (``guidance`` w): D = D_uncond + w (D_cond - D_uncond),
        the standard sharpening extrapolation. Two forward passes per call when w != 1.
        The unconditional branch zeroes the coarse field AND lowers the flag plane — the
        flag is what makes "dropped" distinguishable from "calm" (see SpaceTimeUNet).

        **Consistency projection** (``coarse_project``, ILVR/replacement): the block-mean
        of the denoised field is forced to equal the conditioner exactly,

            full <- full + U_nn(coarse - blockmean(full))

        with a nearest lift (the adjoint of block-mean, so it moves each cell by a
        constant and cannot manufacture sub-cell structure). This turns "reproduce the
        synoptic state you were handed" from a soft learned objective into an exact
        constraint, freeing capacity for the residual band and removing the bilinear
        lift's own cell-mean bias. Applied to the DENOISED estimate inside each Heun
        stage rather than to the noisy iterate, which is the numerically stable variant.
        """
        B = x.shape[0]
        if self.guidance != 1.0:
            ones = torch.ones(B, device=x.device, dtype=x.dtype)
            d_c = self.model(x, s, cond=cond, tfeat=tfeat, coarse=coarse, coarse_mask=ones)
            d_u = self.model(x, s, cond=cond, tfeat=tfeat, coarse=torch.zeros_like(coarse),
                             coarse_mask=torch.zeros_like(ones))
            d = d_u + self.guidance * (d_c - d_u)
        elif coarse is None and not self.coarse_flag:
            d = self.model(x, s, cond=cond, tfeat=tfeat)      # no coarse pathway at all
        else:
            mask = (torch.ones(B, device=x.device, dtype=x.dtype)
                    if self.coarse_flag else None)
            d = self.model(x, s, cond=cond, tfeat=tfeat, coarse=coarse, coarse_mask=mask)
        if self.coarse_project and self.coarse_factor and coarse is not None:
            full = self.model.from_residual(d, coarse) if self.coarse_residual else d
            f = self.coarse_factor
            resid = coarse - coarsen(full.reshape(-1, *full.shape[2:]), f).reshape(coarse.shape)
            lift = resid.repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1)
            full = full + lift
            d = ((full - self.model.coarse_base(coarse, full.shape[-2:]))
                 / self.model.coarse_scale) if self.coarse_residual else full
        return d

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
                      tfeat: torch.Tensor | None = None,
                      coarse: torch.Tensor | None = None,
                      churn_seed: int = 0) -> torch.Tensor:
        """Advance steps ``[start_step, end_step)`` of the EDM Heun trajectory over a block.

        The segment form is what the 4-D InfiniteDiffusion wrapper (``spacetime_infinite``)
        uses for its T=2 split: run to ``split_step``, blend overlapping intermediate states,
        then continue from ``split_step`` on the blended field. ``unit_noise=True`` scales a
        unit-Gaussian input to ``sigma_max`` (only valid from step 0); a continuation passes
        the intermediate noisy state as-is.

        ``s_churn == 0`` (default) is the deterministic probability-flow ODE — bitwise-
        identical to the pre-churn implementation. With ``s_churn > 0`` it is Karras Alg. 2
        (same machinery as trained.TrainedWindowDenoiser): eligible steps bump sigma by gamma
        and inject matching noise, recovering marginal variance the ODE loses. The churn
        stream is seeded from ``churn_seed`` (xor'd so it never collides with the init-noise
        stream of the same seed) — deterministic per block and per segment.

        Every denoiser evaluation goes through :meth:`_denoise`, which applies classifier-free
        guidance and the coarse consistency projection when the checkpoint is coarse-conditioned.
        """
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
        gen = None
        if self.s_churn > 0:
            gen = torch.Generator(device="cpu").manual_seed(int(churn_seed) ^ 0x5F5E1)
        gamma_base = min(self.s_churn / self.num_steps, 2.0 ** 0.5 - 1.0)
        for i in range(start_step, end_step):
            s_cur, s_next = sig[i], sig[i + 1]
            gamma = (gamma_base if (gen is not None and self.s_min <= float(s_cur) <= self.s_max)
                     else 0.0)
            if gamma > 0:
                s_hat = s_cur * (1.0 + gamma)
                eps = torch.randn(x.shape, generator=gen, dtype=torch.float32)
                eps = eps.to(x.device, x.dtype) * self.s_noise
                x = x + (s_hat ** 2 - s_cur ** 2).clamp_min(0).sqrt() * eps
            else:
                s_hat = s_cur
            d = (x - self._denoise(x, s_hat.expand(B), cond=cond, tfeat=tfeat,
                                   coarse=coarse)) / s_hat
            x_next = x + (s_next - s_hat) * d
            if s_next > 0:
                d2 = (x_next - self._denoise(x_next, s_next.expand(B), cond=cond,
                                             tfeat=tfeat, coarse=coarse)) / s_next
                x_next = x + (s_next - s_hat) * 0.5 * (d + d2)
            x = x_next
        return x

    @torch.no_grad()
    def _heun_block(self, x_unit: torch.Tensor,
                    cond: torch.Tensor | None = None,
                    tfeat: torch.Tensor | None = None,
                    coarse: torch.Tensor | None = None,
                    churn_seed: int = 0) -> torch.Tensor:
        """Full trajectory (unit noise -> clean block) = one segment over every step."""
        return self._heun_segment(
            x_unit,
            start_step=0,
            end_step=self.num_steps,
            unit_noise=True,
            cond=cond,
            tfeat=tfeat,
            coarse=coarse,
            churn_seed=churn_seed,
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
    def coarse_from_field(self, us: np.ndarray, vs: np.ndarray) -> torch.Tensor:
        """Build the coarse conditioning tensor from a full-resolution (τ,L,H,W) field.

        Applies the SAME operator as training — normalise, then block-mean by
        ``coarse_factor`` — so what the model sees at inference matches what it was
        trained on. Returns ``(1, τ, 2L, H/f, W/f)``.
        """
        if not self.coarse_factor:
            raise ValueError("checkpoint is not coarse-conditioned")
        f = np.stack([us, vs], axis=2).reshape(self.tau, self.n_channels, *us.shape[-2:])
        x = self.stats.normalize(torch.from_numpy(f.astype(np.float32)))
        return coarsen(x, self.coarse_factor)[None].to(self.device)

    @torch.no_grad()
    def sample_block(self, hw: tuple[int, int], *, seed: int = 0,
                     lat=None, lon=None, times=None,
                     coarse=None) -> tuple[np.ndarray, np.ndarray]:
        """Generate one τ-frame block. Returns ``(us, vs)`` each ``(τ, L, H, W)`` in m/s.

        Coarse-conditioned checkpoints additionally require ``coarse`` — either the
        ``(1,τ,2L,h,w)`` tensor from :meth:`coarse_from_field` or a ``(us, vs)`` pair of
        full-resolution arrays to coarsen here.
        """
        H, W = hw
        cond, tfeat = self._condition(hw, lat, lon, times)
        if self.coarse_factor:
            if coarse is None:
                raise ValueError("coarse-conditioned checkpoint: sample_block needs coarse=")
            if isinstance(coarse, tuple):
                coarse = self.coarse_from_field(*coarse)
            coarse = coarse.to(self.device)
            exp = (1, self.tau, self.n_channels, H // self.coarse_factor,
                   W // self.coarse_factor)
            if tuple(coarse.shape) != exp:
                raise ValueError(f"expected coarse {exp}, got {tuple(coarse.shape)}")
        elif coarse is not None:
            raise ValueError("checkpoint is not coarse-conditioned but coarse= was given")
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(1, self.tau, self.n_channels, H, W, generator=g).to(self.device)
        block = self._heun_block(z, cond=cond, tfeat=tfeat, coarse=coarse,
                                 churn_seed=seed)  # (1,τ,C,H,W) norm (residual if enabled)
        if self.coarse_residual:
            block = self.model.from_residual(block, coarse)
        block = self.stats.denormalize(block.reshape(self.tau, self.n_channels, H, W))
        block = block.reshape(self.tau, self.n_levels, 2, H, W).cpu().numpy()
        return block[:, :, 0], block[:, :, 1]
