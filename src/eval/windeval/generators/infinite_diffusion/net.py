"""The window-denoiser network: an EDM-preconditioned U-Net over (u,v) levels.

This is the Phase-2 *model* half — the trainable function that learns the wind-field
distribution on fixed-size crops. It is deliberately a **boring EDM baseline** (Karras
et al., arXiv 2206.00364): a plain U-Net wrapped in EDM preconditioning, in pixel space,
with the vertical levels carried as channels. No domain tricks yet — those get added only
when an Axis-1 metric demands one (see the progress-tracker's deferred table).

Shapes
------
Channels ``C = 2 * n_levels``, interleaved per level (``2*l`` = u_l, ``2*l+1`` = v_l) to
match the WindowDenoiser contract and the sampler's tile layout. Spatial size is the
training crop (e.g. 64x64); the net is fully convolutional so it also runs on the
inference window the InfiniteDiffusion machinery feeds it.

EDM preconditioning (Karras Eq. 7)
----------------------------------
``D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x; c_noise(sigma))``
with the network ``F`` trained to predict the EDM target. ``sigma_data`` is the dataset's
per-channel-normalised std (≈1 after our per-(level,var) normalisation), so we default it
to 1.0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _num_groups(ch: int, max_groups: int = 32) -> int:
    """Largest group count <= max_groups that divides ``ch`` (GroupNorm requires divisibility)."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


# --------------------------------------------------------------------------- embeddings
class FourierEmbedding(nn.Module):
    """Random-Fourier features of log(sigma)/4 (EDM's noise conditioning input)."""

    def __init__(self, n_channels: int, scale: float = 16.0) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(n_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().ger(2.0 * np.pi * self.freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


# --------------------------------------------------------------------------- blocks
class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv residual block with additive noise-embedding conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, *, n_groups: int = 32) -> None:
        super().__init__()
        g_in = _num_groups(in_ch, n_groups)
        g_out = _num_groups(out_ch, n_groups)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention (used only at the coarsest resolution)."""

    def __init__(self, ch: int, *, n_groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(ch, n_groups), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        w = torch.softmax((q.transpose(1, 2) @ k) / (C ** 0.5), dim=-1)
        h = (v @ w.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


# --------------------------------------------------------------------------- U-Net
class WindUNet(nn.Module):
    """Compact U-Net F. Predicts the EDM-preconditioned residual (raw network output).

    Args:
        in_channels: = out_channels = 2 * n_levels (the diffused target's channels).
        model_channels: base width.
        channel_mult: per-resolution width multipliers (length = #resolutions).
        num_res_blocks: residual blocks per resolution.
        attn_resolutions: downsample factors at which to insert self-attention (e.g. {4}).
        cond_channels: extra *clean* conditioning channels concatenated to the (preconditioned)
            input before ``in_conv`` — e.g. the previous frame for autoregressive (M3) temporal
            generation. ``0`` (default) = the unconditional static model; the output channel
            count is always ``in_channels`` regardless.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        model_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (4,),
        cond_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = int(cond_channels)
        self.num_res_blocks = int(num_res_blocks)
        emb_ch = model_channels * 4

        self.map_noise = FourierEmbedding(model_channels)
        self.map_layer = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )

        self.in_conv = nn.Conv2d(in_channels + self.cond_channels, model_channels, 3, padding=1)

        # encoder
        self.down = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()
        chs = [model_channels]
        ch = model_channels
        ds = 1
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, out_ch, emb_ch))
                self.down_attn.append(AttnBlock(out_ch) if ds in attn_resolutions else nn.Identity())
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
        self.mid2 = ResBlock(ch, ch, emb_ch)

        # decoder (mirror)
        self.up = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(ch + chs.pop(), out_ch, emb_ch))
                self.up_attn.append(AttnBlock(out_ch) if ds in attn_resolutions else nn.Identity())
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
                cond: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.map_layer(self.map_noise(c_noise))

        if self.cond_channels:
            if cond is None:
                raise ValueError(f"net has cond_channels={self.cond_channels} but cond is None")
            x = torch.cat([x, cond], dim=1)
        h = self.in_conv(x)
        hs = [h]
        bi = 0
        for i in range(len(self.downsample)):
            for _ in range(self.num_res_blocks):
                h = self.down[bi](h, emb)
                h = self.down_attn[bi](h)
                hs.append(h)
                bi += 1
            if self.downsample[i] is not None:
                h = self.downsample[i](h)
                hs.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        ui = 0
        for i in range(len(self.upsample)):
            for _ in range(self.num_res_blocks + 1):
                h = self.up[ui](torch.cat([h, hs.pop()], dim=1), emb)
                h = self.up_attn[ui](h)
                ui += 1
            if self.upsample[i] is not None:
                h = self.upsample[i](h)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))


class EDMPrecond(nn.Module):
    """Karras EDM preconditioning around a raw network F (arXiv 2206.00364, Eq. 7).

    Exposes ``forward(x, sigma)`` = D(x; sigma), the denoised (x0) estimate, plus
    ``loss(x0)`` for the training objective. ``sigma_data`` ≈ 1 because crops are
    per-(level,variable) normalised upstream.
    """

    def __init__(
        self,
        n_channels: int,
        *,
        sigma_data: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        cond_channels: int = 0,
        net_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.cond_channels = int(cond_channels)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.net = WindUNet(n_channels, cond_channels=self.cond_channels, **dict(net_kwargs or {}))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4.0)
        # The condition is a *clean* input (already per-channel normalised); it is concatenated
        # un-scaled, so only the noisy target gets the EDM c_in scaling. Preconditioning math
        # (c_skip/c_out) and the loss apply to the target alone.
        F_x = self.net(c_in * x, c_noise, cond=cond)
        return c_skip * x + c_out * F_x

    # --- training objective (EDM loss weighting + lognormal sigma sampling) ---
    def loss(self, x0: torch.Tensor, *, cond: torch.Tensor | None = None,
             P_mean: float = -1.2, P_std: float = 1.2) -> torch.Tensor:
        rnd = torch.randn(x0.shape[0], device=x0.device)
        sigma = (rnd * P_std + P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x0) * sigma.reshape(-1, 1, 1, 1)
        D = self(x0 + n, sigma, cond=cond)
        return (weight.reshape(-1, 1, 1, 1) * (D - x0) ** 2).mean()
