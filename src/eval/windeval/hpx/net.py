"""HEALPix-aware factorised space-time U-Net for the whole-sphere coarse stage.

The regional model (``spacetime.SpaceTimeUNet``) is a 2-D U-Net per frame plus a zero-init
1-D temporal conv at every level. This is the same network on the sphere:

- Tensors carry a face axis: ``(B, τ, C, F=12, H, W)``. Every spatial conv sees the 12
  faces as separate images **after** a cross-face halo exchange (``earth2grid.healpix.pad``),
  so features flow across face edges exactly as they would on a flat domain. Stride-2
  convs pad the same way; per-face transposed convs upsample.
- Attention at the coarsest levels runs over the tokens of **all faces at once**, which
  gives the bottleneck a global receptive field (at nside 32 with three downsamplings the
  bottleneck is 12 x 4 x 4 = 192 tokens).
- Temporal mixing is per pixel and unchanged. GroupNorm/SiLU are pointwise.
- Conditioning: clean coordinate channels (3) concatenated per pixel; per-frame time
  harmonics (6) and an optional slow-state pair (value, present) through the zero-init
  embedding path, like the regional model.

``EDMPrecondHpx`` wraps it with Karras preconditioning; the loss is the regional one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from earth2grid import healpix as _hpx
except ImportError:  # pragma: no cover
    _hpx = None


def hpx_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """``(N, F, C, H, W)`` -> ``(N, F, C, H+2p, W+2p)`` with halos from neighbouring faces."""
    if _hpx is None:
        raise ImportError("earth2grid is required for cross-face padding")
    return _hpx.pad(x, padding=pad)


def _num_groups(ch: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class HpxConv(nn.Module):
    """3x3 (or kxk) conv on 12 faces with cross-face halos. Input/output ``(N, F, C, H, W)``."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, F, C, H, W = x.shape
        if self.pad:
            x = hpx_pad(x, self.pad)
        y = self.conv(x.reshape(N * F, C, x.shape[-2], x.shape[-1]))
        return y.reshape(N, F, y.shape[1], y.shape[2], y.shape[3])


class FourierEmbedding(nn.Module):
    def __init__(self, n_channels: int, scale: float = 16.0) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(n_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().ger(2.0 * np.pi * self.freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = HpxConv(in_ch, out_ch)
        self.emb = nn.Linear(emb_ch, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = HpxConv(out_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _pw(mod, x):                       # pointwise module over (N, F, C, H, W)
        N, F, C, H, W = x.shape
        return mod(x.reshape(N * F, C, H, W)).reshape(N, F, -1, H, W)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nn.functional.silu(self._pw(self.norm1, x)))
        h = h + self.emb(emb)[:, None, :, None, None]
        h = self.conv2(nn.functional.silu(self._pw(self.norm2, h)))
        return h + self._pw(self.skip, x)


class AttnBlock(nn.Module):
    """Single-head self-attention over the tokens of all 12 faces jointly."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(ch), ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, F, C, H, W = x.shape
        h = self.norm(x.reshape(N * F, C, H, W)).reshape(N, F, C, H * W)
        h = h.permute(0, 1, 3, 2).reshape(N, F * H * W, C)               # tokens of all faces
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        a = torch.softmax(q @ k.transpose(1, 2) / C ** 0.5, dim=-1)
        h = self.proj(a @ v)
        h = h.reshape(N, F, H * W, C).permute(0, 1, 3, 2).reshape(N, F, C, H, W)
        return x + h


class TemporalConv(nn.Module):
    """Zero-init 1-D conv along τ at every pixel; ``(N=B*τ, F, C, H, W)``."""

    def __init__(self, ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(ch), ch)
        self.conv = nn.Conv1d(ch, ch, kernel, padding=kernel // 2)
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, tau: int) -> torch.Tensor:
        N, F, C, H, W = x.shape
        B = N // tau
        h = self.norm(x.reshape(N * F, C, H, W)).reshape(B, tau, F, C, H, W)
        h = h.permute(0, 2, 4, 5, 3, 1).reshape(B * F * H * W, C, tau)
        h = self.conv(nn.functional.silu(h))
        h = h.reshape(B, F, H, W, C, tau).permute(0, 5, 1, 4, 2, 3).reshape(N, F, C, H, W)
        return x + h


class HpxSpaceTimeUNet(nn.Module):
    def __init__(self, in_channels: int, *, model_channels: int = 128,
                 channel_mult: tuple[int, ...] = (1, 2, 2, 2), num_res_blocks: int = 2,
                 attn_resolutions: tuple[int, ...] = (8, 4), temporal_kernel: int = 3,
                 cond_channels: int = 3, time_features: int = 6, slow_features: int = 2) -> None:
        super().__init__()
        self.in_channels, self.cond_channels = int(in_channels), int(cond_channels)
        self.time_features, self.slow_features = int(time_features), int(slow_features)
        emb_ch = model_channels * 4
        self.map_noise = FourierEmbedding(model_channels)
        self.map_layer = nn.Sequential(nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))
        self.map_tfeat = nn.Linear(self.time_features + self.slow_features, emb_ch)
        nn.init.zeros_(self.map_tfeat.weight); nn.init.zeros_(self.map_tfeat.bias)
        self.in_conv = HpxConv(in_channels + self.cond_channels, model_channels)

        self.down, self.down_attn, self.down_temporal, self.downsample = (nn.ModuleList() for _ in range(4))
        chs, ch, ds = [model_channels], model_channels, 1
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, out_ch, emb_ch))
                self.down_attn.append(AttnBlock(out_ch) if ds in attn_resolutions else nn.Identity())
                self.down_temporal.append(TemporalConv(out_ch, temporal_kernel))
                ch = out_ch; chs.append(ch)
            if i != len(channel_mult) - 1:
                self.downsample.append(HpxConv(ch, ch, 3, stride=2)); chs.append(ch); ds *= 2
            else:
                self.downsample.append(None)
        self.mid1, self.mid_attn = ResBlock(ch, ch, emb_ch), AttnBlock(ch)
        self.mid_temporal, self.mid2 = TemporalConv(ch, temporal_kernel), ResBlock(ch, ch, emb_ch)
        self.up, self.up_attn, self.up_temporal, self.upsample = (nn.ModuleList() for _ in range(4))
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(ch + chs.pop(), out_ch, emb_ch))
                self.up_attn.append(AttnBlock(out_ch) if ds in attn_resolutions else nn.Identity())
                self.up_temporal.append(TemporalConv(out_ch, temporal_kernel))
                ch = out_ch
            if i != 0:
                self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)); ds //= 2
            else:
                self.upsample.append(None)
        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = HpxConv(ch, in_channels)
        nn.init.zeros_(self.out_conv.conv.weight); nn.init.zeros_(self.out_conv.conv.bias)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, cond: torch.Tensor,
                tfeat: torch.Tensor, slow: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, τ, C, F, H, W); cond: (B, 3, F, H, W); tfeat: (B, τ, 6); slow: (B, τ, 2)
        B, tau, C, F, H, W = x.shape
        emb = self.map_layer(self.map_noise(c_noise)).repeat_interleave(tau, dim=0)   # (B*τ, emb)
        if slow is None:
            slow = torch.zeros(B, tau, self.slow_features, device=x.device, dtype=x.dtype)
        emb = emb + self.map_tfeat(torch.cat([tfeat, slow], dim=-1).reshape(B * tau, -1))
        h = torch.cat([x, cond[:, None].expand(B, tau, -1, F, H, W)], dim=2)
        h = h.reshape(B * tau, C + self.cond_channels, F, H, W).permute(0, 2, 1, 3, 4)  # (N, F, C, H, W)
        h = self.in_conv(h)
        hs = [h]
        i = 0                                    # encoder: num_res_blocks per level, then downsample
        for li, down in enumerate(self.downsample):
            n = (len(self.down) // len(self.downsample))
            for _ in range(n):
                h = self.down[i](h, emb); h = self.down_attn[i](h); h = self.down_temporal[i](h, tau)
                hs.append(h); i += 1
            if down is not None:
                h = down(h); hs.append(h)
        h = self.mid1(h, emb); h = self.mid_attn(h); h = self.mid_temporal(h, tau); h = self.mid2(h, emb)
        i = 0
        for li, up in enumerate(self.upsample):
            n = len(self.up) // len(self.upsample)
            for _ in range(n):
                h = torch.cat([h, hs.pop()], dim=2)
                h = self.up[i](h, emb); h = self.up_attn[i](h); h = self.up_temporal[i](h, tau); i += 1
            if up is not None:
                N, Fq, Cq, Hq, Wq = h.shape
                h = up(h.reshape(N * Fq, Cq, Hq, Wq)).reshape(N, Fq, Cq, Hq * 2, Wq * 2)
        N, Fq, Cq, Hq, Wq = h.shape
        h = nn.functional.silu(self.out_norm(h.reshape(N * Fq, Cq, Hq, Wq))).reshape(N, Fq, Cq, Hq, Wq)
        out = self.out_conv(h)                                              # (B*τ, F, C, H, W)
        return out.permute(0, 2, 1, 3, 4).reshape(B, tau, C, F, H, W)


class EDMPrecondHpx(nn.Module):
    """Karras EDM preconditioning around ``HpxSpaceTimeUNet``; loss as in the regional model."""

    def __init__(self, n_channels: int, *, tau: int, sigma_data: float = 1.0,
                 sigma_min: float = 0.002, sigma_max: float = 80.0, net_kwargs: dict | None = None) -> None:
        super().__init__()
        self.n_channels, self.tau, self.sigma_data = int(n_channels), int(tau), float(sigma_data)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.net = HpxSpaceTimeUNet(n_channels, **dict(net_kwargs or {}))

    def forward(self, x, sigma, cond, tfeat, slow=None):
        sigma = sigma.reshape(-1, 1, 1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4.0
        return c_skip * x + c_out * self.net(c_in * x, c_noise, cond, tfeat, slow)

    def loss(self, x0, *, cond, tfeat, slow=None, P_mean: float = -1.2, P_std: float = 1.2):
        B = x0.shape[0]
        sigma = (torch.randn(B, device=x0.device) * P_std + P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x0) * sigma.reshape(-1, 1, 1, 1, 1, 1)
        D = self(x0 + n, sigma, cond, tfeat, slow)
        return (weight.reshape(-1, 1, 1, 1, 1, 1) * (D - x0) ** 2).mean()
