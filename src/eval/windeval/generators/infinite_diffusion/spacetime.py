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
    from .data import NormStats
    from .net import AttnBlock, FourierEmbedding, ResBlock, _num_groups
    from .trained import edm_sigma_schedule
except ImportError:  # pragma: no cover - standalone script path
    from data import NormStats
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
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_res_blocks = int(num_res_blocks)
        emb_ch = model_channels * 4

        self.map_noise = FourierEmbedding(model_channels)
        self.map_layer = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )
        self.in_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

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

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        # x: (B, τ, C, H, W); c_noise: (B,) per-block. Fold τ into batch for spatial ops,
        # repeat the noise embedding across τ, and run TemporalConv with the real τ.
        B, tau, C, H, W = x.shape
        emb = self.map_layer(self.map_noise(c_noise))            # (B, emb)
        emb = emb.repeat_interleave(tau, dim=0)                  # (B*τ, emb)
        xf = x.reshape(B * tau, C, H, W)

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
        net_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.tau = int(tau)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.net = SpaceTimeUNet(n_channels, **dict(net_kwargs or {}))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1, 1, 1, 1).to(x.dtype)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4.0)
        F_x = self.net(c_in * x, c_noise)
        return c_skip * x + c_out * F_x

    def loss(self, x0: torch.Tensor, *, P_mean: float = -1.2, P_std: float = 1.2) -> torch.Tensor:
        rnd = torch.randn(x0.shape[0], device=x0.device)
        sigma = (rnd * P_std + P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x0) * sigma.reshape(-1, 1, 1, 1, 1)
        D = self(x0 + n, sigma)
        return (weight.reshape(-1, 1, 1, 1, 1) * (D - x0) ** 2).mean()


class SpaceTimeSampler:
    """A trained spacetime block model presented as a block sampler (Heun ODE).

    ``.sample_block((H,W), seed)`` generates one τ-frame block in m/s; temporal tiling for
    long sequences is the deferred extension (wraps the frozen machinery in time).
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

        self.model = EDMPrecondSpaceTime(
            self.n_channels, tau=self.tau, sigma_data=cfg["sigma_data"],
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

    @torch.no_grad()
    def _heun_block(self, x_unit: torch.Tensor) -> torch.Tensor:
        sig = edm_sigma_schedule(self.num_steps, self.sigma_min, self.sigma_max,
                                 device=x_unit.device, dtype=x_unit.dtype)
        x = x_unit * sig[0]
        B = x.shape[0]
        for i in range(self.num_steps):
            s_cur, s_next = sig[i], sig[i + 1]
            d = (x - self.model(x, s_cur.expand(B))) / s_cur
            x_next = x + (s_next - s_cur) * d
            if s_next > 0:
                d2 = (x_next - self.model(x_next, s_next.expand(B))) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
            x = x_next
        return x

    @torch.no_grad()
    def sample_block(self, hw: tuple[int, int], *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Generate one τ-frame block. Returns ``(us, vs)`` each ``(τ, L, H, W)`` in m/s."""
        H, W = hw
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(1, self.tau, self.n_channels, H, W, generator=g).to(self.device)
        block = self._heun_block(z)                                  # (1,τ,C,H,W) norm
        block = self.stats.denormalize(block.reshape(self.tau, self.n_channels, H, W))
        block = block.reshape(self.tau, self.n_levels, 2, H, W).cpu().numpy()
        return block[:, :, 0], block[:, :, 1]
