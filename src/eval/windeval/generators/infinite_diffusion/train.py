"""EDM training loop for the wind window-denoiser — config-driven, resumable, cluster-ready.

Boring-baseline EDM training (Karras arXiv 2206.00364): lognormal sigma sampling + the
EDM loss weighting live in ``net.EDMPrecond.loss``; this module is the harness around it
(data loader, optimiser, EMA, checkpoint/resume, device selection, logging).

Designed to be launched as a cluster (SLURM) job: a single CLI entrypoint, deterministic
config, and checkpoint/resume so a pre-empted job picks up where it left off. The EMA
weights are what inference uses.

Usage
-----
    python -m src.eval.windeval.generators.infinite_diffusion.train --config <cfg.yaml>
    python -m ...train --config <cfg.yaml> --set train.n_steps=500 device=cpu   # overrides

A checkpoint bundles model + EMA + optimiser + step + the NormStats, so a resumed or
loaded run is fully self-describing.
"""
from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Work both as a package module (relative) and as a standalone script run directly on a
# cluster — `python .../train.py` puts this dir on sys.path[0], so the absolute fallback
# resolves WITHOUT importing src/eval/__init__ (which pulls the unrelated jax/gym stack).
try:
    from .data import (N_TIME_FEATURES, NormStats, WindCondSpaceTimeDataset,
                       WindCropDataset, WindPairDataset, WindSpaceTimeDataset)
    from .net import EDMPrecond
    from .spacetime import EDMPrecondSpaceTime
except ImportError:  # pragma: no cover - standalone script path
    from data import (N_TIME_FEATURES, NormStats, WindCondSpaceTimeDataset,
                      WindCropDataset, WindPairDataset, WindSpaceTimeDataset)
    from net import EDMPrecond
    from spacetime import EDMPrecondSpaceTime


# --------------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    data_path: str = "src/eval/windeval/data/era5_real.zarr"
    stats_path: str | None = None
    lazy_data: bool = False
    val_data_path: str | None = None
    crop: int = 64
    levels: tuple[int, int] | None = (49, 66)
    augment: bool = True

    # --- temporal (M3 autoregressive) ---
    # paired=True trains p(frame_{t+stride} | frame_t): the previous frame is concatenated as
    # clean conditioning channels (cond_channels = 2*n_levels). paired=False = the static model.
    paired: bool = False
    frame_stride: int = 1

    # --- temporal (M2 joint spacetime) ---
    # spacetime=True trains a factorized denoiser on H×W×τ blocks (τ = n_frames consecutive
    # frames at frame_stride spacing). Mutually exclusive with paired. temporal_kernel = the
    # 1D conv width along time. Stored as `tau` in the ckpt cfg for the SpaceTimeSampler.
    spacetime: bool = False
    n_frames: int = 4
    temporal_kernel: int = 3

    # --- conditioning (Phase 5, requires spacetime) ---
    # conditional=True trains p(block | location, time): per-pixel lat/lon coord channels
    # (clean, concat at input) + per-frame cyclic time harmonics (via the emb pathway).
    # Forces augment off (reflection is geographically wrong for a located model); the
    # dataset's CoordNorm is saved in the checkpoint so inference normalizes identically.
    conditional: bool = False

    model_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (4,)
    sigma_data: float = 1.0

    batch_size: int = 64
    lr: float = 2e-4
    ema_decay: float = 0.999
    n_steps: int = 100_000
    warmup_steps: int = 1_000
    num_workers: int = 4

    out_dir: str = "runs/idiff_m1"
    ckpt_every: int = 5_000
    log_every: int = 100
    val_every: int = 2_000
    val_batches: int = 8
    resume: bool = True            # auto-resume from out_dir/latest.pt if present
    device: str = "auto"           # auto | cpu | mps | cuda
    seed: int = 0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _coerce(default, value):
    """Coerce a string override to the type of the existing default."""
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes")
    if isinstance(default, (tuple, list)):   # incl. YAML-loaded lists; "" -> ()
        cleaned = str(value).translate({ord(c): None for c in "()[] "})
        return tuple(int(p) for p in cleaned.split(",") if p != "")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_config(path: str | None, overrides: list[str] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    raw: dict = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    # flatten one nested level (sections like train:/model:) into the flat dataclass
    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    for k, v in flat.items():
        if hasattr(cfg, k):
            setattr(cfg, k, _coerce(getattr(cfg, k), v) if isinstance(v, str) else v)
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        k = k.split(".")[-1]
        if hasattr(cfg, k):
            setattr(cfg, k, _coerce(getattr(cfg, k), v))
    # normalise tuple-ish fields that YAML may give as lists
    if isinstance(cfg.channel_mult, list):
        cfg.channel_mult = tuple(cfg.channel_mult)
    if isinstance(cfg.attn_resolutions, list):
        cfg.attn_resolutions = tuple(cfg.attn_resolutions)
    if isinstance(cfg.levels, list):
        cfg.levels = tuple(cfg.levels)
    return cfg


# --------------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# --------------------------------------------------------------------------- checkpoint
def save_ckpt(path: Path, *, model, ema, opt, step, stats: NormStats, cfg: TrainConfig,
              coord_norm: dict | None = None, best_val_loss: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "stats": {"mean_u": stats.mean_u, "std_u": stats.std_u,
                  "mean_v": stats.mean_v, "std_v": stats.std_v, "levels": stats.levels},
        "cfg": cfg.__dict__,
        "best_val_loss": best_val_loss,
    }
    if coord_norm is not None:
        payload["coord_norm"] = coord_norm
    torch.save(payload, path)


def build_model(cfg: TrainConfig, n_channels: int):
    if cfg.spacetime:
        return EDMPrecondSpaceTime(
            n_channels,
            tau=cfg.n_frames,
            sigma_data=cfg.sigma_data,
            cond_channels=(2 if cfg.conditional else 0),
            time_features=(N_TIME_FEATURES if cfg.conditional else 0),
            net_kwargs=dict(
                model_channels=cfg.model_channels,
                channel_mult=tuple(cfg.channel_mult),
                num_res_blocks=cfg.num_res_blocks,
                attn_resolutions=tuple(cfg.attn_resolutions),
                temporal_kernel=cfg.temporal_kernel,
            ),
        )
    return EDMPrecond(
        n_channels,
        sigma_data=cfg.sigma_data,
        cond_channels=(n_channels if cfg.paired else 0),
        net_kwargs=dict(
            model_channels=cfg.model_channels,
            channel_mult=tuple(cfg.channel_mult),
            num_res_blocks=cfg.num_res_blocks,
            attn_resolutions=tuple(cfg.attn_resolutions),
        ),
    )


def _batch_loss(model, batch, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    if cfg.conditional:
        x0, cond, tfeat = batch
        return model.loss(x0.to(device, non_blocking=True),
                          cond=cond.to(device, non_blocking=True),
                          tfeat=tfeat.to(device, non_blocking=True))
    if cfg.paired:
        cond, x0 = batch
        return model.loss(x0.to(device, non_blocking=True),
                          cond=cond.to(device, non_blocking=True))
    x0 = batch.to(device, non_blocking=True)
    return model.loss(x0)


@torch.no_grad()
def validation_loss(model, loader: DataLoader, cfg: TrainConfig,
                    device: torch.device) -> float:
    """Fixed-seed native-window denoising loss for checkpoint selection."""
    was_training = model.training
    model.eval()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    losses = []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(cfg.seed + 10_000_019)
        for batch in loader:
            losses.append(float(_batch_loss(model, batch, cfg, device)))
    model.train(was_training)
    return sum(losses) / len(losses)


# --------------------------------------------------------------------------- train
def train(cfg: TrainConfig) -> Path:
    # line-buffer stdout so step logs stream live to the SLURM .out file (which is block-
    # buffered by default when stdout is a file, making a running job look frozen).
    try:
        import sys
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device}  out={out}")

    if cfg.paired and cfg.spacetime:
        raise ValueError("paired (M3) and spacetime (M2) are mutually exclusive")
    if cfg.conditional and not cfg.spacetime:
        raise ValueError("conditional requires spacetime=True (the Phase-5 M2 route)")
    length = cfg.batch_size * cfg.n_steps
    coord_norm = None
    stats = NormStats.load(cfg.stats_path) if cfg.stats_path else None
    if cfg.lazy_data and stats is None:
        raise ValueError("lazy_data=True requires stats_path computed from training data")
    if cfg.conditional:
        dataset = WindCondSpaceTimeDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                           n_frames=cfg.n_frames, frame_stride=cfg.frame_stride,
                                           stats=stats, length=length, seed=cfg.seed,
                                           lazy=cfg.lazy_data)
        coord_norm = dataset.coord_norm.to_dict()
        mode_note = (f"  | CONDITIONAL SPACETIME (τ={cfg.n_frames}, stride {cfg.frame_stride}, "
                     f"{len(dataset.block_starts)} block starts, coord_norm={coord_norm})")
    elif cfg.spacetime:
        dataset = WindSpaceTimeDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                       n_frames=cfg.n_frames, frame_stride=cfg.frame_stride,
                                       stats=stats, augment=cfg.augment, length=length,
                                       seed=cfg.seed, lazy=cfg.lazy_data)
        mode_note = (f"  | SPACETIME (τ={cfg.n_frames}, stride {cfg.frame_stride}, "
                     f"{len(dataset.block_starts)} block starts)")
    elif cfg.paired:
        dataset = WindPairDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                  frame_stride=cfg.frame_stride, augment=cfg.augment,
                                  length=length, seed=cfg.seed)
        mode_note = (f"  | PAIRED (stride {cfg.frame_stride}, "
                     f"{len(dataset.pair_starts)} pair starts)")
    else:
        dataset = WindCropDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                  augment=cfg.augment, length=length, seed=cfg.seed)
        mode_note = ""
    stats = dataset.stats
    stats.save(out / "norm_stats.npz")
    print(f"[train] data: {dataset.T} steps x {dataset.L} levels x {dataset.Y}x{dataset.X}"
          f"  -> {dataset.n_channels} channels, crop {cfg.crop}" + mode_note)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(cfg.num_workers > 0))

    val_loader = None
    if cfg.val_data_path:
        if not cfg.spacetime:
            raise ValueError("validation data is currently supported for spacetime training")
        val_cls = WindCondSpaceTimeDataset if cfg.conditional else WindSpaceTimeDataset
        val_dataset = val_cls(
            cfg.val_data_path, crop=cfg.crop, levels=cfg.levels,
            n_frames=cfg.n_frames, frame_stride=cfg.frame_stride, stats=stats,
            length=cfg.batch_size * cfg.val_batches, seed=cfg.seed + 1, lazy=cfg.lazy_data,
            **({} if cfg.conditional else {"augment": False}),
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=0, drop_last=True,
                                pin_memory=(device.type == "cuda"))
        print(f"[train] validation: {cfg.val_data_path}  ({cfg.val_batches} fixed batches)")

    model = build_model(cfg, dataset.n_channels).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    start_step = 0
    best_val_loss = float("inf")
    latest = out / "latest.pt"
    if cfg.resume and latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        if ck.get("best_val_loss") is not None:
            best_val_loss = float(ck["best_val_loss"])
        print(f"[train] resumed from {latest} at step {start_step}")

    model.train()
    t0 = time.time()
    running = 0.0
    step = start_step
    for batch in loader:
        if step >= cfg.n_steps:
            break
        lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, cfg, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        running += float(loss.detach())
        step += 1
        if step % cfg.log_every == 0:
            rate = cfg.log_every / (time.time() - t0)
            print(f"[train] step {step:>7d}/{cfg.n_steps}  loss {running / cfg.log_every:.4f}"
                  f"  lr {lr:.2e}  {rate:.1f} it/s")
            running = 0.0
            t0 = time.time()
        if val_loader is not None and step % cfg.val_every == 0:
            val_loss = validation_loss(ema.shadow, val_loader, cfg, device)
            print(f"[train] validation @ step {step}: EMA loss {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt(out / "best.pt", model=model, ema=ema, opt=opt, step=step,
                          stats=stats, cfg=cfg, coord_norm=coord_norm,
                          best_val_loss=best_val_loss)
                print(f"[train] new best validation checkpoint @ step {step}")
        if step % cfg.ckpt_every == 0:
            save_ckpt(latest, model=model, ema=ema, opt=opt, step=step, stats=stats,
                      cfg=cfg, coord_norm=coord_norm, best_val_loss=best_val_loss)
            save_ckpt(out / f"step_{step}.pt", model=model, ema=ema, opt=opt,
                      step=step, stats=stats, cfg=cfg, coord_norm=coord_norm,
                      best_val_loss=best_val_loss)
            print(f"[train] checkpoint @ step {step}")

    save_ckpt(latest, model=model, ema=ema, opt=opt, step=step, stats=stats,
              cfg=cfg, coord_norm=coord_norm, best_val_loss=best_val_loss)
    print(f"[train] done @ step {step} -> {latest}")
    return latest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the InfiniteDiffusion wind denoiser (EDM).")
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.n_steps=500 device=cpu")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
