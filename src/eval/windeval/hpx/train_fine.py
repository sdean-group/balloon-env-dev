"""Train Stage 2: the fine residual patch model on HEALPix nside 256 (see dataset_fine.py).

    python src/eval/windeval/hpx/train_fine.py --config unicorn/configs/stage2_hpx256.yaml
    python ... --set n_steps=30 out_dir=/tmp/smoke num_workers=2          # smoke

Same machinery as train_coarse.py (EMA, warmup, clipping, resume with a shifted data seed,
SIGUSR1 -> checkpoint -> exit 75, run-level validation holdout). The network is the summer
factorised space-time U-Net (generators/infinite_diffusion/spacetime.py) on 64x64 face patches:
target = scaled residual, per-frame conditioning = [lift(lin), lift(prev), lift(next)] with a
coarse-dropout flag for CFG, clean coords, 7 time features (harmonics + hour offset).
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from dataset_fine import HpxFineRuns, block_starts, collate_runs, estimate_scale, split_starts, _open  # noqa: E402
from train_coarse import EMA, _coerce, init_wandb, save_ckpt  # noqa: E402


def _summer_spacetime():
    """Import the summer regional package under its own name so its relative imports resolve
    without putting its directory on sys.path (its ``net.py`` would shadow ours)."""
    import importlib, types
    if "infdiff" not in sys.modules:
        pkg = types.ModuleType("infdiff"); pkg.__path__ = [str(_HERE.parent / "generators/infinite_diffusion")]
        sys.modules["infdiff"] = pkg
    return importlib.import_module("infdiff.spacetime")


EDMPrecondSpaceTime = _summer_spacetime().EDMPrecondSpaceTime

_STOP_REQUESTED = False
_REQUEUE_REQUIRED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[train] signal {signum}: checkpointing after the current step", flush=True)


@dataclass
class Config:
    store: str = "/scratch/sps252/era5_hpx"
    layout_dir: str = "~/data/hpx_layout"
    stats_path: str = ""                 # Stage 1's stats.npz (per-channel mean/std); required
    n_frames: int = 13
    coarse_stride: int = 6
    patch: int = 64
    pad: int = 32
    patches_per_item: int = 16
    lift: str = "nearest"                # residual baseline / conditioner lift: nearest (blocky) or bilinear (smooth)
    val_run_fraction: float = 0.04

    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (4,)
    temporal_kernel: int = 3
    sigma_data: float = 1.0
    sigma_max: float = 200.0
    sigma_dist: str = "log_uniform"
    train_sigma_min: float = 0.02
    train_sigma_max: float = 200.0
    P_mean: float = -1.2
    P_std: float = 1.2
    coarse_dropout: float = 0.1
    init_from: str = ""

    runs_per_batch: int = 1
    lr: float = 2e-4
    ema_decay: float = 0.9999
    n_steps: int = 200_000
    warmup_steps: int = 1_000
    num_workers: int = 8
    amp: bool = True
    out_dir: str = "runs/stage2_hpx256"
    ckpt_every: int = 2_000
    snapshot_every: int = 25_000
    log_every: int = 100
    val_every: int = 2_000
    val_batches: int = 8
    resume: bool = True
    device: str = "cuda"
    seed: int = 0
    wandb_project: str = ""
    wandb_mode: str = "offline"
    wandb_run_name: str = ""
    wandb_entity: str = ""


def load_config(path: str | None, overrides: list[str]) -> Config:
    cfg = Config()
    raw = yaml.safe_load(Path(path).read_text()) or {} if path else {}
    flat = {}
    for k, v in raw.items():
        flat.update(v) if isinstance(v, dict) else flat.__setitem__(k, v)
    for k, v in flat.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key: {k}")
        setattr(cfg, k, _coerce(getattr(cfg, k), v) if isinstance(v, str) else v)
    for ov in overrides:
        k, _, v = ov.partition("="); k = k.split(".")[-1]
        if not hasattr(cfg, k):
            raise ValueError(f"unknown override: {k}")
        setattr(cfg, k, _coerce(getattr(cfg, k), v))
    for k in ("channel_mult", "attn_resolutions"):
        setattr(cfg, k, tuple(getattr(cfg, k)))
    cfg.layout_dir = str(Path(cfg.layout_dir).expanduser()); cfg.store = str(Path(cfg.store).expanduser())
    return cfg


def build_model(cfg: Config, n_channels: int) -> EDMPrecondSpaceTime:
    return EDMPrecondSpaceTime(n_channels, tau=cfg.n_frames, sigma_data=cfg.sigma_data, sigma_max=cfg.sigma_max,
                               cond_channels=3, time_features=7, coarse_channels=3 * n_channels,
                               coarse_residual=False, coarse_flag=True,
                               net_kwargs=dict(model_channels=cfg.model_channels, channel_mult=cfg.channel_mult,
                                               num_res_blocks=cfg.num_res_blocks, attn_resolutions=cfg.attn_resolutions,
                                               temporal_kernel=cfg.temporal_kernel))


def _batch_loss(model, batch, device, cfg: Config):
    r, cond, coords, tfeat = (t.to(device, non_blocking=True) for t in batch)
    r, cond = r.float(), cond.float()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
        return model.loss(r, cond=coords, tfeat=tfeat, coarse=cond, coarse_dropout=cfg.coarse_dropout,
                          P_mean=cfg.P_mean, P_std=cfg.P_std, sigma_dist=cfg.sigma_dist,
                          train_sigma_min=cfg.train_sigma_min, train_sigma_max=cfg.train_sigma_max)


@torch.no_grad()
def validation_loss(model, loader, cfg: Config, device) -> float:
    was = model.training; model.eval(); losses = []
    for i, batch in enumerate(loader):
        if i >= cfg.val_batches:
            break
        losses.append(float(_batch_loss(model, batch, device, cfg)))
    model.train(was)
    return float(np.mean(losses)) if losses else float("nan")


def train(cfg: Config) -> Path:
    global _REQUEUE_REQUIRED
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device} out={out}")

    start_step, best_val, ck, latest = 0, float("inf"), None, out / "latest.pt"
    if cfg.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        start_step = int(ck["step"]); best_val = float(ck.get("best_val_loss") or float("inf"))
    data_seed = cfg.seed + start_step

    z = np.load(Path(cfg.stats_path).expanduser())
    stats = {"mean": z["mean"], "std": z["std"]}
    _, _, hours = _open(cfg.store)
    starts = block_starts(cfg.store, cfg.n_frames)
    train_starts, val_starts = split_starts(starts, hours, cfg.val_run_fraction, cfg.seed)
    kw = dict(n_frames=cfg.n_frames, coarse_stride=cfg.coarse_stride, patch=cfg.patch, pad=cfg.pad,
              patches_per_item=cfg.patches_per_item, lift=cfg.lift)
    scale_file = out / "stage2_scale.npy"
    if scale_file.exists():
        scale = np.load(scale_file)
    else:
        probe = HpxFineRuns(cfg.store, cfg.layout_dir, stats=stats, starts=train_starts, length=1, seed=cfg.seed, **kw)
        t0 = time.time(); scale = estimate_scale(probe, seed=cfg.seed); np.save(scale_file, scale)
        print(f"[train] residual scale (normalised units) mean {scale.mean():.4f} min {scale.min():.4f} max {scale.max():.4f} "
              f"-> {scale_file} ({time.time() - t0:.0f}s)")
    ds = HpxFineRuns(cfg.store, cfg.layout_dir, stats=stats, starts=train_starts, scale=scale,
                     length=cfg.runs_per_batch * max(1, cfg.n_steps - start_step), seed=data_seed, **kw)
    val_ds = HpxFineRuns(cfg.store, cfg.layout_dir, stats=stats, starts=val_starts, scale=scale,
                         length=cfg.val_batches * cfg.runs_per_batch, seed=10_000 + cfg.seed, **kw)
    print(f"[train] data: {len(starts)} block starts ({len(train_starts)} train / {len(val_starts)} val); nside {ds.nside}/{ds.nside_c}, "
          f"{ds.C} channels; blocks τ={cfg.n_frames} h, coarse every {cfg.coarse_stride} h; {cfg.patch}px patches x {cfg.patches_per_item} per run; "
          f"{ds.geom.n_positions()} patch positions; lift {cfg.lift}")
    loader = DataLoader(ds, batch_size=cfg.runs_per_batch, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_runs,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=cfg.runs_per_batch, shuffle=False, num_workers=min(2, cfg.num_workers),
                            collate_fn=collate_runs)

    model = build_model(cfg, ds.C).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"[train] model {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    if ck is not None:
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        print(f"[train] resumed from {latest} at step {start_step} (data_seed={data_seed})")
    elif cfg.init_from:
        ck0 = torch.load(Path(cfg.init_from).expanduser(), map_location="cpu", weights_only=False)
        model.load_state_dict(ck0["model"]); ema.shadow.load_state_dict(ck0["ema"])
        print(f"[train] warm start from {cfg.init_from} (its step {ck0.get('step')})"); del ck0
    wb_run, wb_id = init_wandb(cfg, out, (ck or {}).get("wandb_id"))
    del ck

    def checkpoint(path: Path, step: int) -> None:
        save_ckpt(path, model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step,
                  cfg=dict(cfg.__dict__), stats=stats, scale=scale, nside=ds.nside, nside_coarse=ds.nside_c,
                  n_channels=ds.C, best_val_loss=best_val, wandb_id=wb_id)

    model.train(); t0 = time.time(); running = 0.0; step = start_step
    for batch in loader:
        if step >= cfg.n_steps:
            break
        lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, device, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update(model)
        running += float(loss.detach()); step += 1
        if step % cfg.log_every == 0:
            rate = cfg.log_every / (time.time() - t0); mean_loss = running / cfg.log_every
            print(f"[train] step {step:>7d}/{cfg.n_steps}  loss {mean_loss:.4f}  lr {lr:.2e}  {rate:.2f} it/s")
            if wb_run is not None:
                wb_run.log({"loss": mean_loss, "lr": lr, "it_per_s": rate}, step=step)
            running, t0 = 0.0, time.time()
        if step % cfg.val_every == 0:
            v = validation_loss(ema.shadow, val_loader, cfg, device)
            print(f"[train] validation @ {step}: EMA loss {v:.6f}")
            if wb_run is not None:
                wb_run.log({"val_loss": v}, step=step)
            if v < best_val:
                best_val = v; checkpoint(out / "best.pt", step); print(f"[train] new best @ {step}")
        if step % cfg.ckpt_every == 0:
            checkpoint(latest, step); print(f"[train] checkpoint @ {step}")
        if step % (cfg.snapshot_every or cfg.ckpt_every) == 0:
            checkpoint(out / f"step_{step}.pt", step)
        if _STOP_REQUESTED:
            break
    checkpoint(latest, step)
    state = "paused" if _STOP_REQUESTED and step < cfg.n_steps else "done"
    _REQUEUE_REQUIRED = state == "paused"
    print(f"[train] {state} @ step {step} -> {latest}")
    if wb_run is not None:
        wb_run.finish()
    return latest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the Stage 2 fine residual patch model (EDM) on HEALPix faces.")
    ap.add_argument("--config", default=None); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)
    train(load_config(a.config, a.set))
    if _REQUEUE_REQUIRED:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
