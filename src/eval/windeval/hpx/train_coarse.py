"""Stage 1 training: the whole-sphere coarse wind generator on HEALPix nside 32.

EDM training of :class:`net.EDMPrecondHpx` on :class:`dataset.HpxCoarseBlocks`. The loop is
the regional trainer's (EMA, warmup, clipping, resume with a shifted data seed, W&B resume,
snapshots, validation with the EMA weights, SIGUSR1 -> checkpoint -> exit 75 for Slurm
requeue), duplicated here rather than imported because both packages ship a ``net.py`` and a
standalone import of the regional ``train.py`` would collide with it on the cluster.

Validation holds out whole contiguous runs of the store (each run is a 7- or ~17-day
segment between excluded days), chosen deterministically, so the validation loss is on
weather the model never saw rather than on neighbouring hours of training blocks.

Usage::

    python src/eval/windeval/hpx/train_coarse.py --config unicorn/configs/stage1_hpx32.yaml
    python ... --set train.n_steps=200 batch_size=2 device=cpu       # smoke
"""
from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import HpxCoarseBlocks, compute_stats  # noqa: E402
from net import EDMPrecondHpx  # noqa: E402

_STOP_REQUESTED = False
_REQUEUE_REQUIRED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[train] signal {signum}: checkpointing after the current step", flush=True)


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    stores: list = field(default_factory=lambda: ["/scratch/sps252/era5_hpx"])
    layout_dir: str = "~/data/hpx_layout"
    stats_path: str = ""                 # npz with mean/std; "" = compute at first run and save to out_dir
    n_frames: int = 8
    stride_hours: int = 6
    slow_index: bool = True              # QBO-style slow-state scalar (value, present) into the embedding
    lookback_hours: int = 720
    storage_dtype: str = "float16"
    val_run_fraction: float = 0.04       # whole contiguous runs held out for validation

    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (8, 4)
    temporal_kernel: int = 3
    sigma_data: float = 1.0

    batch_size: int = 6
    lr: float = 2e-4
    ema_decay: float = 0.9999
    n_steps: int = 300_000
    warmup_steps: int = 1_000
    num_workers: int = 4
    amp: bool = True                     # bf16 autocast for the network forward

    out_dir: str = "runs/stage1_hpx32"
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


def _coerce(default, value):
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes")
    if isinstance(default, (tuple, list)):
        cleaned = str(value).translate({ord(c): None for c in "()[] "})
        parts = [p for p in cleaned.split(",") if p]
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return list(parts)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def load_config(path: str | None, overrides: list[str]) -> Config:
    cfg = Config()
    raw = yaml.safe_load(Path(path).read_text()) or {} if path else {}
    flat = {}
    for k, v in raw.items():
        flat.update(v) if isinstance(v, dict) else flat.__setitem__(k, v)
    for k, v in flat.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key: {k}")     # never silently ignore (Kahan lesson)
        setattr(cfg, k, _coerce(getattr(cfg, k), v) if isinstance(v, str) else v)
    for ov in overrides:
        k, _, v = ov.partition("=")
        k = k.split(".")[-1]
        if not hasattr(cfg, k):
            raise ValueError(f"unknown override: {k}")
        setattr(cfg, k, _coerce(getattr(cfg, k), v))
    for k in ("channel_mult", "attn_resolutions"):
        setattr(cfg, k, tuple(getattr(cfg, k)))
    cfg.layout_dir = str(Path(cfg.layout_dir).expanduser())
    cfg.stores = [str(Path(s).expanduser()) for s in cfg.stores]
    return cfg


# --------------------------------------------------------------------------- pieces
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


def save_ckpt(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def init_wandb(cfg: Config, out: Path, resume_id: str | None):
    if not cfg.wandb_project:
        return None, None
    try:
        import wandb
        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name or out.name,
                         id=resume_id, resume="allow" if resume_id else None, mode=cfg.wandb_mode,
                         dir=str(out), config=dict(cfg.__dict__))
        return run, run.id
    except Exception as e:                       # noqa: BLE001 - telemetry must not be fatal
        print(f"[train] wandb disabled ({type(e).__name__}: {e})")
        return None, None


def build_model(cfg: Config, n_channels: int) -> EDMPrecondHpx:
    return EDMPrecondHpx(n_channels, tau=cfg.n_frames, sigma_data=cfg.sigma_data,
                         net_kwargs=dict(model_channels=cfg.model_channels, channel_mult=cfg.channel_mult,
                                         num_res_blocks=cfg.num_res_blocks, attn_resolutions=cfg.attn_resolutions,
                                         temporal_kernel=cfg.temporal_kernel, slow_features=2))


def _split_runs(ds: HpxCoarseBlocks, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Block starts split by contiguous run: ~fraction of runs (deterministic) go to validation."""
    rng = np.random.default_rng(seed + 7)
    n_val = max(1, int(round(fraction * len(ds.runs))))
    val_runs = set(rng.choice(len(ds.runs), size=n_val, replace=False).tolist())
    run_of = np.zeros(ds.T, dtype=np.int64)
    for r, (a, b) in enumerate(ds.runs):
        run_of[a:b] = r
    is_val = np.isin(run_of[ds.block_starts], list(val_runs))
    return ds.block_starts[~is_val], ds.block_starts[is_val]


def _batch_loss(model, batch, device, amp: bool):
    x, cond, tfeat, slow = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        return model.loss(x, cond=cond, tfeat=tfeat, slow=slow)


@torch.no_grad()
def validation_loss(model, loader, cfg: Config, device) -> float:
    was = model.training; model.eval()
    losses = []
    devs = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(cfg.seed + 10_000_019)
        for batch in loader:
            losses.append(float(_batch_loss(model, batch, device, cfg.amp)))
    model.train(was)
    return float(np.mean(losses))


# --------------------------------------------------------------------------- train
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

    # resume peek first: the data seed shifts by start_step so a requeued leg sees new blocks
    start_step, best_val, ck, latest = 0, float("inf"), None, out / "latest.pt"
    if cfg.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        start_step = int(ck["step"]); best_val = float(ck.get("best_val_loss") or float("inf"))
    data_seed = cfg.seed + start_step

    stats_file = Path(cfg.stats_path).expanduser() if cfg.stats_path else out / "stats.npz"
    if stats_file.exists():
        z = np.load(stats_file); stats = {"mean": z["mean"], "std": z["std"], "n_hours": int(z["n_hours"])}
    else:
        stats = compute_stats(cfg.stores); np.savez(stats_file, **stats)
        print(f"[train] stats over {stats['n_hours']} hours -> {stats_file}")

    ds = HpxCoarseBlocks(cfg.stores, cfg.layout_dir, n_frames=cfg.n_frames, stride_hours=cfg.stride_hours,
                         stats=stats, length=cfg.batch_size * max(1, cfg.n_steps - start_step), seed=data_seed,
                         storage_dtype=cfg.storage_dtype, slow_index=cfg.slow_index, lookback_hours=cfg.lookback_hours)
    train_starts, val_starts = _split_runs(ds, cfg.val_run_fraction, cfg.seed)
    ds.block_starts = train_starts
    val_ds = copy.copy(ds); val_ds.block_starts = val_starts; val_ds.length = cfg.batch_size * cfg.val_batches; val_ds.seed = cfg.seed + 1
    print(f"[train] data: {ds.T} hours, nside {ds.nside}, {ds.C} channels; blocks τ={cfg.n_frames} x {cfg.stride_hours} h; "
          f"{len(train_starts)} train / {len(val_starts)} val starts ({len(ds.runs)} runs); slow_index={cfg.slow_index}")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"), persistent_workers=(cfg.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = build_model(cfg, ds.C).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"[train] model {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    if ck is not None:
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        print(f"[train] resumed from {latest} at step {start_step} (data_seed={data_seed})")
    wb_run, wb_id = init_wandb(cfg, out, (ck or {}).get("wandb_id"))
    del ck

    def checkpoint(path: Path, step: int) -> None:
        save_ckpt(path, model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step,
                  cfg=dict(cfg.__dict__), stats={"mean": stats["mean"], "std": stats["std"]},
                  nside=ds.nside, n_channels=ds.C, best_val_loss=best_val, wandb_id=wb_id,
                  slow_norm=(getattr(ds, "slow_mean", 0.0), getattr(ds, "slow_std", 1.0)))

    model.train(); t0 = time.time(); running = 0.0; step = start_step
    for batch in loader:
        if step >= cfg.n_steps:
            break
        lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, device, cfg.amp)
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
            checkpoint(latest, step)
            print(f"[train] checkpoint @ {step}")
        if step % (cfg.snapshot_every or cfg.ckpt_every) == 0:      # independent of ckpt_every
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
    ap = argparse.ArgumentParser(description="Train the whole-sphere coarse HEALPix wind generator (EDM).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)
    train(load_config(a.config, a.set))
    if _REQUEUE_REQUIRED:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
