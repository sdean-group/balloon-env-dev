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
    from .data import (N_TIME_FEATURES, NormStats, WindCoarseCondSpaceTimeDataset,
                       WindCondSpaceTimeDataset,
                       WindCropDataset, WindPairDataset, WindSpaceTimeDataset,
                       measure_residual_scale)
    from .net import EDMPrecond
    from .spacetime import EDMPrecondSpaceTime
except ImportError:  # pragma: no cover - standalone script path
    from data import (N_TIME_FEATURES, NormStats, WindCoarseCondSpaceTimeDataset,
                      WindCondSpaceTimeDataset,
                      WindCropDataset, WindPairDataset, WindSpaceTimeDataset,
                      measure_residual_scale)
    from net import EDMPrecond
    from spacetime import EDMPrecondSpaceTime


# --------------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    data_path: str | list = "src/eval/windeval/data/era5_real.zarr"   # one zarr or a list
    data_dtype: str = "float32"    # in-RAM storage; "float16" halves RAM for multi-year sets
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
    # coarse_factor > 0 adds COARSE conditioning (Phase 5b, requires conditional): a
    # horizontally block-meaned copy of the target block (all levels kept) is given to the
    # model as extra clean input channels. 8 -> a 64 crop becomes 8x8 cells of 2deg
    # (~200 km). This turns the task from "invent weather" into "downscale a forecast";
    # see data.WindCoarseCondSpaceTimeDataset for why, and note the evaluation asymmetry.
    # 0 (default) = exactly the pre-5b model, so old configs/checkpoints are unaffected.
    coarse_factor: int = 0
    # --- Phase 5b-2: point the noise schedule at the band that is actually unknown ---
    # coarse_residual: diffuse (x - bilinear_upsample(coarse)) / coarse_scale instead of x.
    # Measured on the held-out reference at factor 8: upsampling already explains 97.4% of
    # the variance, so WITHOUT this ~74% of EDM's sigma draws land above the residual
    # amplitude (0.139) and train on an already-solved problem. See
    # spacetime.EDMPrecondSpaceTime for the full argument.
    coarse_residual: bool = False
    # 0 = measure from the training set at run start (recommended); >0 pins it. The value
    # actually used is written back into cfg and therefore into the checkpoint, so
    # sampling reproduces the training transform exactly.
    coarse_scale: float = 0.0
    # Probability of dropping the coarse field during training (with a flag channel to
    # mark it), which creates the unconditional branch classifier-free guidance needs.
    # Sampling-time knob only: guidance=1 reproduces the ordinary conditional model.
    coarse_dropout: float = 0.0

    # --- Weights & Biases (optional; "" = off, so every existing config is unaffected) ---
    # wandb_mode defaults to "offline" because Kahan COMPUTE nodes are not assumed to have
    # outbound internet — an online init would block or crash a 30 h run at step 0. Offline
    # writes to wandb/ in out_dir; sync afterwards FROM THE LOGIN NODE with
    #   .venv-idiff/bin/wandb sync runs/<out_dir>/wandb/offline-run-*
    # Set wandb_mode: online only once the compute node is confirmed to have egress.
    wandb_project: str = ""
    wandb_mode: str = "offline"          # offline | online | disabled
    wandb_run_name: str = ""             # defaults to out_dir's basename

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
    # step_N.pt snapshot cadence; 0 = same as ckpt_every. latest.pt still updates every
    # ckpt_every (crash-resume granularity) — snapshots are the ~641 MB/each keepers, and
    # on Kahan's ~100 GB quota a 200k-step run at snapshot=ckpt cadence is a quota killer.
    snapshot_every: int = 0
    log_every: int = 100
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
              coord_norm: dict | None = None, wandb_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "stats": {"mean_u": stats.mean_u, "std_u": stats.std_u,
                  "mean_v": stats.mean_v, "std_v": stats.std_v, "levels": stats.levels},
        "cfg": cfg.__dict__,
    }
    if coord_norm is not None:
        payload["coord_norm"] = coord_norm
    if wandb_id is not None:
        payload["wandb_id"] = wandb_id      # so a resumed leg re-attaches to ONE curve
    torch.save(payload, path)


def init_wandb(cfg: TrainConfig, out: Path, resume_id: str | None):
    """Start (or RESUME) a W&B run. Returns (run, run_id) or (None, None) if disabled.

    Resuming the SAME run id is the whole point: this project's jobs die and resubmit
    (wall limits, cluster resets — the m2coarse2 run was killed once mid-flight), and a
    fresh ``wandb.init`` per leg would fragment one loss curve into several runs that
    cannot be overlaid. The id is stored in the checkpoint, so a resumed leg re-attaches
    and the curve is continuous across restarts.

    Failure here must NEVER kill training — a 30 h run is worth more than its telemetry —
    so every error degrades to "no logging" with a printed warning.
    """
    if not cfg.wandb_project:
        return None, None
    try:
        import wandb
    except ImportError:
        print("[train] wandb_project set but wandb is not installed — continuing without it")
        return None, None
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or out.name,
            id=resume_id, resume="allow" if resume_id else None,
            mode=cfg.wandb_mode,
            dir=str(out),
            config={k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        )
        print(f"[train] wandb {cfg.wandb_mode} run '{run.name}' id={run.id}"
              + ("  (RESUMED)" if resume_id else ""))
        return run, run.id
    except Exception as e:                       # noqa: BLE001 - telemetry must not be fatal
        print(f"[train] wandb init failed ({type(e).__name__}: {e}) — continuing without it")
        return None, None


def build_model(cfg: TrainConfig, n_channels: int):
    if cfg.spacetime:
        return EDMPrecondSpaceTime(
            n_channels,
            tau=cfg.n_frames,
            sigma_data=cfg.sigma_data,
            cond_channels=(2 if cfg.conditional else 0),
            time_features=(N_TIME_FEATURES if cfg.conditional else 0),
            coarse_channels=(n_channels if cfg.coarse_factor else 0),
            coarse_residual=cfg.coarse_residual,
            coarse_scale=(cfg.coarse_scale or 1.0),
            coarse_flag=bool(cfg.coarse_dropout),
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
    if cfg.coarse_factor and not cfg.conditional:
        raise ValueError("coarse_factor requires conditional=True (Phase-5b builds on 5)")
    if cfg.coarse_factor and cfg.crop % cfg.coarse_factor:
        raise ValueError(f"crop {cfg.crop} not divisible by coarse_factor {cfg.coarse_factor}")
    if (cfg.coarse_residual or cfg.coarse_dropout) and not cfg.coarse_factor:
        raise ValueError("coarse_residual / coarse_dropout require coarse_factor > 0")
    if not 0.0 <= cfg.coarse_dropout < 1.0:
        raise ValueError(f"coarse_dropout must be in [0,1), got {cfg.coarse_dropout}")

    # Peek the resume step BEFORE building the dataset: the DataLoader restarts at idx 0
    # on every (re)start and items are idx-seeded, so with an unchanged seed a resumed
    # run REPLAYS the exact crop sequence it already trained on (every 24 h wall-limit
    # leg re-saw leg 1's crops — a large silent cut to effective sample diversity).
    # Shifting the seed by start_step gives each leg a fresh deterministic stream.
    start_step = 0
    latest = out / "latest.pt"
    ck = None
    if cfg.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        start_step = int(ck["step"])
    data_seed = cfg.seed + start_step
    length = cfg.batch_size * max(1, cfg.n_steps - start_step)
    coord_norm = None
    if cfg.conditional:
        ds_kw = dict(crop=cfg.crop, levels=cfg.levels, n_frames=cfg.n_frames,
                     frame_stride=cfg.frame_stride, length=length, seed=data_seed,
                     storage_dtype=cfg.data_dtype)
        if cfg.coarse_factor:
            dataset = WindCoarseCondSpaceTimeDataset(cfg.data_path,
                                                     coarse_factor=cfg.coarse_factor, **ds_kw)
        else:
            dataset = WindCondSpaceTimeDataset(cfg.data_path, **ds_kw)
        coord_norm = dataset.coord_norm.to_dict()
        mode_note = (f"  | CONDITIONAL SPACETIME (τ={cfg.n_frames}, stride {cfg.frame_stride}, "
                     f"{len(dataset.block_starts)} block starts, coord_norm={coord_norm}"
                     + (f", COARSE f={cfg.coarse_factor} -> "
                        f"{cfg.crop // cfg.coarse_factor}²" if cfg.coarse_factor else "") + ")")
        if cfg.coarse_factor and cfg.coarse_residual and not cfg.coarse_scale:
            # Measured once here, then written into cfg so it lands in every checkpoint.
            # A resumed leg re-measures: the dataset seed shifts with start_step, so the
            # sample differs — but the estimate is over 256 blocks x 4 frames x 36
            # channels x 64², i.e. ~1.5e8 values, so the leg-to-leg spread is negligible.
            cfg.coarse_scale = measure_residual_scale(dataset)
            mode_note += (f"\n[train] residual parameterization ON: coarse_scale="
                          f"{cfg.coarse_scale:.4f} (measured over 256 blocks); "
                          f"diffusing (x - upsample(coarse)) / coarse_scale"
                          + (f"; coarse_dropout={cfg.coarse_dropout} (CFG enabled)"
                             if cfg.coarse_dropout else ""))
    elif cfg.spacetime:
        dataset = WindSpaceTimeDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                       n_frames=cfg.n_frames, frame_stride=cfg.frame_stride,
                                       augment=cfg.augment, length=length, seed=data_seed,
                                       storage_dtype=cfg.data_dtype)
        mode_note = (f"  | SPACETIME (τ={cfg.n_frames}, stride {cfg.frame_stride}, "
                     f"{len(dataset.block_starts)} block starts)")
    elif cfg.paired:
        dataset = WindPairDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                  frame_stride=cfg.frame_stride, augment=cfg.augment,
                                  length=length, seed=data_seed)
        mode_note = (f"  | PAIRED (stride {cfg.frame_stride}, "
                     f"{len(dataset.pair_starts)} pair starts)")
    else:
        dataset = WindCropDataset(cfg.data_path, crop=cfg.crop, levels=cfg.levels,
                                  augment=cfg.augment, length=length, seed=data_seed)
        mode_note = ""
    stats = dataset.stats
    stats.save(out / "norm_stats.npz")
    print(f"[train] data: {dataset.T} steps x {dataset.L} levels x {dataset.Y}x{dataset.X}"
          f"  -> {dataset.n_channels} channels, crop {cfg.crop}" + mode_note)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"))

    model = build_model(cfg, dataset.n_channels).to(device)
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if ck is not None:
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])      # optimizer state auto-moves to param device
        print(f"[train] resumed from {latest} at step {start_step} (data_seed={data_seed})")
    # read the id BEFORE `del ck` so a resumed leg re-attaches to the same W&B curve
    wb_run, wb_id = init_wandb(cfg, out, (ck or {}).get("wandb_id"))
    if ck is not None:
        del ck

    model.train()
    t0 = time.time()
    running = 0.0
    step = start_step
    for batch in loader:
        if step >= cfg.n_steps:
            break
        tfeat = coarse = None
        if cfg.conditional:
            if cfg.coarse_factor:
                x0, cond, tfeat, coarse = batch        # + block-meaned coarse field
                coarse = coarse.to(device, non_blocking=True)
            else:
                x0, cond, tfeat = batch                # (block, coords, time features)
            x0 = x0.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            tfeat = tfeat.to(device, non_blocking=True)
        elif cfg.paired:
            cond, x0 = batch                                  # (frame_t, frame_{t+stride})
            cond = cond.to(device, non_blocking=True)
            x0 = x0.to(device, non_blocking=True)
        else:                                                 # static (4D) or spacetime (5D block)
            cond, x0 = None, batch.to(device, non_blocking=True)
        lr = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        if cfg.conditional:
            loss = model.loss(x0, cond=cond, tfeat=tfeat, coarse=coarse,
                              coarse_dropout=cfg.coarse_dropout)
        elif cfg.spacetime:
            loss = model.loss(x0)
        else:
            loss = model.loss(x0, cond=cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        running += float(loss.detach())
        step += 1
        if step % cfg.log_every == 0:
            rate = cfg.log_every / (time.time() - t0)
            mean_loss = running / cfg.log_every
            print(f"[train] step {step:>7d}/{cfg.n_steps}  loss {mean_loss:.4f}"
                  f"  lr {lr:.2e}  {rate:.1f} it/s")
            if wb_run is not None:
                # step= is explicit so a resumed leg writes at its TRUE global step
                # instead of restarting W&B's internal counter at 0.
                wb_run.log({"loss": mean_loss, "lr": lr, "it_per_s": rate}, step=step)
            running = 0.0
            t0 = time.time()
        if step % cfg.ckpt_every == 0:
            save_ckpt(latest, model=model, ema=ema, opt=opt, step=step, stats=stats,
                      cfg=cfg, coord_norm=coord_norm, wandb_id=wb_id)
            snap = cfg.snapshot_every or cfg.ckpt_every
            if step % snap == 0:
                save_ckpt(out / f"step_{step}.pt", model=model, ema=ema, opt=opt,
                          step=step, stats=stats, cfg=cfg, coord_norm=coord_norm,
                          wandb_id=wb_id)
            print(f"[train] checkpoint @ step {step}")

    save_ckpt(latest, model=model, ema=ema, opt=opt, step=step, stats=stats,
              cfg=cfg, coord_norm=coord_norm, wandb_id=wb_id)
    print(f"[train] done @ step {step} -> {latest}")
    if wb_run is not None:
        wb_run.finish()
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
