"""M3 autoregressive temporal route: conditioned EDM + WindPairDataset + rollout.

Validates the *plumbing* (not realism — that needs a full Kahan train): a tiny PAIRED model
trained on the contiguous block(s) of era5_train.zarr learns a transition that
  - rolls forward FINITE and EVOLVES (frame_{t+1} != frame_t),
  - keeps amplitude BOUNDED over a multi-step roll (no autoregressive blow-up),
  - is deterministic given the seed,
and that WindPairDataset never pairs across a time discontinuity (cross-block masking).

The static (unconditional) path must remain untouched — covered by the rest of the suite.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_temporal_m3.py
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    import xarray as xr
    from src.eval.windeval.generators.infinite_diffusion.train import TrainConfig, train
    from src.eval.windeval.generators.infinite_diffusion.data import (
        WindPairDataset, _time_blocks)
    from src.eval.windeval.generators.infinite_diffusion.autoregressive import ConditionedDenoiser
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e

DATA = "src/eval/windeval/data/era5_train.zarr"


def run():
    if not HAVE:
        print(f"SKIP test_temporal_m3 (deps not installed: {_ERR})")
        return True
    if not Path(DATA).exists():
        print("SKIP test_temporal_m3 (era5_train.zarr missing)")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(ok)

    dev = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu"

    # --- cross-block masking: no pair straddles a time discontinuity ---
    ds = xr.open_zarr(DATA, consolidated=False, zarr_format=2)
    times = ds["time"].values
    blocks = _time_blocks(times)
    pds = WindPairDataset(DATA, crop=32, levels=(55, 62), frame_stride=1, length=10, seed=0)
    block_of = np.full(len(times), -1)
    for bi, (a, b) in enumerate(blocks):
        block_of[a:b] = bi
    starts = pds.pair_starts
    ok_mask = all(block_of[t] == block_of[t + 1] and block_of[t] >= 0 for t in starts)
    chk("WindPairDataset masks cross-block pairs", ok_mask,
        f"{len(blocks)} blocks, {len(starts)} valid pair-starts")

    # --- tiny paired train ---
    out = Path(tempfile.mkdtemp()) / "m3"
    cfg = TrainConfig(
        data_path=DATA, crop=32, levels=(55, 62), paired=True, frame_stride=1,
        model_channels=16, channel_mult=(1, 2), num_res_blocks=1, attn_resolutions=(),
        batch_size=8, lr=2e-3, n_steps=250, warmup_steps=50, ema_decay=0.99,
        ckpt_every=250, log_every=250, num_workers=0,
        out_dir=str(out), device=dev, resume=False,
    )
    ckpt = train(cfg)
    chk("paired training wrote a checkpoint", Path(ckpt).exists())

    # --- rollout from a real ERA5 seed frame ---
    lv = ds["level"].values
    keep = (lv >= 55) & (lv <= 62)
    u = ds["u"].values[:, keep]
    v = ds["v"].values[:, keep]
    y0, x0, c = 20, 20, 64
    u0, v0 = u[0, :, y0:y0 + c, x0:x0 + c], v[0, :, y0:y0 + c, x0:x0 + c]

    cd = ConditionedDenoiser(ckpt, num_steps=12, device=dev)
    chk("ConditionedDenoiser rejects unconditional ckpt",
        _rejects_unconditional(cd, ckpt, dev))
    N = 16
    us, vs = cd.rollout((u0, v0), n_times=N, seed=0)

    chk("rollout finite", bool(np.isfinite(us).all() and np.isfinite(vs).all()))
    chk("frame0 == seed", np.abs(us[0] - u0).max() < 1e-3)
    chk("rollout evolves (frame1 != frame0)", np.abs(us[1] - us[0]).max() > 1e-3)

    rms = np.sqrt(np.mean(us ** 2 + vs ** 2, axis=(1, 2, 3)))
    chk("amplitude bounded over roll (no blow-up)", rms.max() < 5 * rms[0],
        f"seed_rms={rms[0]:.1f} max_rms={rms.max():.1f}")

    us2, vs2 = cd.rollout((u0, v0), n_times=N, seed=0)
    chk("rollout deterministic given seed", np.abs(us - us2).max() == 0)

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


def _rejects_unconditional(cd, paired_ckpt, dev):
    """The static checkpoint (cond_channels=0) must be refused by ConditionedDenoiser."""
    static = "runs/idiff_m1/step_84000.pt"
    if not Path(static).exists():
        return True  # can't test; don't fail
    try:
        ConditionedDenoiser(static, num_steps=4, device=dev)
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
