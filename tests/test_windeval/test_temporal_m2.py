"""M2 joint-spacetime route: factorized denoiser + WindSpaceTimeDataset + block sampler.

Validates the *plumbing* (not realism — needs a Kahan train): a tiny SPACETIME model trained
on contiguous blocks of era5_train.zarr samples a τ-frame block that
  - is FINITE with the right shape (τ, L, H, W),
  - has frames that DIFFER (the block isn't τ copies of one frame),
  - is deterministic given the seed,
and that the learned temporal coupling makes *consecutive* frames within a sampled block more
correlated than frames drawn from two INDEPENDENT blocks (the TemporalConv does something).
WindSpaceTimeDataset must never build a block that straddles a time discontinuity.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_temporal_m2.py
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    import xarray as xr
    from src.eval.windeval.generators.infinite_diffusion.train import TrainConfig, train
    from src.eval.windeval.generators.infinite_diffusion.data import (
        WindSpaceTimeDataset, _time_blocks)
    from src.eval.windeval.generators.infinite_diffusion.spacetime import SpaceTimeSampler
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e

DATA = "src/eval/windeval/data/era5_train.zarr"


def _consec_corr(us, vs):
    """Mean correlation between consecutive frames of a (τ,L,H,W) block."""
    cs = []
    for t in range(us.shape[0] - 1):
        for a in (us, vs):
            for k in range(a.shape[1]):
                cs.append(np.corrcoef(a[t, k].ravel(), a[t + 1, k].ravel())[0, 1])
    return float(np.nanmean(cs))


def run():
    if not HAVE:
        print(f"SKIP test_temporal_m2 (deps not installed: {_ERR})")
        return True
    if not Path(DATA).exists():
        print("SKIP test_temporal_m2 (era5_train.zarr missing)")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(ok)

    dev = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu"

    # --- cross-block masking: no block straddles a discontinuity ---
    ds = xr.open_zarr(DATA, consolidated=False, zarr_format=2)
    times = ds["time"].values
    blocks = _time_blocks(times)
    TAU = 4
    sds = WindSpaceTimeDataset(DATA, crop=32, levels=(55, 62), n_frames=TAU, length=10, seed=0)
    block_of = np.full(len(times), -1)
    for bi, (a, b) in enumerate(blocks):
        block_of[a:b] = bi
    span = TAU - 1
    ok_mask = all(block_of[t] >= 0 and block_of[t] == block_of[t + span] for t in sds.block_starts)
    chk("WindSpaceTimeDataset masks cross-block blocks", ok_mask,
        f"{len(blocks)} blocks, {len(sds.block_starts)} valid block-starts")

    # --- tiny spacetime train ---
    out = Path(tempfile.mkdtemp()) / "m2"
    cfg = TrainConfig(
        data_path=DATA, crop=32, levels=(55, 62), spacetime=True, n_frames=TAU, frame_stride=1,
        temporal_kernel=3, model_channels=16, channel_mult=(1, 2), num_res_blocks=1,
        attn_resolutions=(), batch_size=4, lr=2e-3, n_steps=250, warmup_steps=50, ema_decay=0.99,
        ckpt_every=250, log_every=250, num_workers=0, out_dir=str(out), device=dev, resume=False,
    )
    ckpt = train(cfg)
    chk("spacetime training wrote a checkpoint", Path(ckpt).exists())

    samp = SpaceTimeSampler(ckpt, num_steps=12, device=dev)
    us, vs = samp.sample_block((48, 48), seed=0)
    chk("block shape (τ,L,H,W)", us.shape == (TAU, 8, 48, 48), f"{us.shape}")
    chk("block finite", bool(np.isfinite(us).all() and np.isfinite(vs).all()))
    chk("frames differ within block", np.abs(us[1] - us[0]).max() > 1e-3)

    us2, vs2 = samp.sample_block((48, 48), seed=0)
    chk("block deterministic given seed", np.abs(us - us2).max() == 0)

    # temporal coupling: consecutive frames in ONE block vs frames across TWO independent blocks
    ua, va = samp.sample_block((48, 48), seed=1)
    ub, vb = samp.sample_block((48, 48), seed=2)
    within = _consec_corr(us, vs)
    cross = float(np.nanmean([np.corrcoef(ua[t, k].ravel(), ub[t, k].ravel())[0, 1]
                              for t in range(TAU) for k in range(ua.shape[1])]))
    chk("temporal coupling: within-block > cross-block correlation", within > cross,
        f"within {within:.3f} > cross {cross:.3f}")

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
