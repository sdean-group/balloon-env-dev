"""Phase-2 pipeline smoke test: net -> data -> train -> trained Phi -> machinery swap -> gate.

Validates that the *model* half is wired end-to-end and drops into the *machinery* half
with zero changes. NOT a quality test (a few-step overfit on the 24-step slice can't learn
the distribution) — it asserts shapes, finiteness, determinism, and the seamless-lattice
invariants still hold when the trained denoiser replaces the toy.

Needs the full pixi env (torch + infinite-tensor + grib data). Skips cleanly otherwise.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_phase2_pipeline.py
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    from src.eval.windeval.generators.infinite_diffusion import net as netmod
    from src.eval.windeval.generators.infinite_diffusion import data as datamod
    from src.eval.windeval.generators.infinite_diffusion import train as trainmod
    from src.eval.windeval.generators.infinite_diffusion.trained import (
        TrainedWindowDenoiser, build_sampler)
    from src.eval.windeval.generators.infinite_diffusion.denoiser import WindowDenoiser
    from src.eval.windeval.generators.infinite_diffusion import gate as gatemod
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e

DATA = "src/eval/windeval/data/era5_real.zarr"


def run():
    if not HAVE:
        print(f"SKIP test_phase2_pipeline (deps not installed: {_ERR})")
        return True
    if not Path(DATA).exists():
        print(f"SKIP test_phase2_pipeline (data missing: {DATA})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    # --- net: EDM forward/loss/backward, in/out channels = 2L ---
    C = 12
    m = netmod.EDMPrecond(C, net_kwargs=dict(model_channels=16, channel_mult=(1, 2),
                                             num_res_blocks=1, attn_resolutions=()))
    x = torch.randn(2, C, 16, 16)
    D = m(x, torch.rand(2) * 5 + 0.1)
    chk("net D(x,sigma) shape + finite", D.shape == x.shape and torch.isfinite(D).all())
    loss = m.loss(x)
    loss.backward()
    chk("net loss finite + backward", torch.isfinite(loss).item())

    # --- data: normalised crops are ~zero-mean unit-var + round-trip ---
    ds = datamod.WindCropDataset(DATA, crop=16, levels=(49, 66), length=64, augment=True)
    xs = torch.stack([ds[i] for i in range(64)])
    chk("crops ~zero-mean unit-var", abs(float(xs.mean())) < 0.2 and 0.7 < float(xs.std()) < 1.3,
        f"mean {float(xs.mean()):.2f} std {float(xs.std()):.2f}")
    one = ds[0]
    rt = ds.stats.normalize(ds.stats.denormalize(one))
    chk("norm round-trip", float((rt - one).abs().max()) < 1e-4)

    with tempfile.TemporaryDirectory() as tmp:
        # --- train: a handful of steps -> checkpoint ---
        cfg = trainmod.TrainConfig(
            data_path=DATA, crop=16, levels=(49, 66), model_channels=16,
            channel_mult=(1, 2), num_res_blocks=1, attn_resolutions=(),
            batch_size=8, n_steps=12, warmup_steps=4, num_workers=0,
            out_dir=tmp, ckpt_every=12, log_every=12, device="cpu", seed=0)
        ckpt = trainmod.train(cfg)
        chk("training wrote checkpoint", Path(ckpt).exists())

        # --- trained Phi: protocol + finite sample ---
        phi = TrainedWindowDenoiser(ckpt, num_steps=6, device="cpu")
        chk("Phi satisfies WindowDenoiser protocol", isinstance(phi, WindowDenoiser))
        chk("Phi n_channels = 2*levels", phi.n_channels == 2 * phi.n_levels)
        win = phi(torch.randn(phi.n_channels, 16, 16))
        chk("Phi(window) finite + same shape",
            win.shape == (phi.n_channels, 16, 16) and torch.isfinite(win).all())

        # --- swap into UNMODIFIED machinery: finite + deterministic + region-invariant ---
        samp = build_sampler(ckpt, num_steps=6, window=16, device="cpu", cache_bytes=None)
        a = samp.materialize(0, 24, 0, 24).numpy()
        chk("machinery materialize finite", np.isfinite(a).all())
        samp.clear_cache()
        a2 = samp.materialize(0, 24, 0, 24).numpy()
        chk("machinery deterministic (max|d|=0)", float(np.abs(a - a2).max()) == 0.0)
        b = samp.materialize(8, 32, 8, 32).numpy()
        ov = float(np.abs(a[:, 8:24, 8:24] - b[:, 0:16, 0:16]).max())
        chk("region-invariant on overlap (seamless)", ov == 0.0, f"max|d|={ov:g}")

        # --- gate: runs + returns finite scores ---
        r = gatemod.gate(ckpt, n=4, size=16, num_steps=6, device="cpu")
        chk("Axis-1 gate runs + finite COMPOSITE",
            r["finite"] and np.isfinite(r["COMPOSITE"]),
            f"COMPOSITE {r['COMPOSITE']:.2f} (quality not asserted)")

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
