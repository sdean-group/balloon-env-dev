"""T0 temporal baseline: AdvectedField (kinematic O(1) time axis) scaffolding.

Validates the route-agnostic temporal scaffolding that both learned routes (autoregressive,
spacetime) build on:
  - AdvectedField is O(1)-in-time (a coordinate shift), deterministic, and t=0 == the static
    field exactly (advection preserves the seamless/deterministic guarantees).
  - the temporal artifact carries a datetime64 time axis + temporally_evolving=True.

(Temporal *metric* calibration — SR_time / trajectory dispersion ranking real > shuffled —
lives in tests/test_windeval/test_metrics_v2.py since the benchmark-v2 overhaul.)

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_temporal_baseline.py
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    from src.eval.windeval import artifact
    from src.eval.windeval.generators.infinite_diffusion.advected import (
        AdvectedField, velocity_from_stats)
    from src.eval.windeval.generators.infinite_diffusion.trained import (
        build_sampler, TrainedWindowDenoiser)
    from src.eval.windeval.generators.infinite_diffusion import InfiniteDiffusionGenerator
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e

CKPT = "runs/idiff_m1/step_84000.pt"


def run():
    if not HAVE:
        print(f"SKIP test_temporal_baseline (deps not installed: {_ERR})")
        return True
    if not Path(CKPT).exists():
        print("SKIP test_temporal_baseline (ckpt missing)")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(ok)

    dev = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu"
    samp = build_sampler(CKPT, num_steps=8, window=64, seed=0, device=dev)
    phi = TrainedWindowDenoiser(CKPT, num_steps=8, device=dev)
    vel = velocity_from_stats(phi.stats)
    adv = AdvectedField(samp, vel, pixel_km=28.0, dt_seconds=3600.0)

    # t=0 must equal the static field exactly; advection is deterministic
    u0s, v0s = samp.field_uv(0, 48, 0, 48)
    u0a, v0a = adv.field_uv(0, 48, 0, 48, t=0)
    chk("t=0 == static field (advection preserves the base)", np.abs(u0s - u0a).max() == 0)
    a1 = adv.field_uv(0, 48, 0, 48, t=3)[0]
    a2 = adv.field_uv(0, 48, 0, 48, t=3)[0]
    chk("advected field deterministic in t", np.abs(a1 - a2).max() == 0)
    chk("t!=0 differs from t=0 (it actually evolves)", np.abs(a1 - u0a).max() > 0)

    # temporal artifact: datetime64 axis + temporally_evolving flag
    gen = InfiniteDiffusionGenerator(denoiser=phi, levels=phi.stats.levels.astype(float),
                                     window=64, stride=32, T=1, seed=0,
                                     name="infinite_diffusion_temporal", device=dev)
    out = Path(tempfile.mkdtemp()) / "temporal.zarr"
    gen.to_artifact(out, height=64, width=64, n_queries=8, n_times=6, dt_seconds=3600.0,
                    advect_vel=vel)
    ds = artifact.read(out)
    chk("artifact has time dim", ds.sizes.get("time", 0) == 6)
    chk("time axis is datetime64", np.issubdtype(ds["time"].dtype, np.datetime64))
    chk("temporally_evolving capability set",
        ds.attrs.get("capabilities", {}).get("temporally_evolving") is True)

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
