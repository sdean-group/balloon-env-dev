"""Validate the InfiniteDiffusion -> WindArtifact adapter + viz (Phase 1c).

Materialize a crop -> read it back -> confirm schema/capabilities/seam/querylog ->
run an Axis-1 metric on it (it must be metric-ready) -> check revisit determinism via the
querylog -> render the figures. Needs torch + infinite-tensor + matplotlib; skips cleanly
if torch/infinite-tensor are absent.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_idiff_artifact.py
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    from src.eval.windeval import artifact
    from src.eval.windeval.metrics import run_suite
    from src.eval.windeval.generators.infinite_diffusion import (
        InfiniteDiffusionGenerator, viz,
    )
    HAVE_TORCH = True
except ImportError as e:  # pragma: no cover
    HAVE_TORCH = False
    _IMPORT_ERR = e


def run():
    if not HAVE_TORCH:
        print(f"SKIP test_idiff_artifact (torch/infinite-tensor not installed: {_IMPORT_ERR})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gen = InfiniteDiffusionGenerator(n_levels=8, window=64, stride=32, T=2, seed=0)
        path = tmp / "idiff.zarr"
        gen.to_artifact(path, height=160, width=160, n_queries=42)

        ds = artifact.read(path)
        caps = ds.attrs["capabilities"]
        chk("field/ schema", ds["u"].dims == ("time", "level", "y", "x")
            and ds["u"].shape == (1, 8, 160, 160))
        chk("capabilities unbounded/tiled/random_access",
            caps["extent"] == "unbounded" and caps["tiled"] and caps["random_access"])
        seams = ds.attrs["seam_boundaries"]
        chk("seam_boundaries are in-bounds local indices",
            len(seams["x"]) > 0 and all(0 < i < 160 for i in seams["x"]))
        chk("hardware recorded", "device" in ds.attrs["hardware"])

        sc, _ = run_suite(ds, ds)   # self-comparison: residuals must be ~0
        chk("metric suite runs on artifact (self-SR ≈ 0)",
            np.isfinite(sc["SR_E"]) and sc["SR_E"] < 1e-6 and sc["W1 u (m/s)"] < 1e-6,
            f"SR_E={sc['SR_E']:.2e}, W1_u={sc['W1 u (m/s)']:.2e}")

        chk("querylog present", artifact.has_querylog(path))
        q = artifact.read_querylog(path)
        key = list(zip(q["x"].values.tolist(), q["y"].values.tolist(),
                       q["level"].values.tolist(), q["seed"].values.tolist()))
        u, v = q["u"].values, q["v"].values
        seen, maxdiff, n_rev = {}, 0.0, 0
        for i, k in enumerate(key):
            if k in seen:
                j = seen[k]
                maxdiff = max(maxdiff, abs(u[i] - u[j]), abs(v[i] - v[j]))
                n_rev += 1
            else:
                seen[k] = i
        chk("revisit determinism via querylog", n_rev > 0 and maxdiff <= q.attrs["revisit_tolerance"],
            f"{n_rev} revisits, max|Δ|={maxdiff:.1e}")
        chk("budget latency finite & bounded", np.all(np.isfinite(q["latency_s"].values))
            and q["latency_s"].values.max() < 5.0)

        f1 = viz.plot_field(gen.sampler, 0, 192, 0, 192, level=4, out=tmp / "field.png")
        f2 = viz.plot_zoom_montage(gen.sampler, center=(0, 0), sizes=(256, 64, 16),
                                   level=4, out=tmp / "zoom.png")
        chk("viz renders", Path(f1).stat().st_size > 5000 and Path(f2).stat().st_size > 5000)

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
