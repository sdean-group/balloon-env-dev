"""Regression test: vertical + temporal metrics must catch the incoherent anchors.

Reads the artifacts produced by benchmark_stage2 (run it first). Asserts the
separations that make the full suite robust where the spatial spectrum alone is not.

Run:  ../.venv/bin/python -m tests.test_stage2_metrics
"""
from pathlib import Path

from src.eval.windeval import artifact
from src.eval.windeval.metrics import vertical_scores, temporal_scores

DATA = Path(__file__).resolve().parents[2] / "src" / "eval" / "windeval" / "data"


def run():
    real = artifact.read(DATA / "era5_real_stage2.zarr")
    ps = artifact.read(DATA / "anchor_ps_stage2.zarr")
    no = artifact.read(DATA / "anchor_noise_stage2.zarr")

    rv, pv, nv = (vertical_scores(d)["score: vertical"] for d in (real, ps, no))
    rt, pt, nt = (temporal_scores(d)["score: temporal"] for d in (real, ps, no))

    checks = [
        (rv > 0.8, f"real vertical coherence high ({rv:.2f})"),
        (pv < 0.3 and nv < 0.3, f"anchors vertical coherence low (ps {pv:.2f}, noise {nv:.2f})"),
        (rt > 0.8, f"real temporal persistence high ({rt:.2f})"),
        (pt < 0.3 and nt < 0.3, f"anchors temporal persistence low (ps {pt:.2f}, noise {nt:.2f})"),
    ]
    ok = all(c for c, _ in checks)
    for c, msg in checks:
        print(f"{'PASS' if c else 'FAIL'}  {msg}")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
