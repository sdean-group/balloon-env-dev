"""Temporal leaderboard harness — generates a report and ranks coherent > incoherent.

Guards the benchmark_temporal wiring with the always-available rows (ERA5 peer + shuffled
anchor + the kinematic toy when the static ckpt is present). The learned M3/M2 rows are
exercised by their own train+sample tests; here we assert the harness produces a report and
the reference ranking holds (peer realism > shuffled), which must be true regardless of which
generators are present.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_benchmark_temporal.py
"""
from pathlib import Path

try:
    import torch  # noqa: F401
    from src.eval.windeval import benchmark_temporal as bt
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e

ERA5 = "src/eval/windeval/data/era5_real.zarr"


def run():
    if not HAVE:
        print(f"SKIP test_benchmark_temporal (deps not installed: {_ERR})")
        return True
    if not Path(ERA5).exists():
        print("SKIP test_benchmark_temporal (era5_real.zarr missing)")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(ok)

    # no M3/M2 ckpts: exercises era5 + shuffled (+ toy if the static ckpt happens to be present)
    report = bt.run(n_times=12, crop=32)
    chk("report generated", isinstance(report, str) and "Temporal Leaderboard" in report)

    verdict = report.split("## Verdict")[-1]
    chk("reference ranking holds (peer realism > shuffled)",
        "✅ peer realism" in verdict)
    chk("peer flagged temporally coherent", "✅ peer is temporally coherent" in verdict)
    chk("shuffled flagged incoherent", "✅ shuffled anchor is incoherent" in verdict)
    # no spurious failures among the always-on (non-learned) checks
    always_on = [ln for ln in verdict.splitlines()
                 if ln.strip().startswith(("- ✅", "- ❌"))
                 and "autoregressive" not in ln and "spacetime" not in ln]
    chk("no failures among reference checks", all("✅" in ln for ln in always_on),
        f"{sum('✅' in ln for ln in always_on)}/{len(always_on)} pass")

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
