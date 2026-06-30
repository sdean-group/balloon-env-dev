"""Smoke-test the M0 leaderboard runner (Phase 1e).

Runs the full board (ERA5 + BLE-VAE + toy InfiniteDiffusion + anchors) and asserts the
M0 story holds: the toy passes Axis-2 while peers/anchors are Axis-2 N/A, and Axis-1 is
sane. Needs the full pixi env (jax + torch + grib data). Skips cleanly if deps absent.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_benchmark_m0.py
"""
import numpy as np

try:
    import torch  # noqa: F401
    from src.eval.windeval import benchmark_m0
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e


def run():
    if not HAVE:
        print(f"SKIP test_benchmark_m0 (deps not installed: {_ERR})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    report = benchmark_m0.run()
    chk("report generated", isinstance(report, str) and "M0 Leaderboard" in report)
    # the runner's own verdict checks must all pass (rendered as ✅, none ❌)
    chk("all verdict checks pass (no ❌)", "❌" not in report,
        f"{report.count('✅')} ✅")
    chk("Axis-2 column present for toy", "PROC COMPOSITE" in report and "0.9" in report)
    chk("no stray warnings in report", "not recognized" not in report)
    _ = np  # keep import used if checks change
    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
