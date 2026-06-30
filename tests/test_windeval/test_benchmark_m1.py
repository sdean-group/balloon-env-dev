"""Smoke-test the M1 leaderboard (Phase 3): trained denoiser swapped into the machinery.

Asserts the M1 story holds: the trained generator beats the toy on Axis-1 while KEEPING the
machinery's Axis-2 claims (seamless / exact-revisit / O(1)), and bounded peers stay Axis-2 N/A.

Needs the full pixi env (jax + torch + infinite-tensor + grib data) AND a trained checkpoint
(~588 MB, not in version control). SKIPs cleanly if either is missing.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_benchmark_m1.py
"""
from pathlib import Path

import numpy as np

CKPT = Path("runs/idiff_m1/step_84000.pt")

try:
    import torch  # noqa: F401
    from src.eval.windeval import benchmark_m1
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e


def run():
    if not HAVE:
        print(f"SKIP test_benchmark_m1 (deps not installed: {_ERR})")
        return True
    if not CKPT.exists():
        print(f"SKIP test_benchmark_m1 (trained checkpoint missing: {CKPT})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    report = benchmark_m1.run(str(CKPT))
    chk("report generated", isinstance(report, str) and "M1 Leaderboard" in report)
    chk("all verdict checks pass (no ❌)", "❌" not in report, f"{report.count('✅')} ✅")
    _ = np
    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
