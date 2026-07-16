"""Sanity-check L137 coefficients: band 49-66 must land in ~50-140 hPa, monotone.

Run:  ../.venv/bin/python -m tests.test_l137
"""
import numpy as np

from src.eval.windeval.l137 import full_level_pressure


def run():
    levels = np.arange(49, 67)
    sp = 101325.0  # standard surface pressure (Pa)
    p_hpa = full_level_pressure(levels, sp).ravel() / 100.0

    print(f"{'level':>5} {'p [hPa]':>9}")
    for k, p in zip(levels, p_hpa):
        print(f"{k:5d} {p:9.2f}")

    checks = [
        (49 <= p_hpa[0] <= 56, f"top of band (lvl 49) ~50 hPa: {p_hpa[0]:.1f}"),
        (128 <= p_hpa[-1] <= 140, f"bottom of band (lvl 66) ~134 hPa: {p_hpa[-1]:.1f}"),
        (np.all(np.diff(p_hpa) > 0), "pressure increases monotonically with level index"),
        (p_hpa.min() >= 50 and p_hpa.max() <= 140, "entire band within 50-140 hPa"),
    ]
    ok = all(c for c, _ in checks)
    print()
    for c, msg in checks:
        print(f"{'PASS' if c else 'FAIL'}  {msg}")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
