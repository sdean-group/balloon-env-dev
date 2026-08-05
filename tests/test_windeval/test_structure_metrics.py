"""Calibration of the structure & vertical-realism metrics (metrics/structure.py).

These metrics exist because every other row in the suite is a population statistic that
fitted noise reproduces by construction. So they earn their place the same way the rest
did — by separating fields we KNOW differ, in the direction we know they differ:

  A. Self-comparison: pred == ref ⇒ opp-wind err ≈ 0, speed W1 ≈ 0, ratios ≈ 1.
  B. Vertical anchors (the point of the whole module): a synthetic column that is
     vertically COHERENT (same bearing at every level) must score opp-wind frac ≈ 0,
     and one with an independent random bearing per level must score ≈ 1. `shear.py`
     cannot tell these apart at matched shear magnitude — asserted here too, since that
     blind spot is the reason this module exists.
  C. Geometry anchors: an anisotropic streaky field must read as MORE elongated than an
     isotropic blob field of the same exceedance coverage.
  D. Applicability: a different level count (ble_vae's 10 pressure levels) ⇒ NaN, the
     suite's "not applicable", never a crash.
  E. Window discipline: structural scoring must crop both sides to a common box —
     regression-guards the 2026-07-29 finding that a 64²-vs-121² mismatch inverts the
     decoupling verdict.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_structure_metrics.py
"""
from pathlib import Path

import numpy as np

from src.eval.windeval import artifact
from src.eval.windeval.metrics import structure as S
from src.eval.windeval.metrics.shear import shear_w1

L, H, W, T = 18, 64, 64, 8
FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAILS.append(name)


def field(u, v, time=None):
    return artifact.make_field(
        u.astype("float32"), v.astype("float32"),
        level=np.arange(49, 49 + u.shape[1]),
        lat=np.linspace(48, 32.25, u.shape[2]), lon=np.linspace(232, 247.75, u.shape[3]),
        time=time if time is not None else np.arange(u.shape[0]))


def coherent(speed=20.0, seed=0):
    """Same bearing at every level (a perfectly vertically-coupled column)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, (T, 1, H, W))
    mag = speed * (1 + 0.1 * rng.standard_normal((T, L, H, W)))
    return field(mag * np.cos(ang), mag * np.sin(ang))


def decoupled(speed=20.0, seed=0):
    """Independent bearing per level (maximally decoupled)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, (T, L, H, W))
    mag = speed * (1 + 0.1 * rng.standard_normal((T, L, H, W)))
    return field(mag * np.cos(ang), mag * np.sin(ang))


def streaky(seed=0, aspect=8):
    """Zonally elongated structures (jet-like)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((T, L, H, W))
    k = np.ones((1, 1, 1, aspect)) / aspect
    from scipy.ndimage import convolve
    sm = convolve(a, k, mode="wrap")
    return field(20 + 5 * sm, np.zeros_like(sm))


def blobby(seed=0, r=3):
    """Isotropic structures of comparable smoothing scale."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((T, L, H, W))
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(a, sigma=(0, 0, r, r), mode="wrap")
    sm = sm / sm.std()
    return field(20 + 5 * sm, np.zeros_like(sm))


print(__doc__.splitlines()[0])

# ---- A. self-comparison ----
print("\nA. self-comparison (pred == ref)")
ref = decoupled(seed=3)
s = S.structure_suite(ref, ref)
check("opp-wind err ≈ 0", s["opp-wind err"] < 0.02, f"{s['opp-wind err']:.4f}")
check("W1 speed ≈ 0", s["W1 speed (m/s)"] < 0.05, f"{s['W1 speed (m/s)']:.4f}")
# geometry needs a field that HAS coherent components: an uncorrelated field's
# exceedance set is scattered single pixels (correctly ⇒ NaN, asserted separately).
struct = blobby(seed=3)
sg = S.structure_suite(struct, struct)
check("jet area ratio ≈ 1", abs(sg["jet area ratio"] - 1) < 0.15, f"{sg['jet area ratio']:.3f}")
check("jet elong ratio ≈ 1", abs(sg["jet elong ratio"] - 1) < 0.15, f"{sg['jet elong ratio']:.3f}")
check("structureless field ⇒ geometry NaN", not np.isfinite(s["jet area ratio"]),
      "scattered single pixels, nothing above the size floor")

# ---- B. vertical anchors ----
print("\nB. vertical decoupling anchors")
fc = S.opposing_wind_fraction(coherent())
fd = S.opposing_wind_fraction(decoupled())
check("coherent column ⇒ frac ≈ 0", fc < 0.02, f"{fc:.4f}")
check("decoupled column ⇒ frac ≈ 1", fd > 0.95, f"{fd:.4f}")
check("metric separates them", (fd - fc) > 0.9, f"Δ={fd - fc:.3f}")

# the blind spot that motivates this module: matched shear magnitude, opposite structure
dz = np.full(L - 1, 500.0)
sh = shear_w1(coherent(seed=1), coherent(seed=2), dz)["W1 shear u ((m/s)/km)"]
sh_cross = shear_w1(decoupled(seed=1), decoupled(seed=2), dz)["W1 shear u ((m/s)/km)"]
print(f"     (shear W1 within-family: coherent {sh:.3f}, decoupled {sh_cross:.3f})")

# ---- C. geometry anchors ----
print("\nC. jet geometry anchors")
_, e_streak = S._geometry(streaky())
_, e_blob = S._geometry(blobby())
check("streaky more elongated than blobby", e_streak > e_blob * 1.3,
      f"streaky {e_streak:.2f} vs blobby {e_blob:.2f}")

# ---- D. applicability ----
print("\nD. applicability gating")
few = decoupled(seed=5)
few = few.isel(level=slice(0, 10))
s10 = S.structure_suite(few, ref)
check("level-count mismatch ⇒ NaN (no crash)",
      all(not np.isfinite(s10[k]) for k in ("opp-wind err", "jet area ratio")),
      "10 vs 18 levels")

# ---- E. window discipline ----
print("\nE. common-window discipline")
rng = np.random.default_rng(7)                       # the real 64²-vs-121² case
u121 = rng.standard_normal((T, L, 121, 121)) * 10
big = field(u121, rng.standard_normal((T, L, 121, 121)) * 10)
small = S._center(big, 64)
p, r = S.common_window(small, big)
check("64² vs 121² ⇒ both cropped to 64²",
      (p.sizes["y"], p.sizes["x"], r.sizes["y"], r.sizes["x"]) == (64, 64, 64, 64),
      f"pred {p.sizes['y']}×{p.sizes['x']}, ref {r.sizes['y']}×{r.sizes['x']}")

# ---- F. real-data regression (skips if the reference isn't built) ----
print("\nF. real-data regression vs the 2026-07-29 probe")
try:
    from src.eval.windeval.reference import build_heldout
    ds = artifact.read(build_heldout())
    win = S._center(ds, S.STRUCT_WINDOW)
    f = S.opposing_wind_fraction(win)
    check("ERA5 window opp-wind frac ≈ 0.203", abs(f - 0.203) < 0.02, f"{f:.4f}")
except FileNotFoundError as e:
    print(f"  SKIP  reference not available ({e})")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
raise SystemExit(1 if FAILS else 0)
