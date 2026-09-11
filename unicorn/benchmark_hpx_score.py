"""Score the HEALPix two-stage model with the summer benchmark suite (benchmark v2) and write the
poster-style table. Runs locally in the pixi env; the model's blocks come from
``unicorn/benchmark_hpx_blocks.py`` on Unicorn, dropped into ``src/eval/windeval/data/`` as
``idiff_m2cond_blocks_<tag>.npz`` (the summer cache format).

    PYTHONPATH=. .pixi/envs/default/bin/python unicorn/benchmark_hpx_score.py --rows hpx2stage hpx2era5 --out docs/benchmark-reports/benchmark_v2_hpx.md

Columns: self-split floor, simplex noise, ble_vae, the poster's summer model (cache 4yr_500000),
the summer coarse-conditioned model (m2coarse2_300k) with its coarse-upsampled baseline, and the
new rows. Every column is the same metric code against the same held-out NE Pacific reference.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.windeval import artifact
from src.eval.windeval.benchmark import (DATA, SPATIAL_STRIDE_H, _anchor_rows, _climatological_cond_groups, _coarse_upsample_artifacts,
                                         _cond_floor_groups, _conditional_artifacts, _level_hpa, _match_level)
from src.eval.windeval.baselines import baseline_rows
from src.eval.windeval.metrics import climatological_dz, run_suite
from src.eval.windeval.metrics.distributions import conditional_w1_grouped
from src.eval.windeval.reference import build_heldout, split

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="*", default=[], help="cache tags of new rows (idiff_m2cond_blocks_<tag>.npz)")
ap.add_argument("--names", nargs="*", default=[], help="display names for --rows")
ap.add_argument("--out", default="docs/benchmark-reports/benchmark_v2_hpx.md")
ap.add_argument("--cache", default="docs/benchmark-reports/benchmark_v2_hpx_scores.json", help="scores cache so reference columns are computed once")
ap.add_argument("--skip-reference", action="store_true")
a = ap.parse_args()

ref = artifact.read(build_heldout())
ref_sp = ref.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute()
half_a, half_b = split(ref)
a_sp = half_a.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute(); b_sp = half_b.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute()
dz = climatological_dz(DATA / "era5_real_stage2.zarr")
cache = Path(a.cache); scores = json.loads(cache.read_text()) if cache.exists() else {}

def score(name, pred, *, ref_s=None, ref_t=None, dz_s=None, groups=None, clim=False):
    t0 = time.time(); print(f"[score] {name} ...", flush=True)
    s, _ = run_suite(pred, ref_sp if ref_s is None else ref_s, dz=dz if dz_s is None else dz_s, ref_temporal=ref if ref_t is None else ref_t)
    if groups is not None:
        s.update(conditional_w1_grouped(groups))
    elif clim:
        s.update(conditional_w1_grouped(_climatological_cond_groups(pred, ref)))
    scores[name] = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)) for k, v in s.items()}
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(scores, indent=1))
    print(f"[score] {name} done in {time.time() - t0:.0f}s", flush=True)

if not a.skip_reference:
    if "self-split floor" not in scores:
        score("self-split floor", a_sp, ref_s=b_sp, ref_t=half_b, groups=_cond_floor_groups(ref))
    for name, ds in baseline_rows(a_sp, regen=False).items():
        if name == "simplex noise" and name not in scores:
            score(name, ds, clim=True)
    ble = DATA / "ble_vae_0.zarr"
    if ble.exists() and "ble_vae" not in scores:
        ble_ds = artifact.read(ble); idx = [_match_level(ref_sp, p) for p in _level_hpa(ble_ds)]
        dz_ble = np.array([float(dz[i:j].sum()) for i, j in zip(idx[:-1], idx[1:])])
        score("ble_vae", ble_ds, ref_s=ref_sp.isel(level=idx), dz_s=dz_ble if np.all(dz_ble > 0) else None)
    if "summer m2cond (poster)" not in scores:
        pooled, groups = _conditional_artifacts("x", ref, regen=False, cache_tag="4yr_500000"); score("summer m2cond (poster)", pooled, groups=groups)
    if "coarse upsampled" not in scores:
        up_pooled, up_groups = _coarse_upsample_artifacts(ref, 8); score("coarse upsampled", up_pooled, groups=up_groups)
    if "summer m2coarse2" not in scores:
        c_pooled, c_groups = _conditional_artifacts("x", ref, regen=False, cache_tag="m2coarse2_300k"); score("summer m2coarse2", c_pooled, groups=c_groups)
for tag, name in zip(a.rows, a.names or a.rows):
    if (DATA / f"idiff_m2cond_blocks_{tag}.npz").exists():
        pooled, groups = _conditional_artifacts("x", ref, regen=False, cache_tag=tag); score(name, pooled, groups=groups)
    else:
        print(f"[score] no blocks for {tag} yet", flush=True)

POSTER = [("W1 cond u (m/s)", "W1(u | c) (m/s)"), ("W1 cond v (m/s)", "W1(v | c) (m/s)"), ("W1 speed (m/s)", "W1(|V|) (m/s)"),
          ("tail err 0.1% (m/s)", "tail error 0.1% (m/s)"), ("SR_E", "SR_E"), ("SR_div", "SR_div"), ("SR_vort", "SR_vort"), ("L_eff (km)", "L_eff (km)"),
          ("W1 shear u ((m/s)/km)", "W1 shear u"), ("W1 shear v ((m/s)/km)", "W1 shear v"), ("SR_time", "SR_time"), ("final spread ratio", "final spread ratio (1 = ERA5)"),
          ("opp-wind frac", "opposing-wind fraction (ERA5 0.20)"), ("jet speed err 99.9% (m/s)", "jet speed error 99.9% (m/s)"), ("disp log-MSD RMSE", "trajectory dispersion RMSE")]
cols = [c for c in ["self-split floor", "simplex noise", "ble_vae", "summer m2cond (poster)", "coarse upsampled", "summer m2coarse2", *(a.names or a.rows)] if c in scores]
def fmt(v): return "N/A" if v is None else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}")
lines = ["# Benchmark v2 with the HEALPix two-stage model", "", "Same metric code, same held-out NE Pacific reference (days 8-14 of Jan/Apr/Jul/Oct 2023), same 64x64 window and (month, day, hour) conditions as the poster. New rows are the global two-stage model regridded onto the window. Lower is better unless stated.", "",
         "| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
for key, label in POSTER:
    lines.append(f"| {label} | " + " | ".join(fmt(scores[c].get(key)) for c in cols) + " |")
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
