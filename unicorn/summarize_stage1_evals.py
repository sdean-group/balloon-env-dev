"""Print the Stage 1 gate results of a run as one markdown table (rows = eval_* dirs).

    python unicorn/summarize_stage1_evals.py /scratch/sps252/runs/stage1_hpx32
"""
import json, re, sys
from pathlib import Path

run = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/sps252/runs/stage1_hpx32")
rows = []
for d in sorted(run.glob("eval_*")):
    f = d / "metrics.json"
    if not f.exists():
        continue
    m = json.loads(f.read_text())
    step = m.get("step", -1); churn = m.get("s_churn", 0.0); steps = m.get("num_steps", 18)
    if d.name.startswith("eval_best"):
        step = f"best({step})"
    rows.append((d.name, step, churn, steps, m))

def key(r):
    s = r[4].get("step", -1); return (s, r[3], r[2])
rows.sort(key=key)
if not rows:
    sys.exit("no metrics found")
ref = rows[-1][4]; b = ref.get("spectrum_logratio_by_band", {})
cols = [("SR spectrum |logr| (floor %.3f)" % ref["SR_sphere_floor"], lambda m: m["SR_sphere"]),
        ("logr l40-70 / l70-96", lambda m: "%.2f / %.2f" % (m["spectrum_logratio_by_band"]["l40-70"], m["spectrum_logratio_by_band"]["l70-96"])),
        ("W1 u (floor %.2f)" % ref["W1_u_floor"], lambda m: m["W1_u"]),
        ("speed p99.9 (ERA5 %.1f)" % ref["speed_p999_era5"], lambda m: m["speed_p999_model"]),
        ("zonal corr top/bot", lambda m: "%.3f / %.3f" % (m["zonal_top_corr"], m["zonal_bottom_corr"])),
        ("jet top SH (ERA5 %.1f)" % ref["jet_top_SH_era5"], lambda m: m["jet_top_SH_model"]),
        ("jet bot NH (ERA5 %.1f)" % ref["jet_bottom_NH_era5"], lambda m: m["jet_bottom_NH_model"]),
        ("tropics top (ERA5 %+.1f)" % ref["tropics_top_era5"], lambda m: m["tropics_top_model"]),
        ("seed spread (ERA5 day %.1f)" % ref["era5_day_spread_rms"], lambda m: m["seed_spread_rms"]),
        ("coverage", lambda m: m["coverage"]),
        ("6h corr (ERA5 %.3f)" % ref["adjacent_frame_corr_era5"], lambda m: m["adjacent_frame_corr_model"])]
print("| step | steps | churn | " + " | ".join(c for c, _ in cols) + " |")
print("|---|---|---|" + "---|" * len(cols))
for name, step, churn, steps, m in rows:
    vals = []
    for _, fn in cols:
        v = fn(m); vals.append(v if isinstance(v, str) else ("%.3f" % v if abs(v) < 1 else "%.2f" % v))
    print(f"| {step} | {steps} | {churn:g} | " + " | ".join(vals) + " |")
