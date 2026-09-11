"""Generate the HEALPix two-stage model's blocks at the summer benchmark's conditions and save
them in the summer cache format (``idiff_m2cond_blocks_<tag>.npz``), so ``windeval.benchmark``
scores them with the same suite against the same held-out reference.

    python unicorn/benchmark_hpx_blocks.py --mode stage1 --out /scratch/sps252/runs/bench/idiff_m2cond_blocks_hpx2stage.npz
    python unicorn/benchmark_hpx_blocks.py --mode era5   --out /scratch/sps252/runs/bench/idiff_m2cond_blocks_hpx2era5.npz

Conditions: months 1/4/7/10, days 8-14, hours 00 and 12 UTC, 2 seeds; one 13-hour block from
the condition hour; fields regridded onto the benchmark's fixed 64x64 window (0.25 deg, lat
48.0..32.25 N descending, lon 232.0..247.75 E). ``stage1``: Stage 1 draws the coarse frames for
the date (the full generator, the analogue of the poster's "Ours"); ``era5``: the held-out ERA5
block means drive Stage 2 (the analogue of "idiff m2coarse", asymmetric by construction).
"""
import argparse, os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, zarr, healpy as hp
sys.path.insert(0, "src/eval/windeval/hpx")
from sample_fine import HpxFineSampler
from sample import HpxSampler
from eval_coarse import _slow_series

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("stage1", "era5"), required=True); ap.add_argument("--out", required=True)
ap.add_argument("--stage2", default="/scratch/sps252/runs/stage2_hpx256_bl/step_200000.pt")
ap.add_argument("--stage1", default="/home/sps252/data/models/stage1_hpx32_ft/step_60000.pt")
ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--days", default="8-14")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
lay = os.path.expanduser("~/data/hpx_layout")
LAT = 48.0 - 0.25 * np.arange(64); LON = 232.0 + 0.25 * np.arange(64)
GLON, GLAT = np.meshgrid(LON, LAT)                                   # (64, 64) window, lat descending
REGION = (float(LAT.min()), float(LAT.max()), float(LON.min()), float(LON.max()))
d0, d1 = (int(v) for v in a.days.split("-")); days = list(range(d0, d1 + 1))
conds = [(m, d, h) for m in (1, 4, 7, 10) for h in (0, 12) for d in days]
root = zarr.open_group("/scratch/sps252/era5_hpx_heldout_2023", mode="r"); hours = np.asarray(root["time"][:], dtype=np.int64)
row = {int(h): i for i, h in enumerate(hours)}
train = zarr.open_group("/scratch/sps252/era5_hpx", mode="r"); trow = {int(h): i for i, h in enumerate(np.asarray(train["time"][:], dtype=np.int64))}
def coarse_at(h):
    """Coarse frame from the held-out store, else from the training store (the 12 UTC condition on
    day 14 needs the frame at day 15 00:00, which is a training day). Only ever a conditioner."""
    return root["coarse/uv"][row[h]] if h in row else train["coarse/uv"][trow[h]]
s2 = HpxFineSampler(a.stage2, lay, num_steps=18, device="cuda", batch_patches=48, region=REGION)
print(f"regional sampler: {s2.n_patches} active patches; window {REGION}", flush=True)
s1 = sh = su = None
if a.mode == "stage1":
    s1 = HpxSampler(a.stage1, lay, num_steps=18, device="cuda")
    sh, su = _slow_series(["/scratch/sps252/era5_hpx", "~/data/era5_hpx_coarse_2020_2021"], s1.nside)
def regrid(x):                                                        # (τ, C, npix) NEST -> (τ, L, 2, 64, 64)
    tau, C = x.shape[:2]; L = C // 2; out = np.zeros((tau, L, 2, 64, 64), np.float32)
    for f in range(tau):
        for c in range(C):
            m = hp.reorder(x[f, c].astype(np.float64), n2r=True)
            out[f, c // 2, c % 2] = hp.get_interp_val(m, GLON, GLAT, lonlat=True)    # channels interleave (u_l, v_l)
    return out
blocks, times, month, day, hour, seed_idx = [], [], [], [], [], []
s1_cache = {}
t_all = time.time()
for i, (m, d, h) in enumerate(conds):
    t0 = np.datetime64(f"2023-{m:02d}-{d:02d}T{h:02d}", "h"); h0 = int((t0 - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
    hs = h0 + np.arange(13); ts = t0 + np.arange(13).astype("timedelta64[h]")
    for s in range(a.seeds):
        seed = i * a.seeds + s; t1 = time.time()
        if a.mode == "stage1":
            # one Stage 1 block per (month, day, seed), starting 00 UTC and spanning 48 h, drives both the
            # 00 and the 12 UTC condition of that day (frames 0/6/12 h and 12/18/24 h). Independent draws
            # per condition made the day's two 13-hour blocks disagree at their shared hour, which the
            # temporal rows read as a jump (SR_time 3.08 vs 0.40 with ERA5 frames; 2026-09-11).
            key = (m, d, s)
            if key not in s1_cache:
                hd = h0 - h; mk = (sh >= hd - 720) & (sh < hd); sv = float(su[mk].mean()) if mk.sum() >= 360 else None
                s1_cache[key] = s1.sample_block(hd + 6 * np.arange(8), seed=i * a.seeds + s, slow_value=sv)
            cf = s1_cache[key][h // 6: h // 6 + 3]
        else:
            cf = np.stack([coarse_at(h0 + 6 * k) for k in range(3)]).astype(np.float32)
        gen = s2.sample_block(cf, hs, seed=1000 + seed, log=None)
        blocks.append(regrid(gen)); times.append(ts); month.append(m); day.append(d); hour.append(h); seed_idx.append(s)
        print(f"[bench] {a.mode} block {i * a.seeds + s + 1}/{len(conds) * a.seeds} (2023-{m:02d}-{d:02d} {h:02d}h seed {s}) {time.time() - t1:.0f}s", flush=True)
np.savez(a.out, blocks=np.asarray(blocks, np.float32), times=np.asarray(times), month=np.asarray(month), day=np.asarray(day),
         hour=np.asarray(hour), seed_idx=np.asarray(seed_idx), levels=np.arange(49, 67))
print(f"saved {a.out}: {len(blocks)} blocks in {(time.time() - t_all) / 60:.0f} min", flush=True)
