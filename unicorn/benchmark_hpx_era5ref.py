"""ERA5 itself sent through our grid (0.25 deg -> HEALPix nside 256 -> the benchmark's 0.25 deg window),
packaged at the benchmark conditions in the summer cache format. Scored like a model, it is the
attainable floor for the spectral rows (SR_E, L_eff) of anything that lives on our grid. CPU only.
Conditions whose 13-hour window leaves the held-out week (day 14, 12 UTC) are skipped.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, zarr, healpy as hp
out = sys.argv[1]; os.makedirs(os.path.dirname(out), exist_ok=True)
LAT = 48.0 - 0.25 * np.arange(64); LON = 232.0 + 0.25 * np.arange(64); GLON, GLAT = np.meshgrid(LON, LAT)
root = zarr.open_group("/scratch/sps252/era5_hpx_heldout_2023", mode="r"); hours = np.asarray(root["time"][:], dtype=np.int64)
row = {int(h): i for i, h in enumerate(hours)}
def regrid(x):
    tau, C = x.shape[:2]; o = np.zeros((tau, C // 2, 2, 64, 64), np.float32)
    for f in range(tau):
        for c in range(C):
            o[f, c // 2, c % 2] = hp.get_interp_val(hp.reorder(x[f, c].astype(np.float64), n2r=True), GLON, GLAT, lonlat=True)
    return o
blocks, times, month, day, hour, seed_idx = [], [], [], [], [], []
t_all = time.time()
for m in (1, 4, 7, 10):
    for h in (0, 12):
        for d in range(8, 15):
            t0 = np.datetime64(f"2023-{m:02d}-{d:02d}T{h:02d}", "h"); h0 = int((t0 - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
            hs = h0 + np.arange(13)
            if not all(int(x) in row for x in hs):
                print(f"skip 2023-{m:02d}-{d:02d} {h:02d}h (window leaves the held-out week)", flush=True); continue
            era = np.stack([root["fine/uv"][row[int(x)]] for x in hs]).astype(np.float32)
            blocks.append(regrid(era)); times.append(t0 + np.arange(13).astype("timedelta64[h]")); month.append(m); day.append(d); hour.append(h); seed_idx.append(0)
            print(f"2023-{m:02d}-{d:02d} {h:02d}h done ({time.time() - t_all:.0f}s)", flush=True)
np.savez(out, blocks=np.asarray(blocks, np.float32), times=np.asarray(times), month=np.asarray(month), day=np.asarray(day), hour=np.asarray(hour),
         seed_idx=np.asarray(seed_idx), levels=np.arange(49, 67))
print(f"saved {out}: {len(blocks)} blocks", flush=True)
