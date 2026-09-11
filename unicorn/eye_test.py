"""Eye test: render global and zoomed maps of ERA5 vs Stage 2 (given ERA5's coarse field) vs the
full generator (Stage 1 -> Stage 2), for one held-out 13 h block. Writes PNGs to --out.

    python unicorn/eye_test.py --date 2023-07-10T00 --out /scratch/sps252/runs/eye_jul10
"""
import argparse, os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, zarr, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src/eval/windeval/hpx")
from sample_fine import HpxFineSampler
from sample import HpxSampler
from eval_coarse import _slow_series

ap = argparse.ArgumentParser()
ap.add_argument("--date", default="2023-07-10T00"); ap.add_argument("--out", required=True)
ap.add_argument("--stage2", default="/scratch/sps252/runs/stage2_hpx256/step_200000.pt")
ap.add_argument("--stage1", default="/home/sps252/data/models/stage1_hpx32_ft/step_60000.pt")
ap.add_argument("--seed", type=int, default=3); ap.add_argument("--zoom", default="-70,-30,0,50", help="lat0,lat1,lon0,lon1")
ap.add_argument("--render-only", action="store_true", help="re-render from fields.npz saved by an earlier run (no GPU)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
lay = os.path.expanduser("~/data/hpx_layout")
root = zarr.open_group("/scratch/sps252/era5_hpx_heldout_2023", mode="r"); hours = np.asarray(root["time"][:], dtype=np.int64)
row = {int(h): i for i, h in enumerate(hours)}
h0 = int((np.datetime64(a.date) - np.datetime64("1900-01-01T00")) / np.timedelta64(1, "h")); hs = h0 + np.arange(13)
era = np.stack([root["fine/uv"][row[h]] for h in hs]).astype(np.float32)                    # (13, 36, npix)
cera = np.stack([root["coarse/uv"][row[h0 + 6 * k]] for k in range(3)]).astype(np.float32)   # (3, 36, npix_c)

fields = os.path.join(a.out, "fields.npz")
if a.render_only:
    z = np.load(fields); gen_a, gen_b, c1 = z["gen_a"].astype(np.float32), z["gen_b"].astype(np.float32), z["c1"]
else:
    t0 = time.time()
    s2 = HpxFineSampler(a.stage2, lay, num_steps=18, device="cuda", batch_patches=48)
    gen_a = s2.sample_block(cera, hs, seed=a.seed, log=None); print(f"Stage 2 | ERA5 coarse: {time.time()-t0:.0f}s", flush=True)
    s1 = HpxSampler(a.stage1, lay, num_steps=18, device="cuda")
    sh, su = _slow_series(["/scratch/sps252/era5_hpx", "~/data/era5_hpx_coarse_2020_2021"], s1.nside)
    mk = (sh >= h0 - 720) & (sh < h0); sv = float(su[mk].mean()) if mk.sum() >= 360 else None
    c1 = s1.sample_block(h0 + 6 * np.arange(8), seed=a.seed, slow_value=sv)[:3]                  # (3, 36, npix_c) m/s
    t0 = time.time(); gen_b = s2.sample_block(c1, hs, seed=a.seed + 1, log=None); print(f"Stage 1 -> Stage 2: {time.time()-t0:.0f}s (QBO index {sv})", flush=True)
    np.savez(fields, gen_a=gen_a.astype(np.float16), gen_b=gen_b.astype(np.float16), cera=cera, c1=c1, hours=hs)

# ---- regrid helpers
nside, nside_c = 256, 32
LON, LAT = np.meshgrid(np.arange(-180, 180, 0.5), np.arange(-89.75, 90, 0.5))                  # 360 x 720
lat0, lat1, lon0, lon1 = (float(v) for v in a.zoom.split(","))
ZLON, ZLAT = np.meshgrid(np.arange(lon0, lon1, 0.2), np.arange(lat0, lat1, 0.2))
def grid(m_nest, ns, lon, lat):
    return hp.get_interp_val(hp.reorder(m_nest.astype(np.float64), n2r=True), lon, lat, lonlat=True, nest=False) if False else \
           hp.get_interp_val(hp.reorder(m_nest.astype(np.float64), n2r=True), lon, lat, lonlat=True)
def speed(x, ui, vi): return np.sqrt(x[ui] ** 2 + x[vi] ** 2)
levels = {"top_53hPa": (0, 1), "bottom_134hPa": (34, 35)}      # channels interleave (u_l, v_l)
vmax = {k: float(np.percentile(speed(era[6], *v), 99.5)) for k, v in levels.items()}
def panel(ax, img, title, vm, extent):
    im = ax.imshow(img, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=vm, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([]); return im
ext = (-180, 180, -90, 90); zext = (lon0, lon1, lat0, lat1)
for lname, (ui, vi) in levels.items():
    vm = vmax[lname]
    for f in range(13):
        k = f // 6 if f < 12 else 1
        fig, axs = plt.subplots(2, 2, figsize=(12, 6.4), dpi=90)
        panel(axs[0, 0], grid(speed(era[f], ui, vi), nside, LON, LAT), f"ERA5, {lname.split('_')[1]}, hour {f}", vm, ext)
        panel(axs[0, 1], grid(speed(gen_a[f], ui, vi), nside, LON, LAT), "Stage 2 given ERA5's coarse field", vm, ext)
        panel(axs[1, 0], grid(speed(gen_b[f], ui, vi), nside, LON, LAT), "Stage 1 -> Stage 2, generated weather", vm, ext)
        cc = c1[min(k, 2)] if f % 6 == 0 else (1 - (f % 6) / 6) * c1[k] + ((f % 6) / 6) * c1[k + 1]
        im = panel(axs[1, 1], grid(speed(cc, ui, vi), nside_c, LON, LAT), "Stage 1 coarse field (1.8 deg)", vm, ext)
        fig.colorbar(im, ax=axs, shrink=0.8, label="wind speed (m/s)"); fig.savefig(os.path.join(a.out, f"{lname}_f{f:02d}.png"), bbox_inches="tight"); plt.close(fig)
    # zoom at native resolution, hour 6
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=90)
    lifted = np.repeat(cera[1], 64, axis=-1)
    panel(axs[0], grid(speed(era[6], ui, vi), nside, ZLON, ZLAT), f"ERA5 (0.23 deg), {lname.split('_')[1]}, hour 6", vm, zext)
    panel(axs[1], grid(speed(gen_a[6], ui, vi), nside, ZLON, ZLAT), "Stage 2 given ERA5's coarse field", vm, zext)
    im = panel(axs[2], grid(speed(lifted, ui, vi), nside, ZLON, ZLAT), "coarse field alone (1.8 deg)", vm, zext)
    for ax in axs: ax.set_xticks(np.arange(lon0, lon1 + 1, 10)); ax.set_yticks(np.arange(lat0, lat1 + 1, 10)); ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axs, shrink=0.9, label="wind speed (m/s)"); fig.savefig(os.path.join(a.out, f"{lname}_zoom.png"), bbox_inches="tight"); plt.close(fig)
    # u component global at hour 6 (sign matters for balloons)
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), dpi=90); um = float(np.percentile(np.abs(era[6, ui]), 99.5))
    for ax, x, t in ((axs[0], era[6], "ERA5"), (axs[1], gen_a[6], "Stage 2 | ERA5 coarse"), (axs[2], gen_b[6], "Stage 1 -> Stage 2")):
        im = ax.imshow(grid(x[ui], nside, LON, LAT), origin="lower", extent=ext, cmap="RdBu_r", vmin=-um, vmax=um, aspect="auto"); ax.set_title(f"{t}: u, {lname.split('_')[1]}, hour 6", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.9, label="u (m/s), red = eastward"); fig.savefig(os.path.join(a.out, f"{lname}_u.png"), bbox_inches="tight"); plt.close(fig)
print("done", flush=True)
