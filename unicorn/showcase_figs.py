"""Showcase figures for the team (simple, one idea each), from the saved 2023-07-10 fields plus
per-month zonal-mean profiles. Writes PNGs to --out.

    python unicorn/showcase_figs.py --fields /scratch/sps252/runs/eye_jul10/fields.npz --out /scratch/sps252/runs/showcase
"""
import argparse, os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, zarr, healpy as hp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src/eval/windeval/hpx")

ap = argparse.ArgumentParser()
ap.add_argument("--fields", default="/scratch/sps252/runs/eye_jul10/fields.npz"); ap.add_argument("--out", required=True)
ap.add_argument("--stage1", default="/home/sps252/data/models/stage1_hpx32_ft/step_60000.pt")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "semibold", "figure.dpi": 150})
NS, NSC = 256, 32
z = np.load(a.fields); gen_a, gen_b, c1, cera, hs = z["gen_a"].astype(np.float32), z["gen_b"].astype(np.float32), z["c1"], z["cera"], z["hours"]
root = zarr.open_group("/scratch/sps252/era5_hpx_heldout_2023", mode="r"); hours = np.asarray(root["time"][:], dtype=np.int64)
row = {int(h): i for i, h in enumerate(hours)}
era = np.stack([root["fine/uv"][row[int(h)]] for h in hs]).astype(np.float32)
LON, LAT = np.meshgrid(np.arange(-180, 180, 0.5), np.arange(-89.75, 90, 0.5))
def grid(m, lon=LON, lat=LAT): return hp.get_interp_val(hp.reorder(m.astype(np.float64), n2r=True), lon, lat, lonlat=True)
def spd(x, l=0): return np.sqrt(x[l] ** 2 + x[l + 18] ** 2)
def globe(ax, img, title, vmax, cmap="viridis", vmin=0):
    im = ax.imshow(img, origin="lower", extent=(-180, 180, -90, 90), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    for y in (-60, -30, 0, 30, 60): ax.axhline(y, color="w", lw=0.4, alpha=0.5)
    for x in (-120, -60, 0, 60, 120): ax.axvline(x, color="w", lw=0.4, alpha=0.5)
    ax.set_xticks([-120, -60, 0, 60, 120]); ax.set_yticks([-60, -30, 0, 30, 60]); ax.tick_params(labelsize=8, length=2)
    ax.set_title(title); return im
vm = float(np.percentile(spd(era[6]), 99.5))

# ---- Figure 1: coarse -> fine, one hour, one level
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
lift = np.repeat(c1[1], 64, axis=-1)
globe(axs[0], grid(spd(lift)), "1. Stage 1 draws the weather at 1.8 degrees", vm)
globe(axs[1], grid(spd(gen_b[6])), "2. Stage 2 adds the detail at 0.23 degrees", vm)
ZLON, ZLAT = np.meshgrid(np.arange(0, 50, 0.2), np.arange(-70, -30, 0.2))
im = axs[2].imshow(grid(spd(gen_b[6]), ZLON, ZLAT), origin="lower", extent=(0, 50, -70, -30), cmap="viridis", vmin=0, vmax=vm, aspect="auto")
axs[2].set_title("3. Native resolution over the southern ocean"); axs[2].tick_params(labelsize=8)
axs[1].add_patch(plt.Rectangle((0, -70), 50, 40, fill=False, ec="white", lw=1.2))
fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01, label="wind speed at 53 hPa (m/s)")
fig.suptitle("Generated winds for 10 July 2023, 06 UTC: one seed, whole globe, 18 levels", y=1.02, fontsize=13)
fig.savefig(os.path.join(a.out, "fig1_coarse_to_fine.png"), bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: generated vs ERA5, same date (different weather, same climate) + zonal wind
fig, axs = plt.subplots(2, 2, figsize=(13, 7))
globe(axs[0, 0], grid(spd(era[6])), "ERA5, 10 July 2023 06 UTC", vm)
globe(axs[0, 1], grid(spd(gen_b[6])), "Generated for the same date (its own weather)", vm)
um = float(np.percentile(np.abs(era[6, 0]), 99.5))
globe(axs[1, 0], grid(era[6, 0]), "ERA5 eastward wind u", um, cmap="RdBu_r", vmin=-um)
im = globe(axs[1, 1], grid(gen_b[6, 0]), "Generated eastward wind u", um, cmap="RdBu_r", vmin=-um)
fig.colorbar(axs[0, 0].images[0], ax=axs[0], shrink=0.9, pad=0.01, label="speed (m/s)"); fig.colorbar(im, ax=axs[1], shrink=0.9, pad=0.01, label="u (m/s), red = eastward")
fig.suptitle("Same date, same climate, different weather: the winter vortex, the summer easterlies, the tropical QBO winds", y=1.0, fontsize=13)
fig.savefig(os.path.join(a.out, "fig2_generated_vs_era5.png"), bbox_inches="tight"); plt.close(fig)

# ---- Figure 3: ERA5 vs Stage 2 given ERA5's coarse field (pixel-comparable) with the coarse field between
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
im0 = axs[0].imshow(grid(spd(era[6]), ZLON, ZLAT), origin="lower", extent=(0, 50, -70, -30), cmap="viridis", vmin=0, vmax=vm, aspect="auto"); axs[0].set_title("ERA5 at 0.23 degrees")
axs[1].imshow(grid(spd(np.repeat(cera[1], 64, axis=-1)), ZLON, ZLAT), origin="lower", extent=(0, 50, -70, -30), cmap="viridis", vmin=0, vmax=vm, aspect="auto"); axs[1].set_title("its 1.8-degree averages (what Stage 2 is given)")
axs[2].imshow(grid(spd(gen_a[6]), ZLON, ZLAT), origin="lower", extent=(0, 50, -70, -30), cmap="viridis", vmin=0, vmax=vm, aspect="auto"); axs[2].set_title("Stage 2 rebuilds the detail")
for ax in axs: ax.tick_params(labelsize=8)
fig.colorbar(im0, ax=axs, shrink=0.85, pad=0.01, label="wind speed at 53 hPa (m/s)")
fig.suptitle("What Stage 2 adds: the fine detail is generated, not interpolated (southern ocean, 10 July 2023 06 UTC)", y=1.02, fontsize=13)
fig.savefig(os.path.join(a.out, "fig3_stage2_detail.png"), bbox_inches="tight"); plt.close(fig)

# ---- Figure 4: 13 hours of coherent evolution in a window
fig, axs = plt.subplots(1, 5, figsize=(18, 3.4))
for ax, f in zip(axs, (0, 3, 6, 9, 12)):
    im = ax.imshow(grid(spd(gen_b[f]), ZLON, ZLAT), origin="lower", extent=(0, 50, -70, -30), cmap="viridis", vmin=0, vmax=vm, aspect="auto")
    ax.set_title(f"hour {f}" + ("  (coarse frame)" if f % 6 == 0 else "")); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.9, pad=0.01, label="m/s")
fig.suptitle("Hourly evolution: Stage 1 fixes hours 0, 6, 12; Stage 2 generates the hours between, coherently", y=1.04, fontsize=13)
fig.savefig(os.path.join(a.out, "fig4_13_hours.png"), bbox_inches="tight"); plt.close(fig)

# ---- Figure 5: the seasons -- zonal-mean u by latitude, ERA5 vs generated, four months
from sample import HpxSampler
from eval_coarse import _band_means, _slow_series
s1 = HpxSampler(a.stage1, os.path.expanduser("~/data/hpx_layout"), num_steps=18, device="cuda")
lonc, latc = hp.pix2ang(NSC, np.arange(12 * NSC * NSC), nest=True, lonlat=True)
bands = np.arange(-90, 91, 10); mids = (bands[:-1] + bands[1:]) / 2
sh, su = _slow_series(["/scratch/sps252/era5_hpx", "~/data/era5_hpx_coarse_2020_2021"], NSC)
months = np.array([(np.datetime64("1900-01-01T00", "h") + h.astype("timedelta64[h]")).astype("datetime64[M]") for h in hours])
fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for ax, m in zip(axs, np.unique(months)):
    hm = hours[months == m]; cands = [h for h in hm if all((h + 6 * k) in row for k in range(8))]
    starts = [int(cands[i]) for i in np.linspace(0, len(cands) - 1, 6).round().astype(int)]
    e, g = [], []
    for h0 in starts:
        hs8 = h0 + 6 * np.arange(8); e.append(np.stack([root["coarse/uv"][row[int(h)]] for h in hs8]))
        mk = (sh >= h0 - 720) & (sh < h0); sv = float(su[mk].mean()) if mk.sum() >= 360 else None
        for sd in range(2): g.append(s1.sample_block(hs8, seed=100 * sd + h0 % 997, slow_value=sv))
    ze, zg = _band_means(np.stack(e)[:, :, 0], latc), _band_means(np.stack(g)[:, :, 0], latc)
    ax.plot(mids, ze, color="#333", lw=2, label="ERA5"); ax.plot(mids, zg, color="#1f5fa8", lw=2, ls="--", label="generated")
    ax.axhline(0, color="#999", lw=0.6); ax.set_title(str(m)); ax.set_xlabel("latitude"); ax.set_xticks([-60, -30, 0, 30, 60]); ax.grid(alpha=0.3)
axs[0].set_ylabel("zonal-mean eastward wind at 53 hPa (m/s)"); axs[0].legend(frameon=False)
fig.suptitle("The generator knows the seasons: winter vortex, summer easterlies, and the QBO in the tropics (held-out weeks of 2023)", y=1.04, fontsize=13)
fig.savefig(os.path.join(a.out, "fig5_seasons.png"), bbox_inches="tight"); plt.close(fig)
print("done", flush=True)
