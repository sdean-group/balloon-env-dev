"""Turn one cBottle-video rollout into the wind file Zermelo reads.

cBottle writes global winds on a HEALPix grid (49,152 pixels, ~100 km) every 6 hours.
Zermelo wants a plain lat/lon box. This script:
  1. reads U/V at the chosen pressure levels from the rollout netCDF,
  2. interpolates each frame onto a regular lat/lon box,
  3. saves one .npz with the keys Zermelo's load_wind() expects,
  4. also saves per-level statistics (for scaling the simplex/GP winds) and a picture to eyeball.

    python cbottle_to_zermelo.py ROLLOUT_DIR OUT.npz
"""
import argparse, glob, json, os
import numpy as np, xarray as xr, healpy as hp

LEVELS_HPA = [850, 700, 500, 300, 200, 50, 10]            # 1000 hPa dropped: underground over land
ALTITUDE_KM = [1.5, 3.0, 5.5, 9.2, 11.8, 20.6, 31.0]       # standard-atmosphere height of each level

ap = argparse.ArgumentParser()
ap.add_argument("rollout_dir"); ap.add_argument("out_npz")
ap.add_argument("--lat", type=float, nargs=2, default=[20.0, 60.0])
ap.add_argument("--lon", type=float, nargs=2, default=[150.0, 230.0], help="degrees east, 0..360")
ap.add_argument("--step", type=float, default=1.0, help="grid spacing in degrees")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.rollout_dir, "*.nc")))
ds = xr.open_mfdataset(files, combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
ds = ds.sortby("time")
_, keep = np.unique(ds.time.values, return_index=True)       # chained clips can repeat a frame
ds = ds.isel(time=keep)
nest = "nest" in (str(ds.attrs) + str(ds["crs"].attrs if "crs" in ds else "")).lower()
npix = ds.sizes["pix"]; nside = hp.npix2nside(npix)
print(f"{len(files)} file(s), {ds.sizes['time']} frames, nside {nside}, {'NEST' if nest else 'RING'} order")

lat = np.arange(a.lat[0], a.lat[1] + 1e-6, a.step); lon = np.arange(a.lon[0], a.lon[1] + 1e-6, a.step)
LON, LAT = np.meshgrid(lon, lat)
theta, phi = np.radians(90.0 - LAT.ravel()), np.radians(LON.ravel() % 360.0)

def to_box(field):                                            # (time, pix) -> (time, n_lat, n_lon)
    return np.stack([hp.get_interp_val(f, theta, phi, nest=nest).reshape(LAT.shape) for f in field]).astype(np.float32)

u = np.stack([to_box(ds[f"U{l}"].values) for l in LEVELS_HPA], axis=1)   # (time, alt, n_lat, n_lon)
v = np.stack([to_box(ds[f"V{l}"].values) for l in LEVELS_HPA], axis=1)
t = ds.time.values; hours = ((t - t[0]) / np.timedelta64(1, "h")).astype(np.float32)

np.savez_compressed(a.out_npz, latitude=lat.astype(np.float32), longitude=lon.astype(np.float32), hours=hours,
                    altitude_km=np.asarray(ALTITUDE_KM, np.float32), pressure_hpa=np.asarray(LEVELS_HPA, np.float32),
                    u=u, v=v, forecast_u=u, forecast_v=v, start_time=str(t[0])[:13])
print(f"wrote {a.out_npz}: u/v {u.shape} (time, altitude, lat, lon), hours {hours[0]:.0f}..{hours[-1]:.0f}")

stats = {str(l): dict(u_mean=float(u[:, i].mean()), u_std=float(u[:, i].std()), v_mean=float(v[:, i].mean()),
                      v_std=float(v[:, i].std()), speed_mean=float(np.hypot(u[:, i], v[:, i]).mean()),
                      speed_max=float(np.hypot(u[:, i], v[:, i]).max())) for i, l in enumerate(LEVELS_HPA)}
with open(a.out_npz.replace(".npz", "_stats.json"), "w") as f: json.dump(stats, f, indent=1)
for l, s in stats.items(): print(f"  {l:>4} hPa: mean speed {s['speed_mean']:5.1f} m/s, max {s['speed_max']:5.1f}, u std {s['u_std']:4.1f}, v std {s['v_std']:4.1f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = [LEVELS_HPA.index(l) for l in (850, 200, 50)]; frames = [0, len(hours) // 2, len(hours) - 1]
fig, axes = plt.subplots(len(show), len(frames), figsize=(15, 8), constrained_layout=True)
for r, i in enumerate(show):
    vmax = float(np.hypot(u[:, i], v[:, i]).max())
    for c, k in enumerate(frames):
        ax = axes[r, c]; im = ax.pcolormesh(lon, lat, np.hypot(u[k, i], v[k, i]), vmin=0, vmax=vmax, cmap="viridis")
        ax.quiver(lon[::4], lat[::4], u[k, i][::4, ::4], v[k, i][::4, ::4], color="w", scale=vmax * 25, width=0.002)
        ax.set_title(f"{LEVELS_HPA[i]} hPa, hour {hours[k]:.0f}"); ax.set_aspect(1.3)
    fig.colorbar(im, ax=axes[r, :], label="wind speed (m/s)", shrink=0.8)
png = a.out_npz.replace(".npz", "_eyetest.png"); fig.savefig(png, dpi=110); print("wrote", png)
