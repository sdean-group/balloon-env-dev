"""Sanity comparison of pretrained cBottle-3d (HPX64) vs ERA5 at ~50 hPa, the only level in our band.

Distributional/structural checks only (a diffusion sample is not the same weather as ERA5 that day):
zonal-mean u profile, per-latitude eddy std, global marginal W1 of u/v, spherical power spectrum.
ERA5 model level 49 (~51 hPa) from ARCO-ERA5 is regridded to HPX64 the same way for both fields.
"""
import sys, numpy as np, xarray as xr, healpy as hp
from scipy.interpolate import RegularGridInterpolator
out = sys.argv[1]
ds = xr.open_dataset(out)
print("cBottle output attrs:", {k: str(v)[:60] for k, v in ds.attrs.items()})
print("crs attrs:", {k: str(v)[:60] for k, v in ds.crs.attrs.items()})
nest = "nest" in (str(ds.crs.attrs) + str(ds.attrs)).lower()
nside = 64; npix = 12 * nside * nside
u_m = ds["U50"].values; v_m = ds["V50"].values          # (time, pix)
if nest:                                                  # -> RING for healpy spectra
    u_m = np.stack([hp.reorder(x, n2r=True) for x in u_m]); v_m = np.stack([hp.reorder(x, n2r=True) for x in v_m])
theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
lat_p = 90.0 - np.degrees(theta); lon_p = np.degrees(phi)

era = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1", consolidated=True, chunks=None, storage_options={"token": "anon"})
times = ds.time.values
sel = era.sel(time=times, hybrid=49)[["u_component_of_wind", "v_component_of_wind"]]
ue = sel["u_component_of_wind"].values; ve = sel["v_component_of_wind"].values   # (time, 721, 1440)
lat = era.latitude.values; lon = era.longitude.values
order = np.argsort(lat); lat_a = lat[order]
def to_hpx(f):
    f = f[order]; f = np.concatenate([f, f[:, :1]], axis=1); lo = np.concatenate([lon, [360.0]])
    itp = RegularGridInterpolator((lat_a, lo), f, bounds_error=False, fill_value=None)
    return itp(np.stack([lat_p, lon_p % 360.0], axis=1)).astype(np.float32)
u_e = np.stack([to_hpx(x) for x in ue]); v_e = np.stack([to_hpx(x) for x in ve])
print(f"\nERA5 ML49 (~51 hPa) vs cBottle U50/V50, {len(times)} timestamps: {[str(t)[:13] for t in times]}")
print(f"  global mean u   ERA5 {u_e.mean():6.2f}  cBottle {u_m.mean():6.2f}   | std u  ERA5 {u_e.std():5.2f}  cBottle {u_m.std():5.2f}")
print(f"  global mean |V| ERA5 {np.hypot(u_e,v_e).mean():6.2f}  cBottle {np.hypot(u_m,v_m).mean():6.2f}")
bands = np.arange(-90, 91, 10)
print("\n  zonal-mean u and eddy std by 10-deg band (avg over timestamps):")
print("   band        ERA5 mean  cB mean | ERA5 std  cB std")
zm_e, zm_m = [], []
for a, b in zip(bands[:-1], bands[1:]):
    m = (lat_p >= a) & (lat_p < b)
    me, mm = u_e[:, m].mean(), u_m[:, m].mean(); se = (u_e[:, m] - u_e[:, m].mean(1, keepdims=True)).std(); sm = (u_m[:, m] - u_m[:, m].mean(1, keepdims=True)).std()
    zm_e.append(me); zm_m.append(mm)
    print(f"   {a:4d}..{b:4d}  {me:8.2f} {mm:8.2f}  | {se:7.2f} {sm:7.2f}")
zm_e, zm_m = np.array(zm_e), np.array(zm_m)
# regrid check: the same band means straight from the ERA5 lat/lon grid (cos-lat weighted)
w = np.cos(np.radians(lat))[None, :, None]
print("  regrid check, ERA5 band means on the native lat/lon grid (should match column 1):",
      " ".join(f"{(ue[:, (lat >= a) & (lat < b)] * w[:, (lat >= a) & (lat < b)]).sum() / (w[:, (lat >= a) & (lat < b)].sum() * ue.shape[0] * ue.shape[2]):6.2f}" for a, b in zip(bands[:-1], bands[1:])))
print(f"  zonal-mean profile: corr {np.corrcoef(zm_e, zm_m)[0,1]:.3f}, RMS diff {np.sqrt(((zm_e-zm_m)**2).mean()):.2f} m/s")
def w1(a, b, n=2000):
    q = np.linspace(0, 1, n); return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())
print(f"\n  marginal W1 (all pixels, both times): u {w1(u_e.ravel(), u_m.ravel()):.2f} m/s, v {w1(v_e.ravel(), v_m.ravel()):.2f} m/s"
      f"  | ERA5 T00 vs T12 self-floor: u {w1(u_e[0], u_e[1]):.2f}, v {w1(v_e[0], v_e[1]):.2f}")
print(f"  tails (99.9% |V|): ERA5 {np.quantile(np.hypot(u_e,v_e), 0.999):.1f}  cBottle {np.quantile(np.hypot(u_m,v_m), 0.999):.1f} m/s")
lmax = 3 * nside - 1
cl_e = np.mean([hp.anafast(x - x.mean(), lmax=lmax) for x in u_e], axis=0); cl_m = np.mean([hp.anafast(x - x.mean(), lmax=lmax) for x in u_m], axis=0)
ell = np.arange(lmax + 1); r = np.log(cl_m[2:] / cl_e[2:])
print("\n  spherical power spectrum of u, log(cBottle/ERA5) by multipole band (0 = match):")
for a, b in ((2, 10), (10, 30), (30, 60), (60, 100), (100, 150), (150, lmax + 1)):
    print(f"   l {a:3d}..{b:3d}: {r[a-2:b-2].mean():+.2f}")
print(f"  mean |log ratio| over l=10..{lmax}: {np.abs(r[8:]).mean():.2f}  (our SR_E metric is this quantity on a box PSD; ERA5 self-split floor there is 0.25)")
