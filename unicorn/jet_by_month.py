"""Per-month zonal-mean u profiles, model vs held-out ERA5, to localise a jet-strength bias.

    python unicorn/jet_by_month.py --ckpt .../step_300000.pt --churn 8 --steps 18

Same blocks as the gate (blocks-per-month evenly spaced in each held-out week), seeds 0..S-1.
Prints, per month and level: SH max / NH max / tropics of the 10-degree-band zonal-mean u.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/eval/windeval/hpx"))
from eval_coarse import _band_means, _slow_series  # noqa: E402
from sample import HpxSampler  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--heldout", default="/scratch/sps252/era5_hpx_heldout_2023")
ap.add_argument("--layout", default=os.path.expanduser("~/data/hpx_layout"))
ap.add_argument("--blocks-per-month", type=int, default=6); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--steps", type=int, default=18); ap.add_argument("--churn", type=float, default=0.0)
ap.add_argument("--sigma-max", type=float, default=None)
ap.add_argument("--slow-from", nargs="*", default=["/scratch/sps252/era5_hpx", "~/data/era5_hpx_coarse_2020_2021"])
a = ap.parse_args()
import healpy as hp, zarr
s = HpxSampler(a.ckpt, a.layout, num_steps=a.steps, device="cuda", s_churn=a.churn, sigma_max=a.sigma_max)
tau, stride, nside = s.tau, s.stride_hours, s.nside
root = zarr.open_group(a.heldout, mode="r"); ref = root["coarse/uv"]; hours = np.asarray(root["time"][:], dtype=np.int64)
row = {int(h): i for i, h in enumerate(hours)}
lon, lat = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True, lonlat=True)
months = np.array([(np.datetime64("1900-01-01T00", "h") + h.astype("timedelta64[h]")).astype("datetime64[M]") for h in hours])
sh, su = _slow_series(a.slow_from, nside); lookback = int(s.cfg.get("lookback_hours", 720))
print(f"ckpt step {s.step}, {a.steps} steps, churn {a.churn}, sigma_max {s.sigma_max}, {a.seeds} seeds", flush=True)
levels = {"top(53hPa)": 0, "bottom(134hPa)": 34}
t0 = time.time()
for m in np.unique(months):
    hm = hours[months == m]
    cands = [h for h in hm if all((h + k * stride) in row for k in range(tau))]
    starts = [int(cands[i]) for i in np.linspace(0, len(cands) - 1, a.blocks_per_month).round().astype(int)]
    era, gen = [], []
    for h0 in starts:
        hs = h0 + stride * np.arange(tau)
        era.append(np.stack([ref[row[int(h)]] for h in hs]))
        msk = (sh >= h0 - lookback) & (sh < h0); sv = float(su[msk].mean()) if msk.sum() >= 0.5 * lookback else None
        for sd in range(a.seeds):
            gen.append(s.sample_block(hs, seed=1000 * sd + h0 % 997, slow_value=sv))
    era, gen = np.stack(era), np.stack(gen)
    for name, ch in levels.items():
        ze, zg = _band_means(era[:, :, ch], lat), _band_means(gen[:, :, ch], lat)
        print(f"{str(m)} {name:15s} SH max {ze[:9].max():6.1f} vs {zg[:9].max():6.1f} | NH max {ze[9:].max():6.1f} vs {zg[9:].max():6.1f} "
              f"| tropics {ze[8:10].mean():6.1f} vs {zg[8:10].mean():6.1f} | profile rms {np.sqrt(((ze - zg) ** 2).mean()):5.2f}  (ERA5 vs model)", flush=True)
    if str(m).endswith("07"):
        print("   July SH profile (-90..0 by 10): ERA5", np.round(ze[:9], 1).tolist(), " model", np.round(zg[:9], 1).tolist())
print(f"done in {time.time() - t0:.0f}s")
