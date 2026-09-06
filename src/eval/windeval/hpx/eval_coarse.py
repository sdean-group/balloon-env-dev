"""Stage 1 gate: score a coarse-generator checkpoint against held-out ERA5 at nside 32.

Pre-registered (design register §4, 2026-09-05) rows, all on the sphere, all against the
held-out store (days 8-14 of Jan/Apr/Jul/Oct 2023, never trained on):

- **spectrum**: spherical power spectrum of every channel (healpy ``anafast`` on RING maps),
  reported as mean |log(model/ERA5)| over l = 10..3*nside-1, the box-PSD ``SR_E`` analogue.
  The ERA5 self-split floor (first half of the held-out days vs second half) is printed.
- **marginals**: 1-D Wasserstein-1 of u and v pooled over all pixels and channels (m/s),
  and the 99.9% wind-speed tail; both with the self-split floor.
- **zonal structure**: per-10°-band zonal-mean u at the top (~53 hPa) and bottom (~134 hPa)
  levels: correlation and RMS difference of the profiles; jet strength (max zonal-mean u in
  each hemisphere).
- **dispersion / coverage**: seed-to-seed RMS between two samples at the same timestamps,
  divided by the ERA5 RMS between two different held-out days at the same hour. The
  regional model reached 58% (plain) and 6.9% (real-coarse downscaler); Stage 1 should
  approach 100%, since the seed *is* the weather.
- **temporal**: adjacent-frame (6 h) correlation of anomalies inside a block vs ERA5.

Samples are generated in blocks of ``n_frames`` at ``stride_hours`` starting at the hours
of the held-out store, so every generated frame has an ERA5 counterpart at the same hour.

Usage (GPU node with the stores on its scratch)::

    python src/eval/windeval/hpx/eval_coarse.py --ckpt /scratch/sps252/runs/stage1_hpx32/step_100000.pt \
        --heldout /scratch/sps252/era5_hpx_heldout_2023 --layout ~/data/hpx_layout \
        --blocks-per-month 6 --seeds 2 --out /scratch/sps252/runs/stage1_hpx32/eval_100000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample import HpxSampler  # noqa: E402


def _w1(a: np.ndarray, b: np.ndarray, n: int = 4000) -> float:
    q = np.linspace(0, 1, n)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def _spectra(x_nest: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    """(N, C, npix) NEST -> mean over N of C_l per channel, (C, lmax+1)."""
    import healpy as hp
    out = np.zeros((x_nest.shape[1], lmax + 1))
    for f in x_nest:
        for c in range(f.shape[0]):
            m = hp.reorder(f[c].astype(np.float64), n2r=True)
            out[c] += hp.anafast(m - m.mean(), lmax=lmax)
    return out / x_nest.shape[0]


def _band_means(u: np.ndarray, lat: np.ndarray, bands=np.arange(-90, 91, 10)) -> np.ndarray:
    return np.array([u[..., (lat >= a) & (lat < b)].mean() for a, b in zip(bands[:-1], bands[1:])])


def evaluate(ckpt: str, heldout: str, layout: str, *, blocks_per_month: int, seeds: int,
             out: Path, device: str = "cuda", num_steps: int = 18) -> dict:
    import healpy as hp
    import zarr
    sampler = HpxSampler(ckpt, layout, num_steps=num_steps, device=device)
    tau, stride, nside, C = sampler.tau, sampler.stride_hours, sampler.nside, sampler.C
    root = zarr.open_group(heldout, mode="r")
    ref, hours = root["coarse/uv"], np.asarray(root["time"][:], dtype=np.int64)
    hour_to_row = {int(h): i for i, h in enumerate(hours)}
    lon, lat = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True, lonlat=True)
    trop = np.abs(lat) < 10.0

    # block starts: evenly spaced inside each month's held-out week, all frames present
    span = (tau - 1) * stride
    starts = []
    months = np.array([(np.datetime64("1900-01-01T00", "h") + h.astype("timedelta64[h]")).astype("datetime64[M]") for h in hours])
    for m in np.unique(months):
        hm = hours[months == m]
        cands = [h for h in hm if all((h + k * stride) in hour_to_row for k in range(tau))]
        pick = np.linspace(0, len(cands) - 1, blocks_per_month).round().astype(int)
        starts += [int(cands[i]) for i in pick]
    print(f"[eval] ckpt step {sampler.step}; {len(starts)} blocks x {seeds} seeds x {tau} frames; nside {nside}", flush=True)

    # slow-state index from the held-out store's own past is not available (it holds only the
    # 7 held-out days); use the model's training-mean (present=0 -> "absent") for all samples.
    gen = {s: [] for s in range(seeds)}
    era = []
    t0 = time.time()
    for h0 in starts:
        hs = h0 + stride * np.arange(tau)
        era.append(np.stack([ref[hour_to_row[int(h)]] for h in hs]))            # (τ, C, npix)
        for s in range(seeds):
            gen[s].append(sampler.sample_block(hs, seed=1000 * s + h0 % 997))
    print(f"[eval] sampled in {time.time() - t0:.0f}s", flush=True)
    era = np.stack(era)                                                          # (B, τ, C, npix)
    if np.isnan(era).any():
        raise RuntimeError(f"held-out store {heldout} has NaN rows for the requested hours: "
                           "finish (or repair) the reference ingest before scoring")
    gen = {s: np.stack(v) for s, v in gen.items()}
    g0 = gen[0]
    res = {"ckpt": str(ckpt), "step": sampler.step, "blocks": len(starts), "seeds": seeds}

    # --- spectra (channel-mean of |log ratio| over l >= 10), with the ERA5 self-split floor
    lmax = 3 * nside - 1
    flat = lambda x: x.reshape(-1, C, x.shape[-1])
    cl_e, cl_g = _spectra(flat(era), nside, lmax), _spectra(flat(g0), nside, lmax)
    half = len(era) // 2
    cl_a, cl_b = _spectra(flat(era[:half]), nside, lmax), _spectra(flat(era[half:]), nside, lmax)
    lr = np.log(cl_g[:, 10:] / cl_e[:, 10:]); fl = np.log(cl_a[:, 10:] / cl_b[:, 10:])
    res["SR_sphere"] = float(np.abs(lr).mean()); res["SR_sphere_floor"] = float(np.abs(fl).mean())
    res["spectrum_logratio_by_band"] = {f"l{a}-{b}": float(lr[:, a - 10:b - 10].mean())
                                        for a, b in ((10, 20), (20, 40), (40, 70), (70, lmax + 1))}

    # --- marginals and tails
    ue, ve = era[:, :, 0::2].ravel(), era[:, :, 1::2].ravel()
    ug, vg = g0[:, :, 0::2].ravel(), g0[:, :, 1::2].ravel()
    ua, ub = era[:half, :, 0::2].ravel(), era[half:, :, 0::2].ravel()
    res["W1_u"], res["W1_v"] = _w1(ue, ug), _w1(ve, vg)
    res["W1_u_floor"] = _w1(ua, ub)
    spd_e, spd_g = np.hypot(era[:, :, 0::2], era[:, :, 1::2]), np.hypot(g0[:, :, 0::2], g0[:, :, 1::2])
    res["speed_p999_era5"], res["speed_p999_model"] = float(np.quantile(spd_e, 0.999)), float(np.quantile(spd_g, 0.999))

    # --- zonal structure at the top (ch 0) and bottom (ch 2*17) u levels
    for name, ch in (("top", 0), ("bottom", 2 * (C // 2 - 1))):
        ze, zg = _band_means(era[:, :, ch], lat), _band_means(g0[:, :, ch], lat)
        res[f"zonal_{name}_corr"] = float(np.corrcoef(ze, zg)[0, 1])
        res[f"zonal_{name}_rms"] = float(np.sqrt(((ze - zg) ** 2).mean()))
        res[f"jet_{name}_NH_era5"], res[f"jet_{name}_NH_model"] = float(ze[9:].max()), float(zg[9:].max())
        res[f"jet_{name}_SH_era5"], res[f"jet_{name}_SH_model"] = float(ze[:9].max()), float(zg[:9].max())
        res[f"tropics_{name}_era5"] = float(era[:, :, ch][..., trop].mean()); res[f"tropics_{name}_model"] = float(g0[:, :, ch][..., trop].mean())

    # --- dispersion: seed spread vs ERA5 day-to-day spread at the same hour of day
    if seeds >= 2:
        seed_rms = float(np.sqrt(((gen[0] - gen[1]) ** 2).mean()))
        # pair each block with another block of the same month at the same hour-of-day
        hod = np.array([(h0 % 24) for h0 in starts]); mon = np.array([str(m)[:7] for m in months[[hour_to_row[h] for h in starts]]])
        pairs = [(i, j) for i in range(len(starts)) for j in range(i + 1, len(starts)) if hod[i] == hod[j] and mon[i] == mon[j]]
        day_rms = float(np.sqrt(np.mean([((era[i] - era[j]) ** 2).mean() for i, j in pairs]))) if pairs else float("nan")
        res["seed_spread_rms"], res["era5_day_spread_rms"] = seed_rms, day_rms
        res["coverage"] = seed_rms / day_rms if day_rms == day_rms else float("nan")

    # --- temporal: adjacent-frame anomaly correlation (6 h)
    def adj_corr(x):
        a = x - x.mean(axis=-1, keepdims=True)
        num = (a[:, :-1] * a[:, 1:]).sum(-1); den = np.sqrt((a[:, :-1] ** 2).sum(-1) * (a[:, 1:] ** 2).sum(-1))
        return float((num / den).mean())
    res["adjacent_frame_corr_era5"], res["adjacent_frame_corr_model"] = adj_corr(era), adj_corr(g0)

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "samples.npz", starts=np.asarray(starts), gen0=g0[:, :, :, ::64].astype(np.float32),
                        era=era[:, :, :, ::64].astype(np.float32))   # thinned for figures
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "spectrum_logratio_by_band"}, indent=1))
    print("spectrum log-ratio by band:", {k: round(v, 3) for k, v in res["spectrum_logratio_by_band"].items()})
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage 1 gate at nside 32 vs held-out ERA5")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--heldout", required=True)
    ap.add_argument("--layout", default=str(Path.home() / "data/hpx_layout"))
    ap.add_argument("--blocks-per-month", type=int, default=6); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=18); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    evaluate(a.ckpt, a.heldout, a.layout, blocks_per_month=a.blocks_per_month, seeds=a.seeds,
             out=Path(a.out), device=a.device, num_steps=a.steps)


if __name__ == "__main__":
    main()
