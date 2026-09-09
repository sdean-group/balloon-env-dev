"""Stage 2 gate: fine fields (nside 256) from a Stage 2 checkpoint vs held-out ERA5.

    python src/eval/windeval/hpx/eval_fine.py --ckpt .../step_75000.pt --out .../eval_fine_75000 \
        [--stage1 ~/data/models/stage1_hpx32_ft/step_60000.pt]      # E2: coarse frames from Stage 1 samples

Per block (13 hourly frames starting inside a held-out week) the coarse frames at hours 0/6/12
are the true block means (or Stage 1 samples for the same hours with ``--stage1``); Stage 2
generates the fine field and it is scored against the ERA5 fine field of the same hours:

- **spectrum**: spherical power per channel at frames 0/6/12, mean |log ratio| over l >= 10 and
  by band (10-96 = the coarse grid's range, 96-256, 256-512, 512-767 = pixel scale), with the
  ERA5 self-split floor (half the blocks vs the other half);
- **residual**: RMS of x - lift(block_mean(x)) gen/ERA5 and its 1/3/6 h autocorrelation;
- **distribution**: W1 of u and v on a pixel subset (floor from the ERA5 split); 99.9% speed;
- **opposing winds**: fraction of columns holding two levels whose winds oppose by > 90 deg
  with both >= 5 m/s (the summer's structural row), gen vs ERA5;
- **seams**: RMS jump across face edges vs between interior neighbours, gen vs ERA5;
- **consistency**: max |block_mean(gen) - coarse| at the coarse hours (must be ~0).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_fine import HpxFineSampler  # noqa: E402


def _w1(a: np.ndarray, b: np.ndarray, n: int = 2000) -> float:
    q = np.linspace(0, 1, n)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def _spectra(x: np.ndarray, lmax: int) -> np.ndarray:
    """(C, npix) NEST -> (C, lmax+1) C_l."""
    import healpy as hp
    out = np.zeros((x.shape[0], lmax + 1))
    for c in range(x.shape[0]):
        m = hp.reorder(x[c].astype(np.float64), n2r=True)
        out[c] = hp.anafast(m - m.mean(), lmax=lmax)
    return out


def _opposing(x: np.ndarray, sub: np.ndarray, n_levels: int = 18) -> float:
    """x: (C, npix) with channels [u_1..u_L, v_1..v_L]. Fraction of columns (on the pixel subset) that
    hold a pair of levels whose winds oppose by > 90 deg with both speeds >= 5 m/s."""
    u, v = x[:n_levels][:, sub], x[n_levels:][:, sub]                      # (L, n)
    spd = np.sqrt(u * u + v * v)
    dot = np.einsum("in,jn->ijn", u, u) + np.einsum("in,jn->ijn", v, v)     # (L, L, n)
    ok = (spd[:, None, :] >= 5.0) & (spd[None, :, :] >= 5.0)
    return float(((dot < 0) & ok).any(axis=(0, 1)).mean())


def _residual(x: np.ndarray, ratio2: int = 64) -> np.ndarray:
    bm = x.reshape(*x.shape[:-1], -1, ratio2).mean(-1)
    return x - np.repeat(bm, ratio2, axis=-1)


def _autocorr(r: np.ndarray, k: int) -> float:
    a, b = r[:-k].ravel(), r[k:].ravel()
    return float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))


def _seam(x: np.ndarray, pidx: np.ndarray, pad: int) -> tuple[float, float]:
    """RMS jump between the first interior row of every face and its halo neighbour, and between the
    first two interior rows (same orientation), for one (npix,) map."""
    edge = x[pidx[:, pad, pad:-pad]] - x[pidx[:, pad - 1, pad:-pad]]
    inner = x[pidx[:, pad + 1, pad:-pad]] - x[pidx[:, pad, pad:-pad]]
    return float(np.sqrt((edge ** 2).mean())), float(np.sqrt((inner ** 2).mean()))


def evaluate(ckpt: str, heldout: str, layout: str, *, blocks_per_month: int, out: Path, device: str = "cuda",
             num_steps: int = 18, sigma_min: float | None = None, sigma_max: float | None = None,
             guidance: float = 1.0, stage1: str | None = None, slow_from: list[str] | None = None,
             seed: int = 0, batch_patches: int = 48) -> dict:
    import zarr
    t_start = time.time()
    s = HpxFineSampler(ckpt, layout, num_steps=num_steps, device=device, sigma_min=sigma_min, sigma_max=sigma_max,
                       guidance=guidance, batch_patches=batch_patches)
    tau, stride, L = s.tau, s.coarse_stride, s.C // 2
    root = zarr.open_group(heldout, mode="r")
    fine, coarse, hours = root["fine/uv"], root["coarse/uv"], np.asarray(root["time"][:], dtype=np.int64)
    row = {int(h): i for i, h in enumerate(hours)}
    months = np.array([(np.datetime64("1900-01-01T00", "h") + h.astype("timedelta64[h]")).astype("datetime64[M]") for h in hours])
    starts = []
    for m in np.unique(months):
        hm = hours[months == m]
        cands = [h for h in hm if all((h + k) in row for k in range(tau))]
        pick = np.linspace(0, len(cands) - 1, blocks_per_month + 2).round().astype(int)[1:-1]   # avoid the week's edges
        starts += [int(cands[i]) for i in pick]
    print(f"[eval] ckpt step {s.step}; {len(starts)} blocks x {tau} h; nside {s.nside}; sigma [{s.sigma_min}, {s.sigma_max}]; "
          f"guidance {guidance}; coarse from {'Stage 1 ' + stage1 if stage1 else 'ERA5 block means'}", flush=True)

    s1 = None
    if stage1:
        from eval_coarse import _slow_series
        from sample import HpxSampler
        s1 = HpxSampler(stage1, layout, num_steps=18, device=device)
        slow = _slow_series(slow_from, s1.nside) if slow_from else None
        lookback = int(s1.cfg.get("lookback_hours", 720))

    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(12 * s.nside * s.nside, size=50_000, replace=False))
    lmax = 3 * s.nside - 1
    key_frames = list(range(0, tau, stride))
    spec_e, spec_g, res = [], [], {"blocks": len(starts), "ckpt": str(ckpt), "step": s.step, "num_steps": num_steps,
                                   "sigma_min": s.sigma_min, "sigma_max": s.sigma_max, "guidance": guidance, "stage1": stage1}
    acc = {k: [] for k in ("res_rms_gen", "res_rms_era", "ac_gen", "ac_era", "w1u", "w1v", "p999_gen", "p999_era",
                           "opp_gen", "opp_era", "seam_edge_gen", "seam_inner_gen", "seam_edge_era", "seam_inner_era",
                           "consistency", "u_gen", "u_era", "v_gen", "v_era")}
    for bi, h0 in enumerate(starts):
        hs = h0 + np.arange(tau)
        era = np.stack([fine[row[int(h)]] for h in hs]).astype(np.float32)                       # (τ, C, npix)
        if s1 is None:
            cf = np.stack([coarse[row[int(h0 + stride * k)]] for k in range(len(key_frames))]).astype(np.float32)
        else:
            h1 = h0 + s1.stride_hours * np.arange(s1.tau)
            sv = None
            if slow is not None:
                sh, su = slow; mk = (sh >= h0 - lookback) & (sh < h0)
                if mk.sum() >= 0.5 * lookback:
                    sv = float(su[mk].mean())
            cf = s1.sample_block(h1, seed=1000 + bi, slow_value=sv)[: len(key_frames)]           # frames at 0/6/12 h
        t0 = time.time()
        gen = s.sample_block(cf, hs, seed=seed * 1000 + bi, log=None)
        print(f"[eval] block {bi + 1}/{len(starts)} ({np.datetime64('1900-01-01T00', 'h') + np.timedelta64(int(h0), 'h')}) "
              f"sampled in {time.time() - t0:.0f}s", flush=True)
        bm = gen.reshape(tau, s.C, -1, 64).mean(-1)
        acc["consistency"].append(max(float(np.abs(bm[f] - cf[k]).max()) for k, f in enumerate(key_frames)))
        rg, re = _residual(gen), _residual(era)
        acc["res_rms_gen"].append(float(np.sqrt((rg ** 2).mean()))); acc["res_rms_era"].append(float(np.sqrt((re ** 2).mean())))
        acc["ac_gen"].append([_autocorr(rg, k) for k in (1, 3, 6)]); acc["ac_era"].append([_autocorr(re, k) for k in (1, 3, 6)])
        for f in key_frames:
            spec_g.append(_spectra(gen[f], lmax)); spec_e.append(_spectra(era[f], lmax))
            acc["opp_gen"].append(_opposing(gen[f], sub, L)); acc["opp_era"].append(_opposing(era[f], sub, L))
            for name, x in (("gen", gen[f]), ("era", era[f])):
                e, i = _seam(x[0], s.geom.pidx, s.geom.pad)
                acc[f"seam_edge_{name}"].append(e); acc[f"seam_inner_{name}"].append(i)
        acc["u_gen"].append(gen[:, :L][:, :, sub].ravel()); acc["u_era"].append(era[:, :L][:, :, sub].ravel())
        acc["v_gen"].append(gen[:, L:][:, :, sub].ravel()); acc["v_era"].append(era[:, L:][:, :, sub].ravel())
        spd_g = np.sqrt(gen[:, :L] ** 2 + gen[:, L:] ** 2)[:, :, sub]; spd_e = np.sqrt(era[:, :L] ** 2 + era[:, L:] ** 2)[:, :, sub]
        acc["p999_gen"].append(float(np.percentile(spd_g, 99.9))); acc["p999_era"].append(float(np.percentile(spd_e, 99.9)))

    # ---- aggregate
    cl_g, cl_e = np.mean(spec_g, axis=0), np.mean(spec_e, axis=0)                              # (C, lmax+1)
    half = len(spec_e) // 2
    cl_a, cl_b = np.mean(spec_e[:half], axis=0), np.mean(spec_e[half:], axis=0)
    lr, fl = np.log(cl_g[:, 10:] / cl_e[:, 10:]), np.log(cl_a[:, 10:] / cl_b[:, 10:])
    res["SR_sphere_fine"], res["SR_sphere_fine_floor"] = float(np.abs(lr).mean()), float(np.abs(fl).mean())
    res["spectrum_logratio_by_band"] = {f"l{a}-{b}": float(lr[:, a - 10:b - 10].mean())
                                        for a, b in ((10, 96), (96, 256), (256, 512), (512, lmax + 1))}
    res["spectrum_floor_by_band"] = {f"l{a}-{b}": float(np.abs(fl[:, a - 10:b - 10]).mean())
                                     for a, b in ((10, 96), (96, 256), (256, 512), (512, lmax + 1))}
    res["residual_rms_gen"], res["residual_rms_era"] = float(np.mean(acc["res_rms_gen"])), float(np.mean(acc["res_rms_era"]))
    res["residual_autocorr_1_3_6h_gen"] = np.mean(acc["ac_gen"], axis=0).round(4).tolist()
    res["residual_autocorr_1_3_6h_era"] = np.mean(acc["ac_era"], axis=0).round(4).tolist()
    ug, ue, vg, ve = (np.concatenate(acc[k]) for k in ("u_gen", "u_era", "v_gen", "v_era"))
    res["W1_u"], res["W1_v"] = _w1(ug, ue), _w1(vg, ve)
    hb = len(acc["u_era"]) // 2
    res["W1_u_floor"] = _w1(np.concatenate(acc["u_era"][:hb]), np.concatenate(acc["u_era"][hb:])) if hb else float("nan")
    res["speed_p999_gen"], res["speed_p999_era"] = float(np.mean(acc["p999_gen"])), float(np.mean(acc["p999_era"]))
    res["opposing_wind_frac_gen"], res["opposing_wind_frac_era"] = float(np.mean(acc["opp_gen"])), float(np.mean(acc["opp_era"]))
    for name in ("gen", "era"):
        res[f"seam_edge_rms_{name}"] = float(np.mean(acc[f"seam_edge_{name}"])); res[f"seam_inner_rms_{name}"] = float(np.mean(acc[f"seam_inner_{name}"]))
    res["seam_ratio_gen_over_era"] = (res["seam_edge_rms_gen"] / res["seam_inner_rms_gen"]) / (res["seam_edge_rms_era"] / res["seam_inner_rms_era"])
    res["consistency_max_abs"] = float(np.max(acc["consistency"]))
    res["minutes"] = (time.time() - t_start) / 60
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage 2 gate at nside 256 vs held-out ERA5")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--heldout", default="/scratch/sps252/era5_hpx_heldout_2023")
    ap.add_argument("--layout", default=str(Path.home() / "data/hpx_layout"))
    ap.add_argument("--blocks-per-month", type=int, default=1); ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--sigma-min", type=float, default=None); ap.add_argument("--sigma-max", type=float, default=None)
    ap.add_argument("--guidance", type=float, default=1.0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage1", default=None, help="Stage 1 checkpoint: condition on its samples instead of ERA5 block means (E2)")
    ap.add_argument("--slow-from", nargs="*", default=["/scratch/sps252/era5_hpx", "~/data/era5_hpx_coarse_2020_2021"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--batch-patches", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    evaluate(a.ckpt, a.heldout, a.layout, blocks_per_month=a.blocks_per_month, out=Path(a.out), device=a.device,
             num_steps=a.steps, sigma_min=a.sigma_min, sigma_max=a.sigma_max, guidance=a.guidance, stage1=a.stage1,
             slow_from=a.slow_from, seed=a.seed, batch_patches=a.batch_patches)


if __name__ == "__main__":
    main()
