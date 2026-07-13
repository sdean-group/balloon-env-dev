"""Single-crop training gate: do sampled crops match the training data's statistics?

Benchmark-v2 version (see docs/benchmark-v2-changes.md): the old reference-free COMPOSITE
is gone. The gate now samples N standalone crops (no blending) from a checkpoint, samples
matching random crops from the *training* zarr, and reports the reference-based metrics
between them — spectral residuals (SR_E / SR_div / SR_vort), level-averaged marginal W1,
and tail error — next to a self-split floor (training crops vs other training crops, the
sampling-noise level of each metric at this crop size).

There is deliberately NO hard pass/fail threshold: read each value against its floor
(`ratio` column ≈ 1 is ideal; ≲2–3 is healthy mid-training). `finite` is the only boolean.

Runs standalone on the cluster (no jax, no package import): torch+numpy+xarray only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _load_standalone():
    """Load the needed modules WITHOUT the package (cluster path).

    Run as a script (`python gate.py ...`), the relative imports below fail because there's
    no package context — and `-m src.eval...` would import `src/eval/__init__` which pulls
    the unrelated jax/gym harness stack. So we load just the modules we need by file path,
    registering stub parent packages so their own relative imports resolve.
    """
    import importlib.util
    import sys
    import types

    here = Path(__file__).resolve()
    idiff = here.parent                 # .../generators/infinite_diffusion
    windeval = here.parents[2]          # .../windeval

    def pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m

    def mod(name, path, package):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        m.__package__ = package
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    for p in ("w", "w.generators", "w.generators.infinite_diffusion", "w.metrics"):
        pkg(p)
    idp = "w.generators.infinite_diffusion"
    mod("w.artifact", windeval / "artifact.py", "w")        # spectra needs ..artifact
    mod(f"{idp}.net", idiff / "net.py", idp)
    da = mod(f"{idp}.data", idiff / "data.py", idp)
    mod(f"{idp}.trained", idiff / "trained.py", idp)
    st = mod(f"{idp}.spacetime", idiff / "spacetime.py", idp)
    sp = mod("w.metrics.spectra", windeval / "metrics" / "spectra.py", "w.metrics")
    di = mod("w.metrics.distributions", windeval / "metrics" / "distributions.py", "w.metrics")
    return (sp, di, sys.modules[f"{idp}.trained"].TrainedWindowDenoiser,
            st.SpaceTimeSampler, da._time_blocks)


try:
    from ...metrics import spectra as _spectra
    from ...metrics import distributions as _dist
    from .data import _time_blocks
    from .spacetime import SpaceTimeSampler
    from .trained import TrainedWindowDenoiser
except ImportError:  # pragma: no cover - standalone script path (cluster)
    _spectra, _dist, TrainedWindowDenoiser, SpaceTimeSampler, _time_blocks = _load_standalone()

# fabricated ERA5-like geometry (0.25 deg); identical on both sides, so comparisons are fair
SF_LAT, SF_LON, DEG = 37.77, 237.58, 0.25
DEFAULT_REF = "src/eval/windeval/data/era5_train.zarr"


def crops_to_dataset(crops, levels) -> xr.Dataset:
    """(N, C, H, W) m/s, C=2L interleaved -> field Dataset (time=N, level=L, y, x)."""
    crops = np.asarray(crops)
    N, C, H, W = crops.shape
    L = C // 2
    f = crops.reshape(N, L, 2, H, W)
    u, v = f[:, :, 0], f[:, :, 1]
    lat = SF_LAT + np.arange(H) * DEG
    lon = SF_LON + np.arange(W) * DEG
    return xr.Dataset(
        {"u": (("time", "level", "y", "x"), u.astype("float32")),
         "v": (("time", "level", "y", "x"), v.astype("float32"))},
        coords={"time": np.arange(N), "level": np.asarray(levels),
                "lat": ("y", lat), "lon": ("x", lon)},
    )


def _ref_crops(ref_path, n, size, levels, seed=0) -> xr.Dataset:
    """N random (size×size) crops from the training zarr as a field Dataset."""
    z = xr.open_zarr(ref_path, consolidated=False, zarr_format=2)
    u, v = z["u"].values, z["v"].values            # (t, L, y, x)
    rng = np.random.default_rng(seed)
    nt, _, ny, nx = u.shape
    us, vs = [], []
    for _ in range(n):
        t = rng.integers(nt)
        y0 = rng.integers(ny - size + 1)
        x0 = rng.integers(nx - size + 1)
        us.append(u[t, :, y0:y0 + size, x0:x0 + size])
        vs.append(v[t, :, y0:y0 + size, x0:x0 + size])
    lat = SF_LAT + np.arange(size) * DEG
    lon = SF_LON + np.arange(size) * DEG
    return xr.Dataset(
        {"u": (("time", "level", "y", "x"), np.stack(us).astype("float32")),
         "v": (("time", "level", "y", "x"), np.stack(vs).astype("float32"))},
        coords={"time": np.arange(n), "level": np.asarray(levels),
                "lat": ("y", lat), "lon": ("x", lon)},
    )


def _ref_blocks(ref_path, n, size, tau, seed=0):
    """N random τ-frame blocks from the training zarr: matched conditions + ref frames.

    Returns ``(ds, conds)``: the n·τ reference frames as a field Dataset, and the per-block
    conditioning tuples ``(lat (H,), lon (W,), times (τ,))`` at which the model should be
    sampled — so the gate compares model vs reference **at the same locations and times**.
    Block starts never straddle a time discontinuity (same rule as training).
    """
    z = xr.open_zarr(ref_path, consolidated=False, zarr_format=2)
    u, v = z["u"].values, z["v"].values                    # (t, L, y, x)
    lat_g = np.asarray(z["lat"].values, dtype=np.float64)
    lon_g = np.asarray(z["lon"].values, dtype=np.float64)
    times_g = np.asarray(z["time"].values)
    blocks = _time_blocks(times_g)
    starts = np.asarray([t for (a, b) in blocks for t in range(a, b - tau + 1)])
    rng = np.random.default_rng(seed)
    _, _, ny, nx = u.shape
    us, vs, conds = [], [], []
    for _ in range(n):
        t0 = int(starts[rng.integers(len(starts))])
        y0 = int(rng.integers(ny - size + 1))
        x0 = int(rng.integers(nx - size + 1))
        us.append(u[t0:t0 + tau, :, y0:y0 + size, x0:x0 + size])
        vs.append(v[t0:t0 + tau, :, y0:y0 + size, x0:x0 + size])
        conds.append((lat_g[y0:y0 + size], lon_g[x0:x0 + size], times_g[t0:t0 + tau]))
    u_all = np.concatenate(us, axis=0)                     # (n*τ, L, H, W)
    v_all = np.concatenate(vs, axis=0)
    lat = SF_LAT + np.arange(size) * DEG
    lon = SF_LON + np.arange(size) * DEG
    ds = xr.Dataset(
        {"u": (("time", "level", "y", "x"), u_all.astype("float32")),
         "v": (("time", "level", "y", "x"), v_all.astype("float32"))},
        coords={"time": np.arange(u_all.shape[0]), "level": np.asarray(z["level"].values),
                "lat": ("y", lat), "lon": ("x", lon)},
    )
    return ds, conds


def _metrics(pred_ds, ref_ds) -> dict:
    ps, rs = _spectra.dataset_spectra(pred_ds), _spectra.dataset_spectra(ref_ds)
    out = dict(_spectra.spectral_residual(ps, rs))
    m, _ = _dist.marginal_w1(pred_ds, ref_ds)
    out.update(m)
    out.update(_dist.extreme_quantile_error(pred_ds, ref_ds))
    return out


def gate(
    ckpt_path: str | Path,
    *,
    ref: str | Path = DEFAULT_REF,
    n: int = 8,
    size: int = 64,
    num_steps: int = 18,
    seed: int = 0,
    device: str = "cpu",
    use_ema: bool = True,
) -> dict:
    """Sample N crops (or τ-frame blocks), compare to training statistics + a floor.

    Static checkpoints: N independent crops vs N random training crops (as before).
    Spacetime checkpoints (``cfg["spacetime"]``): N τ-frame blocks; conditional ones are
    sampled at the SAME (lat, lon, times) as the matched reference blocks, so the gate is
    condition-matched. Metrics pool the n·τ frames on each side.
    """
    import torch

    cfg = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)["cfg"]
    if cfg.get("spacetime"):
        sampler = SpaceTimeSampler(ckpt_path, num_steps=num_steps, device=device,
                                   use_ema=use_ema)
        tau = sampler.tau
        ref_a, conds = _ref_blocks(ref, n, size, tau, seed=seed)
        ref_b, _ = _ref_blocks(ref, n, size, tau, seed=seed + 1)
        frames = []
        for i, (la, lo, ts) in enumerate(conds):
            kw = dict(lat=la, lon=lo, times=ts) if sampler.conditional else {}
            us, vs = sampler.sample_block((size, size), seed=seed + i, **kw)
            frames.append(np.stack([us, vs], axis=2).reshape(tau, -1, size, size))
        crops = np.concatenate(frames, axis=0)               # (n*τ, 2L, H, W)
        step, levels = sampler.step, sampler.stats.levels
    else:
        phi = TrainedWindowDenoiser(ckpt_path, num_steps=num_steps, device=device,
                                    use_ema=use_ema)
        crops = phi.sample_crops(n, size, seed=seed).cpu().numpy()
        step, levels = phi.step, phi.stats.levels
        ref_a = _ref_crops(ref, n, size, levels, seed=seed)
        ref_b = _ref_crops(ref, n, size, levels, seed=seed + 1)

    finite = bool(np.isfinite(crops).all())
    model_ds = crops_to_dataset(crops, levels)
    model_vs_ref = _metrics(model_ds, ref_a)
    floor = _metrics(ref_b, ref_a)
    return {"step": step, "finite": finite,
            "model_vs_ref": model_vs_ref, "floor": floor}


def _print(r: dict) -> None:
    print("\n=== Training gate (benchmark v2: vs training-crop statistics) ===")
    print(f"  checkpoint step : {r['step']}")
    print(f"  finite sample   : {r['finite']}")
    print(f"  {'metric':26s} {'model':>9s} {'floor':>9s} {'ratio':>7s}   (ratio ≈1 ideal, ≲2–3 healthy)")
    for k, v in r["model_vs_ref"].items():
        fl = r["floor"].get(k, float("nan"))
        ratio = v / fl if fl and np.isfinite(fl) and fl > 0 else float("nan")
        print(f"  {k:26s} {v:9.3f} {fl:9.3f} {ratio:7.2f}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Training gate: sampled crops vs training statistics.")
    ap.add_argument("ckpt", help="checkpoint path (.pt)")
    ap.add_argument("--ref", default=DEFAULT_REF, help="training zarr for reference crops")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    _print(gate(args.ckpt, ref=args.ref, n=args.n, size=args.size, num_steps=args.steps,
                device=args.device, seed=args.seed))


if __name__ == "__main__":
    main()
