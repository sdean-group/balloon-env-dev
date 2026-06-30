"""Single-crop Axis-1 gate (Phase 2c): does one generated crop look like wind?

Before plugging the trained denoiser into the InfiniteDiffusion machinery, a *single*
sampled crop must already pass the reference-free field-quality metrics — if a lone crop
fails, blending many of them won't save it. This isolates the model's job (Axis-1) from
the machinery's job (Axis-2).

We sample N standalone crops (no blending), wrap them as a minimal WindArtifact-shaped
Dataset, and run the existing ``metrics.realism.field_scores``. The COMPOSITE is
spectrum + intermittency; we also surface ``vort/div`` as the rotational-balance
diagnostic. Grid spacing does not affect these scores (slope, ratios and kurtosis are all
scale-free), so the fabricated lat/lon are only there to satisfy the artifact schema.

Reference points (M0 leaderboard): ERA5 ~0.93, phase-shuffle 0.57, toy 0.48, noise 0.00.
The trained model must clear the toy's 0.48 to justify the swap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _load_standalone():
    """Load field_scores + TrainedWindowDenoiser WITHOUT the package (cluster path).

    Run as a script (`python gate.py ...`), the relative imports below fail because there's
    no package context — and `-m src.eval...` would import `src/eval/__init__` which pulls
    the unrelated jax/gym harness stack. So we load just the modules we need (all light:
    torch/numpy/xarray) by file path, registering stub parent packages so their own relative
    imports resolve, and never touching the jax stack. Lets the train->gate loop stay on the
    GPU node with a minimal venv (torch+numpy+xarray+zarr).
    """
    import importlib.util
    import sys
    import types

    here = Path(__file__).resolve()
    idiff = here.parent                 # .../generators/infinite_diffusion
    windeval = here.parents[2]          # .../windeval

    def pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []                 # mark as a package so submodule imports work
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
    mod("w.artifact", windeval / "artifact.py", "w")            # realism needs ..artifact
    mod(f"{idp}.net", idiff / "net.py", idp)
    mod(f"{idp}.data", idiff / "data.py", idp)
    mod(f"{idp}.trained", idiff / "trained.py", idp)            # uses .data/.net (loaded above)
    mod("w.metrics.realism", windeval / "metrics" / "realism.py", "w.metrics")
    return (sys.modules["w.metrics.realism"].field_scores,
            sys.modules[f"{idp}.trained"].TrainedWindowDenoiser)


try:
    from ...metrics.realism import field_scores
    from .trained import TrainedWindowDenoiser
except ImportError:  # pragma: no cover - standalone script path (cluster)
    field_scores, TrainedWindowDenoiser = _load_standalone()

# fabricate ERA5-like geometry (0.25 deg ~ 28 km); see module docstring re scale-invariance.
SF_LAT, SF_LON, DEG = 37.77, 237.58, 0.25
TOY_COMPOSITE = 0.48


def crops_to_dataset(crops, levels) -> xr.Dataset:
    """(N, C, H, W) m/s, C=2L interleaved -> field Dataset (time=N, level=L, y, x)."""
    crops = np.asarray(crops)
    N, C, H, W = crops.shape
    L = C // 2
    f = crops.reshape(N, L, 2, H, W)
    u, v = f[:, :, 0], f[:, :, 1]            # (N, L, H, W)
    lat = SF_LAT + np.arange(H) * DEG
    lon = SF_LON + np.arange(W) * DEG
    return xr.Dataset(
        {"u": (("time", "level", "y", "x"), u.astype("float32")),
         "v": (("time", "level", "y", "x"), v.astype("float32"))},
        coords={"time": np.arange(N), "level": np.asarray(levels),
                "lat": ("y", lat), "lon": ("x", lon)},
    )


def gate(
    ckpt_path: str | Path,
    *,
    n: int = 8,
    size: int = 64,
    num_steps: int = 18,
    seed: int = 0,
    device: str = "cpu",
    use_ema: bool = True,
) -> dict:
    """Sample N crops from the trained denoiser and score them on Axis-1. Returns a dict."""
    phi = TrainedWindowDenoiser(ckpt_path, num_steps=num_steps, device=device, use_ema=use_ema)
    crops = phi.sample_crops(n, size, seed=seed).cpu().numpy()
    finite = bool(np.isfinite(crops).all())
    ds = crops_to_dataset(crops, phi.stats.levels)
    scores = field_scores(ds)

    composite = float(scores["COMPOSITE"])
    result = {
        "step": phi.step,
        "finite": finite,
        "spectrum slope": float(scores["spectrum slope"]),
        "vort/div ratio": float(scores["vort/div ratio"]),
        "increment kurtosis": float(scores["increment kurtosis"]),
        "score: spectrum": float(scores["score: spectrum"]),
        "score: intermittency": float(scores["score: intermittency"]),
        "COMPOSITE": composite,
        "beats_toy": composite >= TOY_COMPOSITE,
    }
    return result


def _print(r: dict) -> None:
    print("\n=== Axis-1 single-crop gate ===")
    print(f"  checkpoint step      : {r['step']}")
    print(f"  finite sample        : {r['finite']}")
    print(f"  spectrum slope (raw) : {r['spectrum slope']:+.2f}   (ideal ~ -3)")
    print(f"  vort/div ratio (raw) : {r['vort/div ratio']:.2f}    (ideal > 1)")
    print(f"  increment kurtosis   : {r['increment kurtosis']:.2f}    (ideal > 3)")
    print(f"  score: spectrum      : {r['score: spectrum']:.2f}")
    print(f"  score: intermittency : {r['score: intermittency']:.2f}")
    print(f"  COMPOSITE            : {r['COMPOSITE']:.2f}   (toy {TOY_COMPOSITE}, era5 ~0.93)")
    print(f"  beats toy (>= {TOY_COMPOSITE})   : {'YES' if r['beats_toy'] else 'no'}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Axis-1 single-crop gate for a trained denoiser.")
    ap.add_argument("ckpt", help="checkpoint path (.pt)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    _print(gate(args.ckpt, n=args.n, size=args.size, num_steps=args.steps,
                device=args.device, seed=args.seed))


if __name__ == "__main__":
    main()
