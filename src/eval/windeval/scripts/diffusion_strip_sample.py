"""Capture the reverse-diffusion trajectory for the poster's explainer strip.

Re-runs SpaceTimeSampler._heun_block step for step (same EDM sigma schedule, same
deterministic s_churn=0 settings the benchmark board uses) but keeps every
intermediate state x_i instead of only the final block. The saved states are the
model's real trajectory, not a re-noised final sample.

Output: diffusion_strip_states.npz with speed fields in m/s at one level/frame,
plus the sigma each panel sits at.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ..generators.infinite_diffusion.spacetime import SpaceTimeSampler, edm_sigma_schedule

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


@torch.no_grad()
def trajectory(sampler: SpaceTimeSampler, z, *, cond, tfeat):
    """_heun_block with snapshots. Mirrors the deterministic (s_churn=0) branch."""
    sig = edm_sigma_schedule(sampler.num_steps, sampler.sigma_min, sampler.sigma_max,
                             device=z.device, dtype=z.dtype)
    x = z * sig[0]
    B = x.shape[0]
    states = [x.clone()]                      # x at sigma[0] == pure scaled noise
    for i in range(sampler.num_steps):
        s_cur, s_next = sig[i], sig[i + 1]
        s_hat = s_cur                          # s_churn == 0 -> gamma == 0
        d = (x - sampler._denoise(x, s_hat.expand(B), cond=cond, tfeat=tfeat,
                                  coarse=None)) / s_hat
        x_next = x + (s_next - s_hat) * d
        if s_next > 0:
            d2 = (x_next - sampler._denoise(x_next, s_next.expand(B), cond=cond,
                                            tfeat=tfeat, coarse=None)) / s_next
            x_next = x + (s_next - s_hat) * 0.5 * (d + d2)
        x = x_next
        states.append(x.clone())
        print(f"  step {i + 1:2d}/{sampler.num_steps}  sigma {float(s_cur):9.3f} -> "
              f"{float(s_next):7.3f}", flush=True)
    return states, sig.cpu().numpy()


def to_speed(sampler, x, *, level_idx: int, frame: int) -> np.ndarray:
    """Trajectory state -> wind speed (m/s) on one level and frame."""
    H, W = x.shape[-2:]
    blk = sampler.stats.denormalize(x.reshape(sampler.tau, sampler.n_channels, H, W))
    blk = blk.reshape(sampler.tau, sampler.n_levels, 2, H, W).cpu().numpy()
    u = blk[frame, level_idx, 0]
    v = blk[frame, level_idx, 1]
    return np.hypot(u, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/idiff_m2cond_4yr/step_250000.pt")
    ap.add_argument("--out", default=str(HERE / "diffusion_strip_states.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=18)
    ap.add_argument("--level-idx", type=int, default=9, help="index into the 18 levels")
    ap.add_argument("--frame", type=int, default=0, help="which of the tau frames")
    ap.add_argument("--date", default="2023-01-11T12", help="held-out conditioning time")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    print("loading checkpoint …", flush=True)
    sampler = SpaceTimeSampler(args.ckpt, num_steps=args.num_steps, s_churn=0.0,
                               device=device)
    print(f"  step {sampler.step}, conditional={sampler.conditional}, "
          f"tau={sampler.tau}, levels={sampler.n_levels}")

    # Same fixed 64^2 window the benchmark's single-location protocol uses.
    ref = xr.open_zarr(DATA / "era5_heldout.zarr", consolidated=False, zarr_format=2)
    y0 = (ref.sizes["y"] - 64) // 2
    x0 = (ref.sizes["x"] - 64) // 2
    lat = ref["lat"].values[y0:y0 + 64]
    lon = ref["lon"].values[x0:x0 + 64]
    times = (np.datetime64(args.date, "h")
             + np.arange(sampler.tau).astype("timedelta64[h]"))
    print(f"  window lat {lat[0]:.2f}..{lat[-1]:.2f}  lon {lon[0]:.2f}..{lon[-1]:.2f}")
    print(f"  times {times[0]} .. {times[-1]}")

    cond, tfeat = sampler._condition((64, 64), lat, lon, times)

    g = torch.Generator(device="cpu").manual_seed(int(args.seed))
    z = torch.randn(1, sampler.tau, sampler.n_channels, 64, 64, generator=g).to(device)

    print("sampling …", flush=True)
    states, sig = trajectory(sampler, z, cond=cond, tfeat=tfeat)

    speeds = np.stack([to_speed(sampler, s, level_idx=args.level_idx, frame=args.frame)
                       for s in states])
    level_hpa = int(sampler.stats.levels[args.level_idx])
    np.savez(args.out, speeds=speeds.astype("float32"), sigma=sig.astype("float32"),
             level=level_hpa, frame=args.frame, seed=args.seed,
             date=str(args.date), lat=lat, lon=lon)
    print(f"\nsaved {args.out}")
    print(f"  {speeds.shape[0]} states, level index {args.level_idx} "
          f"(model level {level_hpa}), frame {args.frame}")
    print(f"  final speed range {speeds[-1].min():.2f} .. {speeds[-1].max():.2f} m/s")


if __name__ == "__main__":
    main()
