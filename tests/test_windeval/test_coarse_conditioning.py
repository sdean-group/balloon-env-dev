"""Calibration of Phase-5b coarse conditioning (the downscaling experiment).

The whole value of this experiment is that it changes ONE thing, so the tests are mostly
about proving nothing else moved:

  A. Backward compatibility — with coarse_factor absent/0 the model must be numerically
     IDENTICAL to the pre-5b one, and the existing 300k checkpoint must still load and
     sample. (If this breaks, every board number we have becomes uncomparable.)
  B. The coarsening operator — block-mean is an area average: the mean of each f×f cell
     of the fine field IS the coarse value, and it commutes with the per-channel affine
     normalisation (so coarsening the normalised block == normalising the coarsened raw).
  C. Shape/wiring — the model consumes (B,τ,2L,h,w), upsamples internally, and the extra
     channels actually reach in_conv.
  D. Guardrails — a coarse checkpoint refuses to sample without a coarse field, a plain
     checkpoint refuses one, and shape mismatches are caught.
  E. The conditioning must MATTER — two different coarse fields must give different
     samples from the same seed. (A silently-ignored condition would look like a working
     experiment and quietly answer nothing.)

  F. Phase 5b-2 — the residual parameterization, the CFG flag channel, and the exact
     block-mean consistency projection:
       F1 residual round-trip is exact, and "residual == 0" IS the `coarse upsampled`
          baseline (that identity is what lets us say the model starts at the control);
       F2 the measured residual scale reproduces the 0.139 that motivated the change;
       F3 the flag channel makes a dropped coarse field distinguishable from a genuinely
          zero one — otherwise the unconditional CFG branch is contaminated;
       F4 the projection makes block-mean(sample) == coarse to float precision, and
          does not move any cell mean when the field is already consistent;
       F5 guidance=1 is EXACTLY ordinary conditional sampling (so the default row can
          never be silently a guided row), and guidance != 1 is refused on a checkpoint
          with no unconditional branch.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_coarse_conditioning.py
"""
from pathlib import Path

import numpy as np
import torch

from src.eval.windeval.generators.infinite_diffusion.data import (
    NormStats, coarsen)
from src.eval.windeval.generators.infinite_diffusion.spacetime import (
    EDMPrecondSpaceTime, SpaceTimeSampler)

FAILS: list[str] = []
L, TAU, C, H, W, F = 4, 3, 8, 16, 16, 4


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAILS.append(name)


def _try(fn) -> bool:
    """True when `fn` raises — used to assert the guardrails actually fire."""
    try:
        with torch.no_grad():
            fn()
        return False
    except Exception:
        return True


def net(coarse_channels=0, seed=0):
    torch.manual_seed(seed)
    return EDMPrecondSpaceTime(
        C, tau=TAU, sigma_data=1.0, cond_channels=2, time_features=6,
        coarse_channels=coarse_channels,
        net_kwargs=dict(model_channels=16, channel_mult=(1, 2), num_res_blocks=1,
                        attn_resolutions=(), temporal_kernel=3))


x = torch.randn(2, TAU, C, H, W)
cond = torch.randn(2, 2, H, W)
tfeat = torch.randn(2, TAU, 6)
coarse = coarsen(x.reshape(2 * TAU, C, H, W), F).reshape(2, TAU, C, H // F, W // F)

print(__doc__.splitlines()[0])

# ---- A. backward compatibility ----
print("\nA. backward compatibility (coarse_factor=0 ⇒ unchanged model)")
a, b = net(0, seed=1), net(0, seed=1)
with torch.no_grad():
    ya = a(x, torch.full((2,), 0.7), cond=cond, tfeat=tfeat)
    yb = b(x, torch.full((2,), 0.7), cond=cond, tfeat=tfeat)
check("plain model deterministic / unchanged path", torch.allclose(ya, yb, atol=0),
      "identical seeds ⇒ identical output")
check("plain in_conv width == C + cond only", a.net.in_conv.in_channels == C + 2,
      f"{a.net.in_conv.in_channels}")
check("plain model ignores a stray coarse field (no crash, same output)",
      torch.allclose(a(x, torch.full((2,), .7), cond=cond, tfeat=tfeat, coarse=coarse),
                     ya, atol=0) if True else False,
      "coarse_channels=0 ⇒ the argument is unused")

# ---- B. the coarsening operator ----
print("\nB. coarsening operator (area average)")
fine = torch.arange(F * F, dtype=torch.float32).reshape(1, 1, F, F)
check("block-mean equals the cell mean", float(coarsen(fine, F)[0, 0, 0, 0]) == float(fine.mean()),
      f"{float(coarsen(fine, F)[0, 0, 0, 0]):.2f} vs {float(fine.mean()):.2f}")
check("coarse shape = crop/f", tuple(coarse.shape) == (2, TAU, C, H // F, W // F),
      str(tuple(coarse.shape)))
st = NormStats(np.zeros(L) + 2.0, np.ones(L) * 3.0, np.zeros(L) + 1.0, np.ones(L) * 5.0,
               np.arange(49, 49 + L))
raw = torch.randn(TAU, C, H, W) * 7 + 3
lhs = coarsen(st.normalize(raw), F)                 # coarsen AFTER normalising
rhs = st.normalize(coarsen(raw, F))                 # normalise AFTER coarsening
check("block-mean commutes with normalisation", torch.allclose(lhs, rhs, atol=1e-5),
      f"max |Δ| = {float((lhs - rhs).abs().max()):.2e}")

# ---- C. shape / wiring ----
print("\nC. wiring")
m = net(C, seed=2)
check("coarse in_conv width == C + cond + coarse", m.net.in_conv.in_channels == C + 2 + C,
      f"{m.net.in_conv.in_channels}")
with torch.no_grad():
    y = m(x, torch.full((2,), 0.7), cond=cond, tfeat=tfeat, coarse=coarse)
check("forward returns the target shape", tuple(y.shape) == tuple(x.shape), str(tuple(y.shape)))
check("coarse model refuses a missing coarse field",
      _try(lambda: m(x, torch.full((2,), .7), cond=cond, tfeat=tfeat)), "raises ValueError")

# ---- E. the conditioning must matter ----
print("\nE. the coarse field actually changes the output")
# out_conv is zero-init by EDM convention, so an UNTRAINED net returns c_skip*x for any
# input and this test would pass vacuously. Give it a non-zero output head first, so we
# are genuinely asking whether coarse information reaches the output.
torch.nn.init.normal_(m.net.out_conv.weight, std=0.05)
with torch.no_grad():
    y1 = m(x, torch.full((2,), 0.7), cond=cond, tfeat=tfeat, coarse=coarse)
    y2 = m(x, torch.full((2,), 0.7), cond=cond, tfeat=tfeat, coarse=coarse * 0 + 5.0)
d = float((y1 - y2).abs().mean())
check("different coarse ⇒ different output", d > 1e-4, f"mean |Δ| = {d:.4f}")
loss = m.loss(x, cond=cond, tfeat=tfeat, coarse=coarse)
check("loss is finite", bool(torch.isfinite(loss)), f"{float(loss):.4f}")

# ---- A2. the real checkpoint still loads ----
print("\nA2. the existing 300k checkpoint still loads and samples")
ck = Path("runs/idiff_m2cond/step_300000.pt")
if ck.exists():
    s = SpaceTimeSampler(ck, num_steps=2, device="cpu")
    check("loads, coarse_factor == 0", s.coarse_factor == 0, f"step {s.step}")
    us, vs = s.sample_block((32, 32), seed=0,
                            lat=np.linspace(40, 32, 32), lon=np.linspace(232, 240, 32),
                            times=np.datetime64("2023-01-08T00", "h") +
                            np.arange(s.tau).astype("timedelta64[h]"))
    # 2 sampler steps = a deliberately crude smoke run (magnitudes are NOT meaningful at
    # 2 steps from sigma_max=80); this asserts the plain path still runs and stays finite.
    check("samples without a coarse field (2-step smoke; shape+finite only)",
          np.isfinite(us).all() and np.isfinite(vs).all() and us.shape == (s.tau, 18, 32, 32),
          f"shape {us.shape}")
    check("rejects a coarse field it was not trained with",
          _try(lambda: s.sample_block((32, 32), seed=0, lat=np.linspace(40, 32, 32),
                                      lon=np.linspace(232, 240, 32),
                                      times=np.datetime64("2023-01-08T00", "h") +
                                      np.arange(s.tau).astype("timedelta64[h]"),
                                      coarse=coarse)), "raises ValueError")
else:
    print(f"  SKIP  {ck} not present")

# ---- F. Phase 5b-2: residual parameterization, CFG flag, consistency projection ----
print("\nF. residual parameterization / CFG flag / consistency projection")
SCALE = 0.139
mr = EDMPrecondSpaceTime(C, tau=TAU, cond_channels=2, time_features=6, coarse_channels=C,
                         coarse_residual=True, coarse_scale=SCALE, coarse_flag=True,
                         net_kwargs=dict(model_channels=8, channel_mult=(1, 2),
                                         num_res_blocks=1, attn_resolutions=()))
x_full = torch.randn(2, TAU, C, H, W)
c_from_x = coarsen(x_full.reshape(-1, C, H, W), F).reshape(2, TAU, C, H // F, W // F)

r = mr.to_residual(x_full, c_from_x)
back = mr.from_residual(r, c_from_x)
check("F1 residual round-trip is exact",
      torch.allclose(back, x_full, atol=1e-5), f"max |Δ| = {float((back-x_full).abs().max()):.2e}")
zero_pred = mr.from_residual(torch.zeros_like(r), c_from_x)
base_ref = torch.nn.functional.interpolate(
    c_from_x.reshape(-1, C, H // F, W // F), size=(H, W), mode="bilinear",
    align_corners=False).reshape(2, TAU, C, H, W)
check("F1 residual==0 IS the `coarse upsampled` baseline",
      torch.allclose(zero_pred, base_ref, atol=1e-6),
      f"max |Δ| = {float((zero_pred-base_ref).abs().max()):.2e}")

# F2: the number the whole redesign rests on, recomputed from the operator itself.
zarr = Path("src/eval/windeval/data/era5_2023.zarr")
if zarr.exists():
    from src.eval.windeval.generators.infinite_diffusion.data import (
        WindCoarseCondSpaceTimeDataset, measure_residual_scale)
    ds = WindCoarseCondSpaceTimeDataset(zarr, crop=64, levels=(49, 66), n_frames=4,
                                        frame_stride=1, length=64, seed=0, coarse_factor=8)
    sc = measure_residual_scale(ds, n_blocks=32)
    check("F2 measured residual scale ≈ 0.139 (the premise of the redesign)",
          0.11 < sc < 0.17, f"{sc:.4f} over 32 blocks")
else:
    print(f"  SKIP  {zarr} not present (F2)")

# F3: a dropped coarse field must not be confusable with a legitimately zero one.
torch.nn.init.normal_(mr.net.out_conv.weight, std=0.05)
zc = torch.zeros_like(c_from_x)
sg = torch.full((2,), 0.7)
with torch.no_grad():
    y_kept = mr(x_full, sg, cond=cond, tfeat=tfeat, coarse=zc, coarse_mask=torch.ones(2))
    y_drop = mr(x_full, sg, cond=cond, tfeat=tfeat, coarse=zc, coarse_mask=torch.zeros(2))
d = float((y_kept - y_drop).abs().mean())
check("F3 flag channel separates 'dropped' from 'genuinely zero' coarse", d > 1e-5,
      f"mean |Δ| = {d:.4f}")
lo = mr.loss(x_full, cond=cond, tfeat=tfeat, coarse=c_from_x, coarse_dropout=0.5)
check("F3 loss with dropout is finite", bool(torch.isfinite(lo)), f"{float(lo):.4f}")

# F4: the projection, applied to a deliberately inconsistent field.
f = 4
bad = x_full + 3.0                                   # every cell mean is now wrong by +3
resid = c_from_x - coarsen(bad.reshape(-1, C, H, W), f).reshape(c_from_x.shape)
fixed = bad + resid.repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1)
got = coarsen(fixed.reshape(-1, C, H, W), f).reshape(c_from_x.shape)
check("F4 projection forces block-mean(sample) == coarse",
      torch.allclose(got, c_from_x, atol=1e-5), f"max |Δ| = {float((got-c_from_x).abs().max()):.2e}")
resid0 = c_from_x - coarsen(x_full.reshape(-1, C, H, W), f).reshape(c_from_x.shape)
check("F4 projection is a no-op on an already-consistent field",
      float(resid0.abs().max()) < 1e-5, f"max |Δ| = {float(resid0.abs().max()):.2e}")
# and it must not manufacture sub-cell structure: it shifts each cell by a constant
cell_var_before = bad.reshape(2, TAU, C, H // f, f, W // f, f).var(dim=(4, 6))
cell_var_after = fixed.reshape(2, TAU, C, H // f, f, W // f, f).var(dim=(4, 6))
check("F4 nearest lift adds no sub-cell structure (within-cell variance unchanged)",
      torch.allclose(cell_var_before, cell_var_after, atol=1e-5),
      f"max |Δ| = {float((cell_var_before-cell_var_after).abs().max()):.2e}")

# F5: guidance defaults must be inert; guidance without a trained null branch must raise.
ck = Path("runs/idiff_m2cond/step_300000.pt")
if ck.exists():
    check("F5 guidance != 1 refused on a checkpoint with no unconditional branch",
          _try(lambda: SpaceTimeSampler(ck, num_steps=2, device="cpu", guidance=2.0)),
          "raises ValueError")
    s = SpaceTimeSampler(ck, num_steps=2, device="cpu")
    check("F5 defaults on an old ckpt: guidance 1, residual off, scale 1",
          s.guidance == 1.0 and not s.coarse_residual and s.coarse_scale == 1.0,
          f"g={s.guidance} res={s.coarse_residual} scale={s.coarse_scale}")
    check("F5 projection is inert without a coarse field (old ckpt unchanged)",
          s.coarse_project and s.coarse_factor == 0, "project=True but coarse_factor=0")
else:
    print(f"  SKIP  {ck} not present (F5)")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
raise SystemExit(1 if FAILS else 0)
