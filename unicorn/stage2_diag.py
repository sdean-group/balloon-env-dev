import sys, time, os, numpy as np, torch, zarr, healpy as hp, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src/eval/windeval/hpx")
from sample_fine import HpxFineSampler
root = zarr.open_group("/scratch/sps252/era5_hpx_heldout_2023", mode="r"); hours = np.asarray(root["time"][:], dtype=np.int64)
row = {int(h): i for i, h in enumerate(hours)}
h0 = int((np.datetime64("2023-07-10T00") - np.datetime64("1900-01-01T00")) / np.timedelta64(1, "h")); hs = h0 + np.arange(13)
coarse = np.stack([root["coarse/uv"][row[h0 + 6 * k]] for k in range(3)]).astype(np.float32)
era = np.stack([root["fine/uv"][row[h]] for h in hs]).astype(np.float32)
lmax = 767
def spec(x):
    m = hp.reorder(x.astype(np.float64), n2r=True); return hp.anafast(m - m.mean(), lmax=lmax)
bands = ((10, 96), (96, 256), (256, 512), (512, 768))
ce = spec(era[6, 0]); print("ERA5 ch0 f6 band power (mean C_l):", ["%.2e" % ce[a:b].mean() for a, b in bands], flush=True)
lin = np.repeat(0.5 * (coarse[0] + coarse[1]), 64, axis=-1)  # not used for f6; use the baseline at f6 = coarse[1] lifted
base6 = np.repeat(coarse[1], 64, axis=-1)
cb = spec(base6[0]); print("lifted-coarse (nearest) ch0 f6 band power:", ["%.2e" % cb[a:b].mean() for a, b in bands], " -> log ratio vs ERA5", [round(float(np.log(cb[a:b].mean() / ce[a:b].mean())), 2) for a, b in bands], flush=True)
out = {}
for label, kw in (("smin0.002", dict()), ("smin0.02", dict(sigma_min=0.02)), ("smin0.05", dict(sigma_min=0.05))):
    s = HpxFineSampler("/scratch/sps252/runs/stage2_hpx256/step_75000.pt", os.path.expanduser("~/data/hpx_layout"), num_steps=18, device="cuda", batch_patches=48, **kw)
    t1 = time.time(); gen = s.sample_block(coarse, hs, seed=0, log=None); dt = time.time() - t1
    cg = spec(gen[6, 0]); d = gen[6, 0] - era[6, 0]; cd = spec(d)
    print(f"{label}: {dt:.0f}s | gen band log-ratio vs ERA5:", [round(float(np.log(cg[a:b].mean() / ce[a:b].mean())), 2) for a, b in bands],
          "| (gen-era) band power:", ["%.2e" % cd[a:b].mean() for a, b in bands],
          "| residual rms %.3f | max|gen-era| %.1f | pixel-diff rms gen %.3f era %.3f" % (
              np.sqrt(((gen - np.repeat(gen.reshape(13, 36, 12288, 64).mean(-1), 64, -1)) ** 2).mean()), np.abs(d).max(),
              np.sqrt((np.diff(gen[6, 0].reshape(12288, 64), axis=1) ** 2).mean()), np.sqrt((np.diff(era[6, 0].reshape(12288, 64), axis=1) ** 2).mean())), flush=True)
    out[label] = gen[6, :2].astype(np.float16)
    del s; torch.cuda.empty_cache()
np.savez_compressed("/scratch/sps252/runs/stage2_hpx256/diag_jul10_f6.npz", era=era[6, :2].astype(np.float16), coarse=coarse[1, :2], **out)
