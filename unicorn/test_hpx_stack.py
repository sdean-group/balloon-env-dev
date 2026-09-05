"""Cluster smoke test for the HEALPix stack: layout caches, coarse dataset on the stores, net fwd/bwd.

    srun -p dean -w dean-compute-02 -c 4 --mem=32G -t 00:20:00 python unicorn/test_hpx_stack.py
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/eval/windeval/hpx"))
from layout import FaceLayout, check_layout
from dataset import HpxCoarseBlocks, valid_rows, compute_stats
from net import EDMPrecondHpx

LAYOUT = Path.home() / "data/hpx_layout"
STORES = ["/scratch/sps252/era5_hpx", str(Path.home() / "data/era5_hpx_coarse_2020_2021")]

print("=== layout ===")
for n in (32, 256):
    lay = FaceLayout.build(n, LAYOUT)
    print(f"  nside {n}: {check_layout(n, lay.perm)}")

print("=== dataset (stores may be partial) ===")
t0 = time.time()
for s in STORES:
    v = valid_rows(s, refresh=True); print(f"  {s}: {int(v.sum())}/{len(v)} valid rows")
st = compute_stats(STORES, max_hours=800)
print(f"  stats over {st['n_hours']} hours; per-channel std range {st['std'].min():.2f}..{st['std'].max():.2f} m/s")
ds = HpxCoarseBlocks(STORES, LAYOUT, n_frames=8, stride_hours=6, stats=st, length=64, seed=0, slow_index=True)
print(f"  T={ds.T} hours in RAM ({ds.uv.nbytes/1e9:.1f} GB), {len(ds.block_starts)} block starts, {len(ds.runs)} runs; init {time.time()-t0:.0f}s")
x, c, tf, sl = ds[3]
print(f"  item: x{tuple(x.shape)} coords{tuple(c.shape)} tfeat{tuple(tf.shape)} slow{tuple(sl.shape)} | x mean/std {x.mean():.2f}/{x.std():.2f} | slow present {sl[:,1].mean():.2f}")
assert torch.isfinite(x).all()

print("=== net fwd/bwd, CPU, nside 32, small width ===")
m = EDMPrecondHpx(36, tau=4, net_kwargs=dict(model_channels=32, channel_mult=(1, 2, 2, 2), num_res_blocks=1, attn_resolutions=(8, 4)))
print(f"  params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
xb = torch.stack([ds[i][0][:4] for i in range(2)]); cb = c[None].expand(2, -1, -1, -1, -1).contiguous()
tfb = torch.stack([ds[i][2][:4] for i in range(2)]); slb = torch.stack([ds[i][3][:4] for i in range(2)])
t0 = time.time(); loss = m.loss(xb, cond=cb, tfeat=tfb, slow=slb); loss.backward()
print(f"  loss {loss.item():.3f}, fwd+bwd {time.time()-t0:.1f}s, grads finite: {all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)}")
if torch.cuda.is_available():
    print("=== net on GPU, full width (128 ch), batch 2, tau 8 ===")
    m = EDMPrecondHpx(36, tau=8, net_kwargs=dict(model_channels=128)).cuda()
    xb = torch.stack([ds[i][0] for i in range(2)]).cuda(); tfb = torch.stack([ds[i][2] for i in range(2)]).cuda(); slb = torch.stack([ds[i][3] for i in range(2)]).cuda()
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(3):
        loss = m.loss(xb, cond=cb.cuda(), tfeat=tfb, slow=slb); loss.backward()
    torch.cuda.synchronize()
    print(f"  params {sum(p.numel() for p in m.parameters())/1e6:.1f}M, {(time.time()-t0)/3:.2f} s/iter at batch 2, peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("ALL OK")
