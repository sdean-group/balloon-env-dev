"""Stage 2 block-length cost on one GPU: the summer space-time U-Net on 64x64 patches, 36 ch,
conditioned on coords (3), time features (6) and 3x36 coarse channels (bracketing frames +
linear interpolation, the S3 proposal). Random data; bf16 autocast; reports s/it and peak GB."""
import sys, time, importlib
from pathlib import Path
import torch
root = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "src").exists() else Path.cwd()
d = root / "src/eval/windeval/generators/infinite_diffusion"
sys.path.insert(0, str(d))
try:
    spacetime = importlib.import_module("spacetime")
except Exception as e:  # relative imports -> go through the package with a jax stub
    print("direct import failed:", type(e).__name__, str(e)[:80]); sys.modules.setdefault("jax", None)
    sys.path.insert(0, str(root / "src/eval")); spacetime = importlib.import_module("windeval.generators.infinite_diffusion.spacetime")
dev = torch.device("cuda"); print("GPU:", torch.cuda.get_device_name(0))
def run(tau, B, iters=6):
    m = spacetime.EDMPrecondSpaceTime(36, tau=tau, cond_channels=3, time_features=6, coarse_channels=108).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    x = torch.randn(B, tau, 36, 64, 64, device=dev); c = torch.randn(B, 3, 64, 64, device=dev)
    tf = torch.randn(B, tau, 6, device=dev); co = torch.randn(B, tau, 108, 64, 64, device=dev)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m.loss(x, cond=c, tfeat=tf, coarse=co)
        loss.backward(); opt.step()
    for _ in range(3): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / iters
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  tau={tau:2d} B={B:2d}: {dt:5.2f} s/it  {B*tau/dt:6.1f} patch-frames/s  peak {torch.cuda.max_memory_allocated()/1e9:5.1f} GB  ({n:.1f}M params)", flush=True)
    del m, opt, x, c, tf, co; torch.cuda.empty_cache()
for tau, B in ((4, 16), (7, 16), (13, 8), (13, 16), (25, 8)):
    try: run(tau, B)
    except torch.cuda.OutOfMemoryError: print(f"  tau={tau} B={B}: OOM"); torch.cuda.empty_cache()
