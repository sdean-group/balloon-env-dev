"""Throughput/memory benchmark for the Stage 1 network on one GPU (random data).

    srun -p dean -w dean-compute-02 --gres=gpu:1 -c 8 --mem=32G -t 00:30:00 python unicorn/bench_stage1.py

Measures it/s and peak memory for: fp32 vs bf16 autocast, padding backend (whatever
earth2grid exposes), torch.compile, block length τ, and batch size — the numbers that decide
the Stage 1 config. Prints one table.
"""
import inspect, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/eval/windeval/hpx"))
from net import EDMPrecondHpx
from earth2grid import healpix

dev = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("pad_backend:", inspect.signature(healpix.pad_backend), "| PaddingBackends:", [n for n in dir(healpix.PaddingBackends) if not n.startswith("_")])


def make(tau, ch=128):
    m = EDMPrecondHpx(36, tau=tau, net_kwargs=dict(model_channels=ch)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    return m, opt


def run(m, opt, B, tau, amp, iters=8, compiled=None):
    x = torch.randn(B, tau, 36, 12, 32, 32, device=dev); c = torch.randn(B, 3, 12, 32, 32, device=dev)
    tf = torch.randn(B, tau, 6, device=dev); sl = torch.zeros(B, tau, 2, device=dev)
    f = compiled or m
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = f.loss(x, cond=c, tfeat=tf, slow=sl)
        loss.backward(); opt.step()
    for _ in range(3):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt, torch.cuda.max_memory_allocated() / 1e9


rows = []
def record(label, B, tau, dt, mem):
    rows.append((label, B, tau, dt, mem))
    print(f"  {label:34s} B={B} tau={tau}: {dt:6.2f} s/it  {1/dt:5.2f} it/s  {B*tau*12288/dt/1e6:6.2f} Mpx-frames/s  peak {mem:5.1f} GB", flush=True)

m, opt = make(8)
print("params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
record("fp32", 2, 8, *run(m, opt, 2, 8, amp=False))
record("bf16 autocast", 2, 8, *run(m, opt, 2, 8, amp=True))
# padding backends, if selectable
for name in [n for n in dir(healpix.PaddingBackends) if not n.startswith("_")]:
    try:
        be = getattr(healpix.PaddingBackends, name)
        try:
            ctx = healpix.pad_backend(be)
            if hasattr(ctx, "__enter__"):
                with ctx:
                    dt, mem = run(m, opt, 2, 8, amp=True)
            else:
                dt, mem = run(m, opt, 2, 8, amp=True)
        except TypeError:
            healpix.pad_backend(be); dt, mem = run(m, opt, 2, 8, amp=True)
        record(f"bf16 + pad backend {name}", 2, 8, dt, mem)
    except Exception as e:
        print(f"  pad backend {name}: not usable ({type(e).__name__}: {str(e)[:80]})")
# batch scaling and shorter blocks
for B in (4, 6):
    try:
        record("bf16 autocast", B, 8, *run(m, opt, B, 8, amp=True))
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B} tau=8: OOM"); torch.cuda.empty_cache()
m4, opt4 = make(4)
record("bf16 autocast", 4, 4, *run(m4, opt4, 4, 4, amp=True))
record("bf16 autocast", 8, 4, *run(m4, opt4, 8, 4, amp=True))
# compile
try:
    mc = torch.compile(m)
    record("bf16 + torch.compile", 2, 8, *run(m, opt, 2, 8, amp=True, compiled=mc))
    record("bf16 + torch.compile", 4, 8, *run(m, opt, 4, 8, amp=True, compiled=mc))
except Exception as e:
    print(f"  torch.compile failed: {type(e).__name__}: {str(e)[:120]}")
print("\nlabel, B, tau, s/it, peak GB")
for r in rows:
    print("  ", r)
