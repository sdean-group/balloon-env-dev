"""Phase-4: amplitude (scale) metric + stochastic-sampler determinism.

Two independent guarantees introduced in Phase 4:
  1. ``field_scores(ds, ref_rms=...)`` adds a scale metric (``amplitude rms`` always;
     ``score: amplitude`` when a peer RMS is given) WITHOUT changing the reference-free
     structure scores or COMPOSITE — amplitude is a diagnostic, not folded into COMPOSITE.
  2. The stochastic internal EDM sampler (``s_churn > 0``) stays *deterministic in its
     input window* (churn noise is seeded from the window), so the machinery's tile cache
     and Axis-2 exact revisit-determinism survive. This is the property the whole
     two-halves design rests on, so it gets a regression test.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_amplitude_churn.py
"""
from pathlib import Path

import numpy as np

try:
    from src.eval.windeval import artifact
    from src.eval.windeval.metrics import field_scores, amplitude_rms
    from src.eval.windeval.metrics.realism import _amplitude_score
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e

DATA = "src/eval/windeval/data/era5_real.zarr"
CKPT = "runs/idiff_m1/step_84000.pt"


def run():
    if not HAVE:
        print(f"SKIP test_amplitude_churn (deps not installed: {_ERR})")
        return True
    if not Path(DATA).exists():
        print(f"SKIP test_amplitude_churn (data missing: {DATA})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(ok)

    # --- 1. amplitude metric ---
    ds = artifact.read(DATA)
    ref = amplitude_rms(ds)
    chk("amplitude_rms positive + finite", np.isfinite(ref) and ref > 0, f"rms {ref:.2f}")

    base = field_scores(ds)
    withref = field_scores(ds, ref_rms=ref)
    chk("amplitude rms reported unconditionally", "amplitude rms" in base)
    chk("score: amplitude only when ref given",
        "score: amplitude" not in base and "score: amplitude" in withref)
    chk("self-score ~ 1.0 (field vs its own rms)",
        abs(withref["score: amplitude"] - 1.0) < 1e-6, f"{withref['score: amplitude']:.4f}")
    chk("COMPOSITE is reference-free (unchanged by ref_rms)",
        base["COMPOSITE"] == withref["COMPOSITE"], f"{base['COMPOSITE']:.4f}")

    # symmetric ratio score: too-calm and too-energetic by the same factor score equal
    s_half = _amplitude_score(ref * 0.5, ref)
    s_double = _amplitude_score(ref * 2.0, ref)
    chk("amplitude score symmetric (0.5x == 2x == 0.5)",
        abs(s_half - 0.5) < 1e-9 and abs(s_double - 0.5) < 1e-9,
        f"half {s_half:.3f} double {s_double:.3f}")

    # --- 2. stochastic sampler stays deterministic-in-window ---
    if not Path(CKPT).exists():
        print(f"  [SKIP] churn determinism (checkpoint missing: {CKPT})")
    else:
        import torch
        from src.eval.windeval.generators.infinite_diffusion.trained import TrainedWindowDenoiser
        dev = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
               else "cpu")
        phi = TrainedWindowDenoiser(CKPT, num_steps=6, device=dev, s_churn=40.0)
        g = torch.Generator(device="cpu").manual_seed(7)
        x = torch.randn(phi.n_channels, 32, 32, generator=g)
        a = phi(x.clone()).cpu().numpy()
        b = phi(x.clone()).cpu().numpy()
        chk("churn ON: identical window -> identical output (max|d|=0)",
            np.abs(a - b).max() == 0.0, f"max|d|={np.abs(a - b).max():.1e}")
        # a *different* window must give a *different* sample (churn is actually active)
        y = torch.randn(phi.n_channels, 32, 32, generator=torch.Generator().manual_seed(8))
        c = phi(y).cpu().numpy()
        chk("churn ON: different window -> different output", np.abs(a - c).max() > 0)

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
