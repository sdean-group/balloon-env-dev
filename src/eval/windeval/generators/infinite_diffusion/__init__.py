"""InfiniteDiffusion wind-field generator (PyTorch).

A self-contained generator module. The eval harness stays framework-agnostic — this
package depends on torch + infinite-tensor and only ever emits zarr WindArtifacts.

Layout:
- denoiser.py  : the window-denoiser contract (Phi) + an analytic divergence-free toy.
- sampler.py   : InfiniteDiffusion — lazy MultiDiffusion on an infinite lattice, built on
                 the `infinite-tensor` library (the blending *machinery*).
- generator.py : adapter that materializes a WindArtifact for the benchmark.
- net.py       : EDM-preconditioned U-Net (the trainable Phase-2 model).
- data.py      : random-crop ERA5 dataset + per-(level,variable) normalisation.
- train.py     : EDM training loop (config-driven, resumable, cluster entrypoint).
- trained.py   : TrainedWindowDenoiser — wraps a trained model as Phi (the Phase-3 swap).
- gate.py      : single-crop Axis-1 gate (does one sample look like wind?).

See `infinite-diffusion-progress.md` for the staged plan.
"""
from __future__ import annotations

from .denoiser import WindowDenoiser, ToyDivFreeDenoiser
from .sampler import InfiniteDiffusion
from .generator import InfiniteDiffusionGenerator
from .trained import TrainedWindowDenoiser, build_sampler
from .spacetime_infinite import InfiniteSpaceTimeDiffusion, SpaceTimeGrid

__all__ = [
    "WindowDenoiser",
    "ToyDivFreeDenoiser",
    "InfiniteDiffusion",
    "InfiniteDiffusionGenerator",
    "TrainedWindowDenoiser",
    "build_sampler",
    "InfiniteSpaceTimeDiffusion",
    "SpaceTimeGrid",
]
# viz is intentionally NOT imported here (pulls matplotlib); import it explicitly:
#   from src.eval.windeval.generators.infinite_diffusion import viz
