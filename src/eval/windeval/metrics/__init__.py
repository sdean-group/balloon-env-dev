"""Benchmark v2 metrics — reference-based, raw values only (see docs/benchmark-v2-changes.md)."""
from .suite import run_suite, tiling_penalty, METRIC_INFO  # noqa: F401
from .spectra import dataset_spectra, spectral_residual, effective_resolution  # noqa: F401
from .distributions import wasserstein1, marginal_w1, extreme_quantile_error, conditional_w1  # noqa: F401
from .shear import climatological_dz, shear_w1  # noqa: F401
from .temporal import temporal_psd, trajectory_dispersion, dispersion_compare, has_time  # noqa: F401
