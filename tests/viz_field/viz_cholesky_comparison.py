"""# %% [markdown]
# RFF vs Cholesky GP Comparison (Interactive)
#
# This notebook-style script compares RFF sampling (`SyntheticFlowField`) against exact
# Cholesky sampling for a 2D Matern GP.
#
# Experiments included:
# 1. Single-sample visual comparison (multiple seeds)
# 2. Distribution of empirical means/variances over grid points
# 3. Empirical covariance matrix comparison against theoretical covariance
#
# Notes:
# - Histograms aggregate statistics over **all grid points**.
# - Mean/variance experiments use **M sampled fields** for each method.
# """

# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import jax
from scipy.spatial.distance import cdist
from scipy.stats import norm

try:
    from tests.viz_field._viz_common import (
        add_project_root_to_path,
        format_params,
        make_grid_locations_2d,
        save_figure,
    )
except ModuleNotFoundError:
    from _viz_common import (  # type: ignore
        add_project_root_to_path,
        format_params,
        make_grid_locations_2d,
        save_figure,
    )

add_project_root_to_path()

from src.env.field import SyntheticFlowField
from src.env.utils.types import GridConfig


# %% [markdown]
# ## Shared Sampling Utilities

# %%
def matern_kernel(
    r: np.ndarray,
    r_prime: np.ndarray,
    sigma: float = 1.0,
    lengthscale: float = 1.0,
    nu: float = 2.5,
) -> np.ndarray:
    """Compute Matern covariance matrix for nu in {0.5, 1.5, 2.5, inf}."""
    dists = cdist(r, r_prime, metric="euclidean")
    tau = dists / lengthscale

    if nu == 0.5:
        return sigma**2 * np.exp(-tau)
    if nu == 1.5:
        sqrt3_tau = np.sqrt(3) * tau
        return sigma**2 * (1 + sqrt3_tau) * np.exp(-sqrt3_tau)
    if nu == 2.5:
        sqrt5_tau = np.sqrt(5) * tau
        return sigma**2 * (1 + sqrt5_tau + (5 * tau**2) / 3) * np.exp(-sqrt5_tau)
    if nu == np.inf:
        return sigma**2 * np.exp(-0.5 * tau**2)
    raise ValueError(f"Unsupported nu={nu}. Use 0.5, 1.5, 2.5, or np.inf.")


class CholeskyGPSampler:
    """Exact GP sampling via Cholesky decomposition."""

    def __init__(
        self,
        locations: np.ndarray,
        sigma: float = 1.0,
        lengthscale: float = 1.0,
        nu: float = 2.5,
        jitter: float = 1e-6,
    ):
        self.locations = locations
        kernel = matern_kernel(locations, locations, sigma, lengthscale, nu)
        kernel += jitter * np.eye(len(locations))
        self.l_chol = np.linalg.cholesky(kernel)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one exact GP sample at predefined locations."""
        z = rng.standard_normal(len(self.locations))
        return self.l_chol @ z


def sample_rff_and_cholesky(
    rff_field: SyntheticFlowField,
    chol_sampler: CholeskyGPSampler,
    n_points: int,
    n_samples: int,
    progress_step: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate matched sets of RFF and Cholesky samples."""
    rff_samples = np.zeros((n_samples, n_points))
    chol_samples = np.zeros((n_samples, n_points))
    for i in range(n_samples):
        rff_field.reset(jax.random.PRNGKey(i))
        rff_samples[i] = np.asarray(rff_field._precomputed_u).ravel()

        rng = np.random.default_rng(i + 10_000)
        chol_samples[i] = chol_sampler.sample(rng)

        if progress_step > 0 and (i + 1) % progress_step == 0:
            print(f"  completed {i + 1}/{n_samples} samples")
    return rff_samples, chol_samples


# %% [markdown]
# ## 1) Single Sample Realizations
# Shows side-by-side realizations from both methods for multiple random seeds.

# %%
def compare_single_samples(
    sigma: float = 1.0,
    lengthscale: float = 2.0,
    nu: float = 2.5,
    n_x: int = 25,
    n_y: int = 25,
    num_features: int = 1_000,
    seeds: tuple[int, ...] = (42, 123, 456),
) -> None:
    config = GridConfig.create(n_x=n_x, n_y=n_y)
    locations = make_grid_locations_2d(n_x, n_y)
    rff_field = SyntheticFlowField(
        config,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
    )
    chol_sampler = CholeskyGPSampler(locations, sigma, lengthscale, nu)

    fig, axes = plt.subplots(2, len(seeds), figsize=(5 * len(seeds), 10))
    for col, seed in enumerate(seeds):
        rff_field.reset(jax.random.PRNGKey(seed))
        rff_sample = np.asarray(rff_field._precomputed_u)
        chol_sample = chol_sampler.sample(np.random.default_rng(seed)).reshape(n_x, n_y)

        vmin = min(rff_sample.min(), chol_sample.min())
        vmax = max(rff_sample.max(), chol_sample.max())

        im0 = axes[0, col].imshow(rff_sample.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"RFF sample (seed={seed})")
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("y" if col == 0 else "")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

        im1 = axes[1, col].imshow(chol_sample.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"Cholesky sample (seed={seed})")
        axes[1, col].set_xlabel("x")
        axes[1, col].set_ylabel("y" if col == 0 else "")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "nu": nu,
            "L_features": num_features,
            "grid": f"{n_x}x{n_y}",
        }
    )
    fig.suptitle(
        "RFF vs Cholesky: Single GP Realizations Across Seeds\n"
        f"{subtitle}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, f"single_samples_l{lengthscale}_nu{nu}.png", bbox_inches="tight")
    print(f"Saved to: {out}")
    plt.show()

# %%
# Single-sample comparison
compare_single_samples(sigma=4.0, lengthscale=2.0, nu=2.5)

# %% [markdown]
# ## 2) Mean/Variance Distribution
# We sample `M` fields from each method.  
# For each grid point, we compute empirical mean and variance over those `M` draws.  
# Histograms below are over **all grid points**.

# %%
def compare_mean_variance(
    sigma: float = 1.0,
    lengthscale: float = 2.0,
    nu: float = 2.5,
    n_x: int = 15,
    n_y: int = 15,
    num_features: int = 500,
    n_samples: int = 2_000,
) -> None:
    config = GridConfig.create(n_x=n_x, n_y=n_y)
    locations = make_grid_locations_2d(n_x, n_y)
    n_points = n_x * n_y

    rff_field = SyntheticFlowField(
        config,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
    )
    chol_sampler = CholeskyGPSampler(locations, sigma, lengthscale, nu)

    print(f"Generating {n_samples} samples for mean/variance comparison...")
    rff_samples, chol_samples = sample_rff_and_cholesky(rff_field, chol_sampler, n_points, n_samples)

    rff_means = rff_samples.mean(axis=0)
    rff_vars = rff_samples.var(axis=0, ddof=1)
    chol_means = chol_samples.mean(axis=0)
    chol_vars = chol_samples.var(axis=0, ddof=1)
    expected_mean_std = sigma / np.sqrt(n_samples)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(chol_means, bins=40, alpha=0.6, label="Cholesky", density=True)
    axes[0, 0].hist(rff_means, bins=40, alpha=0.6, label="RFF", density=True)
    axes[0, 0].axvline(0, color="k", ls="--", lw=2, label="target mean=0")
    x_mean = np.linspace(min(rff_means.min(), chol_means.min()), max(rff_means.max(), chol_means.max()), 200)
    axes[0, 0].plot(
        x_mean,
        norm.pdf(x_mean, 0, expected_mean_std),
        "r-",
        lw=2,
        label=f"theory N(0, {expected_mean_std:.3f}^2)",
    )
    axes[0, 0].set_title("Histogram of empirical means over grid points")
    axes[0, 0].set_xlabel("mean at each grid point")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(chol_vars, bins=40, alpha=0.6, label="Cholesky", density=True)
    axes[0, 1].hist(rff_vars, bins=40, alpha=0.6, label="RFF", density=True)
    axes[0, 1].axvline(sigma**2, color="k", ls="--", lw=2, label=f"target var={sigma**2:.2f}")
    axes[0, 1].set_title("Histogram of empirical variances over grid points")
    axes[0, 1].set_xlabel("variance at each grid point")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    im_mean = axes[1, 0].imshow(rff_means.reshape(n_x, n_y).T, origin="lower", cmap="RdBu_r")
    axes[1, 0].set_title("RFF mean map (expected near 0)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    plt.colorbar(im_mean, ax=axes[1, 0], fraction=0.046)

    var_vmin = min(rff_vars.min(), chol_vars.min())
    var_vmax = max(rff_vars.max(), chol_vars.max())
    im_var = axes[1, 1].imshow(
        rff_vars.reshape(n_x, n_y).T,
        origin="lower",
        cmap="viridis",
        vmin=var_vmin,
        vmax=var_vmax,
    )
    axes[1, 1].set_title(f"RFF variance map (target {sigma**2:.2f})")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    plt.colorbar(im_var, ax=axes[1, 1], fraction=0.046)

    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "nu": nu,
            "L_features": num_features,
            "M_samples": n_samples,
            "grid": f"{n_x}x{n_y}",
        }
    )
    fig.suptitle(
        "RFF vs Cholesky: Mean/Variance Statistics\n"
        "Histograms are over all grid points after M sample draws per method.\n"
        f"{subtitle}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, f"mean_var_comparison_l{lengthscale}_nu{nu}.png")
    print(f"Saved to: {out}")
    plt.show()

# %%
# Mean/variance comparison over M samples
compare_mean_variance(sigma=4.0, lengthscale=4.0, nu=2.5, n_samples=2_000)

# %% [markdown]
# ## 3) Covariance Structure
# Compares empirical covariance from sampled fields to theoretical Matern covariance.
#
# **Empirical Covariance Calculation:**
# Given $M$ samples of the field at $N$ grid points, we form a matrix $Z \in \mathbb{R}^{M \times N}$
# where each row is one field realization. The empirical covariance is:
# $$\hat{K} = \frac{1}{M-1} (Z - \bar{Z})^T (Z - \bar{Z})$$
# where $\bar{Z} \in \mathbb{R}^{M \times N}$ has each row equal to the sample mean $\frac{1}{M}\sum_{i=1}^M Z_i$.
#
# **Theoretical Covariance:**
# The Matérn kernel between locations $\mathbf{x}_i, \mathbf{x}_j$ is:
# $$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu} \frac{r}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu} \frac{r}{\ell}\right)$$
# where $r = \|\mathbf{x}_i - \mathbf{x}_j\|$, $\ell$ is the lengthscale, $\nu$ is the smoothness, and $K_\nu$ is the modified Bessel function.
#
# **Comparison Metrics:**
# - Visualize: $\hat{K}_{\text{RFF}}$, $\hat{K}_{\text{Cholesky}}$, and $K_{\text{theory}}$ as heatmaps
# - Error matrices: $E_{\text{RFF}} = \hat{K}_{\text{RFF}} - K_{\text{theory}}$ and $E_{\text{Cholesky}} = \hat{K}_{\text{Cholesky}} - K_{\text{theory}}$
# - Relative Frobenius norm: $\epsilon = \frac{\|E\|_F}{\|K_{\text{theory}}\|_F}$ where $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$
# - Scatter plots: Each point $(K_{\text{theory}}[i,j], \hat{K}_{\text{empirical}}[i,j])$ for all pairs $(i,j)$ in the covariance matrix.
#   Perfect agreement lies on the diagonal $y=x$. Deviations indicate sampling variability or approximation error.

# %%
def compare_covariance_structure(
    sigma: float = 1.0,
    lengthscale: float = 2.0,
    nu: float = 2.5,
    n_x: int = 12,
    n_y: int = 12,
    num_features: int = 500,
    n_samples: int = 3_000,
) -> None:
    config = GridConfig.create(n_x=n_x, n_y=n_y)
    locations = make_grid_locations_2d(n_x, n_y)
    n_points = n_x * n_y

    rff_field = SyntheticFlowField(
        config,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
    )
    chol_sampler = CholeskyGPSampler(locations, sigma, lengthscale, nu)

    print(f"Generating {n_samples} samples for covariance comparison...")
    rff_samples, chol_samples = sample_rff_and_cholesky(
        rff_field,
        chol_sampler,
        n_points=n_points,
        n_samples=n_samples,
        progress_step=1_000,
    )

    rff_centered = rff_samples - rff_samples.mean(axis=0, keepdims=True)
    chol_centered = chol_samples - chol_samples.mean(axis=0, keepdims=True)

    k_emp_rff = (rff_centered.T @ rff_centered) / (n_samples - 1)
    k_emp_chol = (chol_centered.T @ chol_centered) / (n_samples - 1)
    k_theory = matern_kernel(locations, locations, sigma, lengthscale, nu)

    err_rff = k_emp_rff - k_theory
    err_chol = k_emp_chol - k_theory
    frob_rff = np.linalg.norm(err_rff, "fro") / np.linalg.norm(k_theory, "fro")
    frob_chol = np.linalg.norm(err_chol, "fro") / np.linalg.norm(k_theory, "fro")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    vmin = min(k_theory.min(), k_emp_rff.min(), k_emp_chol.min())
    vmax = max(k_theory.max(), k_emp_rff.max(), k_emp_chol.max())

    im0 = axes[0, 0].imshow(k_theory, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Theoretical covariance K")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(k_emp_rff, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("RFF empirical covariance")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(k_emp_chol, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Cholesky empirical covariance")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    err_max = max(np.abs(err_rff).max(), np.abs(err_chol).max())
    im3 = axes[1, 0].imshow(np.abs(err_rff), cmap="hot", vmin=0, vmax=err_max)
    axes[1, 0].set_title(f"|RFF error| (rel Frobenius={frob_rff:.4f})")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(np.abs(err_chol), cmap="hot", vmin=0, vmax=err_max)
    axes[1, 1].set_title(f"|Cholesky error| (rel Frobenius={frob_chol:.4f})")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Randomly select up to 80 grid points (without replacement) for scatter plot visualization
    # This subsampling prevents overcrowding when comparing theoretical vs empirical covariance entries
    sample_idx = np.random.default_rng(7).choice(n_points, size= n_points, replace=False) 
    k_sub_theory = k_theory[np.ix_(sample_idx, sample_idx)].ravel()
    k_sub_rff = k_emp_rff[np.ix_(sample_idx, sample_idx)].ravel()
    k_sub_chol = k_emp_chol[np.ix_(sample_idx, sample_idx)].ravel()
    axes[1, 2].scatter(k_sub_theory, k_sub_rff, alpha=0.3, s=10, label="RFF")
    axes[1, 2].scatter(k_sub_theory, k_sub_chol, alpha=0.3, s=10, label="Cholesky")
    axes[1, 2].plot([vmin, vmax], [vmin, vmax], "k--", lw=2, alpha=0.7, label="y=x")
    axes[1, 2].set_xlabel("theoretical K[i,j]")
    axes[1, 2].set_ylabel("empirical K[i,j]")
    axes[1, 2].set_title("Covariance entry comparison")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    # Covariance vs distance plot
    # Compute pairwise distances between all grid points
    distances = cdist(locations, locations, metric="euclidean").ravel()
    k_theory_flat = k_theory.ravel()
    k_emp_rff_flat = k_emp_rff.ravel()
    k_emp_chol_flat = k_emp_chol.ravel()

    # Sort by distance for cleaner visualization
    sort_idx = np.argsort(distances)
    distances_sorted = distances[sort_idx]
    k_theory_sorted = k_theory_flat[sort_idx]
    k_emp_rff_sorted = k_emp_rff_flat[sort_idx]
    k_emp_chol_sorted = k_emp_chol_flat[sort_idx]

    # Subsample for plotting (to avoid overcrowding)
    subsample_step = max(1, len(distances_sorted) // 2000)
    axes[0, 3].scatter(
        distances_sorted[::subsample_step],
        k_theory_sorted[::subsample_step],
        alpha=0.4,
        s=5,
        label="Theory",
        color="black",
    )
    axes[0, 3].scatter(
        distances_sorted[::subsample_step],
        k_emp_rff_sorted[::subsample_step],
        alpha=0.3,
        s=5,
        label="RFF",
        color="blue",
    )
    axes[0, 3].scatter(
        distances_sorted[::subsample_step],
        k_emp_chol_sorted[::subsample_step],
        alpha=0.3,
        s=5,
        label="Cholesky",
        color="red",
    )
    axes[0, 3].set_xlabel("Distance |r - r'|")
    axes[0, 3].set_ylabel("Covariance K(r, r')")
    axes[0, 3].set_title("Covariance vs Distance")
    axes[0, 3].legend()
    axes[0, 3].grid(alpha=0.3)

    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "nu": nu,
            "L_features": num_features,
            "M_samples": n_samples,
            "grid": f"{n_x}x{n_y}",
        }
    )
    fig.suptitle(
        "RFF vs Cholesky: Covariance Structure\n"
        "Empirical covariance estimated from M sample draws for each method.\n"
        f"{subtitle}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, f"cov_comparison_l{lengthscale}_nu{nu}.png")
    print(f"Saved to: {out}")
    print(f"Relative Frobenius error -> RFF: {frob_rff:.4f}, Cholesky: {frob_chol:.4f}")
    plt.show()

# %%
# Covariance comparison over M samples
compare_covariance_structure(sigma=4.0, lengthscale=4.0, nu=1.5, n_samples=3_000)