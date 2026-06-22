"""# %% [markdown]
# RFF GP Parameter Study (Interactive)
#
# Notebook-style exploration of how key GP/RFF parameters affect generated fields:
# - Lengthscale: spatial correlation range
# - Sigma: amplitude (marginal std)
# - Nu: Matern smoothness
# - Number of RFF features L: approximation quality
# - 3D field slices: divergence-free velocity behavior across z levels
# """

# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import jax

from _viz_common import add_project_root_to_path, format_params, save_figure

add_project_root_to_path()

from src.env.field import SyntheticFlowField
from src.env.utils.types import GridConfig


# %% [markdown]
# ## Shared Helpers

# %%
def build_2d_field(
    n_x: int,
    n_y: int,
    sigma: float,
    lengthscale: float,
    nu: float,
    num_features: int,
    seed: int = 42,
) -> np.ndarray:
    """Build and sample one 2D RFF field realization."""
    config = GridConfig.create(n_x=n_x, n_y=n_y)
    field = SyntheticFlowField(
        config=config,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
    )
    field.reset(jax.random.PRNGKey(seed))
    return np.asarray(field._precomputed_u)


def _plot_map_and_hist_grid(
    fields: list[np.ndarray],
    top_titles: list[str],
    fig_title: str,
    filename: str,
    cmap: str = "RdBu_r",
    value_limits: tuple[float, float] | None = None,
    xlim_hist: tuple[float, float] | None = None,
) -> None:
    """Generic 2xN panel: map on top, histogram below."""
    n_cols = len(fields)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.8 * n_cols, 9))
    for col, (u_field, title) in enumerate(zip(fields, top_titles)):
        if value_limits is None:
            v_abs = float(np.max(np.abs(u_field)))
            vmin, vmax = -v_abs, v_abs
        else:
            vmin, vmax = value_limits

        im = axes[0, col].imshow(u_field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(title, fontsize=11)
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("y" if col == 0 else "")
        plt.colorbar(im, ax=axes[0, col], shrink=0.8)

        axes[1, col].hist(u_field.ravel(), bins=30, density=True, edgecolor="black", alpha=0.7)
        axes[1, col].axvline(0, color="red", linestyle="--", lw=1.8)
        axes[1, col].set_xlabel("u displacement", fontsize=10)
        axes[1, col].set_ylabel("density" if col == 0 else "", fontsize=10)
        axes[1, col].set_title(f"Histogram over all grid points (std={u_field.std():.2f})", fontsize=10)
        axes[1, col].grid(alpha=0.3)
        if xlim_hist is not None:
            axes[1, col].set_xlim(*xlim_hist)

    fig.suptitle(fig_title, fontsize=13, y=1.02)
    plt.tight_layout()
    out = save_figure(fig, filename, bbox_inches="tight")
    print(f"Saved to: {out}")
    plt.show()


# %% [markdown]
# ## 1) Lengthscale Study
# Histograms below are over all grid points for each realization.

# %%
def study_lengthscale_effect(
    lengthscales: tuple[float, ...] = (1.0, 3.0, 6.0, 12.0),
    sigma: float = 1.0,
    nu: float = 2.5,
    num_features: int = 500,
    grid_shape: tuple[int, int] = (30, 30),
    seed: int = 42,
) -> None:
    n_x, n_y = grid_shape
    fields = [build_2d_field(n_x, n_y, sigma, ell, nu, num_features, seed) for ell in lengthscales]
    titles = [f"lengthscale={ell}" for ell in lengthscales]
    subtitle = format_params(
        {
            "sigma": sigma,
            "nu": nu,
            "L_features": num_features,
            "grid": f"{n_x}x{n_y}",
            "seed": seed,
        }
    )
    _plot_map_and_hist_grid(
        fields=fields,
        top_titles=titles,
        fig_title=(
            "Effect of Lengthscale on Spatial Correlation\n"
            "Top: one field realization. Bottom: histogram over all grid points.\n"
            f"{subtitle}"
        ),
        filename="param_study_lengthscale.png",
        value_limits=(-3.0, 3.0),
    )

# %%
study_lengthscale_effect(lengthscales=(1.0, 3.0, 6.0, 12.0), sigma=4.0, nu=2.5, num_features=500, grid_shape=(30, 30), seed=42)

# %% [markdown]
# ## 2) Sigma Study
# Sigma controls marginal amplitude while leaving correlation shape largely intact.

# %%
def study_sigma_effect(
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    lengthscale: float = 4.0,
    nu: float = 2.5,
    num_features: int = 500,
    grid_shape: tuple[int, int] = (25, 25),
    seed: int = 42,
) -> None:
    n_x, n_y = grid_shape
    fields = [build_2d_field(n_x, n_y, sigma, lengthscale, nu, num_features, seed) for sigma in sigmas]
    titles = [f"sigma={sigma}" for sigma in sigmas]
    subtitle = format_params(
        {
            "lengthscale": lengthscale,
            "nu": nu,
            "L_features": num_features,
            "grid": f"{n_x}x{n_y}",
            "seed": seed,
        }
    )
    max_sigma = max(sigmas)
    _plot_map_and_hist_grid(
        fields=fields,
        top_titles=titles,
        fig_title=(
            "Effect of Sigma on Field Amplitude\n"
            "Top: one field realization. Bottom: histogram over all grid points.\n"
            f"{subtitle}"
        ),
        filename="param_study_sigma.png",
        value_limits=(-3 * max_sigma, 3 * max_sigma),
        xlim_hist=(-3.5 * max_sigma, 3.5 * max_sigma),
    )

# %%
study_sigma_effect(sigmas=(0.5, 1.0, 2.0, 4.0), lengthscale=4.0, nu=2.5, num_features=500, grid_shape=(25, 25), seed=42)
# %% [markdown]
# ## 3) Nu Study (Matern Smoothness)
# Bottom row uses a center-line cross-section to reveal rough vs smooth behavior.

# %%
def study_nu_effect(
    nus: tuple[float, ...] = (0.5, 1.5, 2.5, 10.0),
    sigma: float = 1.0,
    lengthscale: float = 4.0,
    num_features: int = 500,
    grid_shape: tuple[int, int] = (30, 30),
    seed: int = 42,
) -> None:
    n_x, n_y = grid_shape
    nu_labels = ["nu=0.5 (rough)", "nu=1.5", "nu=2.5", "nu=10 (~RBF)"]
    fields = [build_2d_field(n_x, n_y, sigma, lengthscale, nu, num_features, seed) for nu in nus]

    fig, axes = plt.subplots(2, len(fields), figsize=(4.8 * len(fields), 9))
    for col, (u_field, label) in enumerate(zip(fields, nu_labels)):
        im = axes[0, col].imshow(u_field.T, origin="lower", cmap="RdBu_r", vmin=-3.0, vmax=3.0)
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("y" if col == 0 else "")
        plt.colorbar(im, ax=axes[0, col], shrink=0.8)

        mid_idx = n_y // 2
        axes[1, col].plot(u_field[:, mid_idx], "b-", lw=1.6)
        axes[1, col].axhline(0, color="gray", linestyle="--", alpha=0.6)
        axes[1, col].set_xlabel("x", fontsize=10)
        axes[1, col].set_ylabel("u(x, y_mid)" if col == 0 else "", fontsize=10)
        axes[1, col].set_title("Center-line cross-section", fontsize=10)
        axes[1, col].set_ylim(-3.5, 3.5)
        axes[1, col].grid(alpha=0.3)

    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "L_features": num_features,
            "grid": f"{n_x}x{n_y}",
            "seed": seed,
        }
    )
    fig.suptitle(
        "Effect of Nu (Matern Smoothness)\n"
        "Top: one field realization. Bottom: center-line cross section.\n"
        f"{subtitle}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, "param_study_nu.png", bbox_inches="tight")
    print(f"Saved to: {out}")
    plt.show()

# %%
study_nu_effect(nus=(0.5, 1.5, 2.5, 10.0), sigma=4.0, lengthscale=4.0, num_features=500, grid_shape=(30, 30), seed=42)
# %% [markdown]
# ## 4) Number of RFF Features
# Increasing `L` should improve approximation quality to the target kernel.

# %%
def study_num_features_effect(
    feature_counts: tuple[int, ...] = (20, 100, 500, 2_000),
    sigma: float = 1.0,
    lengthscale: float = 4.0,
    nu: float = 2.5,
    grid_shape: tuple[int, int] = (25, 25),
    seed: int = 42,
) -> None:
    n_x, n_y = grid_shape
    fields = [build_2d_field(n_x, n_y, sigma, lengthscale, nu, l_count, seed) for l_count in feature_counts]
    titles = [f"L={l_count} features" for l_count in feature_counts]
    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "nu": nu,
            "grid": f"{n_x}x{n_y}",
            "seed": seed,
        }
    )
    _plot_map_and_hist_grid(
        fields=fields,
        top_titles=titles,
        fig_title=(
            "Effect of Number of RFF Features (Approximation Quality)\n"
            "Top: one field realization. Bottom: histogram over all grid points.\n"
            f"{subtitle}"
        ),
        filename="param_study_num_features.png",
        value_limits=(-3.0, 3.0),
        xlim_hist=(-3.5, 3.5),
    )

# %%
study_num_features_effect(feature_counts=(20, 100, 500, 2_000), sigma=4.0, lengthscale=4.0, nu=2.5, grid_shape=(25, 25), seed=42)

# %% [markdown]
# ## 5) 3D Field Slices
# Shows divergence-free 2D velocity slices `(u, v)` at multiple z levels.

# %%
def study_3d_field(
    sigma: float = 1.0,
    lengthscale: float = 4.0,
    nu: float = 2.5,
    num_features: int = 500,
    grid_shape: tuple[int, int, int] = (20, 20, 8),
    seed: int = 42,
) -> None:
    n_x, n_y, n_z = grid_shape
    config = GridConfig.create(n_x=n_x, n_y=n_y, n_z=n_z)
    field = SyntheticFlowField(
        config=config,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
    )
    field.reset(jax.random.PRNGKey(seed))
    mean_field = field.velocity_field()

    z_levels = [0, max(0, n_z // 4), max(0, n_z // 2), n_z - 1]
    fig, axes = plt.subplots(2, len(z_levels), figsize=(4.8 * len(z_levels), 9))

    x_coords = np.arange(1, n_x + 1)
    y_coords = np.arange(1, n_y + 1)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    for col, z_idx in enumerate(z_levels):
        u_slice = mean_field[:, :, z_idx, 0]
        v_slice = mean_field[:, :, z_idx, 1]
        speed = np.sqrt(u_slice**2 + v_slice**2)

        im = axes[0, col].imshow(speed.T, origin="lower", cmap="viridis", extent=[1, n_x, 1, n_y])
        axes[0, col].quiver(
            x_mesh[::2, ::2],
            y_mesh[::2, ::2],
            u_slice.T[::2, ::2],
            v_slice.T[::2, ::2],
            color="white",
            alpha=0.8,
            scale=15,
        )
        axes[0, col].set_title(f"z={z_idx + 1}: speed + vectors", fontsize=11)
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("y" if col == 0 else "")
        plt.colorbar(im, ax=axes[0, col], label="speed", shrink=0.8)

        axes[1, col].streamplot(x_coords, y_coords, u_slice.T, v_slice.T, density=1.4, color=speed.T, cmap="viridis")
        axes[1, col].set_title(f"z={z_idx + 1}: streamlines", fontsize=11)
        axes[1, col].set_xlabel("x")
        axes[1, col].set_ylabel("y" if col == 0 else "")
        axes[1, col].set_xlim(1, n_x)
        axes[1, col].set_ylim(1, n_y)
        axes[1, col].set_aspect("equal")

    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "nu": nu,
            "L_features": num_features,
            "grid": f"{n_x}x{n_y}x{n_z}",
            "seed": seed,
        }
    )
    fig.suptitle(
        "3D Divergence-Free Velocity Field Across Z Slices\n"
        "Top: speed + quiver. Bottom: streamlines.\n"
        f"{subtitle}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, "param_study_3d_field.png", bbox_inches="tight")
    print(f"Saved to: {out}")
    plt.show()


# %%
study_3d_field(sigma=4.0, lengthscale=4.0, nu=1.5, num_features=500, grid_shape=(20, 20, 8), seed=42)
