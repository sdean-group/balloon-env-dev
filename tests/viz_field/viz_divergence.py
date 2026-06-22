"""# %% [markdown]
# Divergence Verification for 3D RFF Streamfunction Field (Interactive)
#
# This notebook-style script validates the divergence-free construction in 3D:
# - The field defines `u = -dpsi/dy` and `v = dpsi/dx`
# - Hence `du/dx + dv/dy` should be approximately 0 (up to numerical precision)
#
# Experiments included:
# 1. Single realization divergence map + histogram
# 2. Distribution of divergence over multiple realizations and random points
# 3. Analytical autodiff divergence vs finite-difference divergence
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
# ## Shared Divergence Helpers

# %%
def build_3d_field(
    n_x: int = 15,
    n_y: int = 15,
    n_z: int = 8,
    sigma: float = 1.0,
    lengthscale: float = 3.0,
    nu: float = 2.5,
    num_features: int = 500,
    seed: int = 42,
) -> SyntheticFlowField:
    """Create and reset a 3D RFF streamfunction field."""
    config = GridConfig.create(n_x=n_x, n_y=n_y, n_z=n_z)
    field = SyntheticFlowField(
        config=config,
        sigma=sigma,
        lengthscale=lengthscale,
        nu=nu,
        num_features=num_features,
    )
    field.reset(jax.random.PRNGKey(seed))
    return field


def compute_divergence_at_point(field: SyntheticFlowField, x: float, y: float, z: float) -> float:
    """Compute divergence du/dx + dv/dy using JAX autodiff."""
    def u_fn(x_, y_, z_):
        u, _ = field.velocity_at_point(x_, y_, z_)
        return u

    def v_fn(x_, y_, z_):
        _, v = field.velocity_at_point(x_, y_, z_)
        return v

    du_dx = jax.grad(u_fn, argnums=0)(x, y, z)
    dv_dy = jax.grad(v_fn, argnums=1)(x, y, z)
    return float(du_dx + dv_dy)


def compute_divergence_grid(field: SyntheticFlowField, n_samples: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate divergence on a regular x-y grid at mid z."""
    x_vals = np.linspace(1.5, field.config.n_x - 0.5, n_samples)
    y_vals = np.linspace(1.5, field.config.n_y - 0.5, n_samples)
    z_mid = field.config.n_z / 2.0
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    div_grid = np.zeros_like(x_mesh)

    for i in range(n_samples):
        for j in range(n_samples):
            div_grid[i, j] = compute_divergence_at_point(field, x_mesh[i, j], y_mesh[i, j], z_mid)
    return x_mesh, y_mesh, div_grid


# %% [markdown]
# ## 1) Single Realization: Velocity + Divergence
# The middle panel visualizes divergence on a dense x-y grid at a fixed z level.

# %%
def visualize_divergence_single_field(
    sigma: float = 1.0,
    lengthscale: float = 3.0,
    nu: float = 2.5,
    num_features: int = 500,
    grid_shape: tuple[int, int, int] = (15, 15, 8),
    seed: int = 42,
) -> None:
    n_x, n_y, n_z = grid_shape
    field = build_3d_field(n_x, n_y, n_z, sigma, lengthscale, nu, num_features, seed)
    mean_field = field.velocity_field()
    z_idx = n_z // 2
    u_field = mean_field[:, :, z_idx, 0]
    v_field = mean_field[:, :, z_idx, 1]

    x_mesh, y_mesh, div_grid = compute_divergence_grid(field, n_samples=25)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    speed = np.sqrt(u_field**2 + v_field**2)
    im0 = axes[0].imshow(speed.T, origin="lower", cmap="viridis", extent=[1, n_x, 1, n_y])
    skip = 1
    x_coords = np.arange(1, n_x + 1)
    y_coords = np.arange(1, n_y + 1)
    vx, vy = np.meshgrid(x_coords, y_coords)
    axes[0].quiver(
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        u_field.T[::skip, ::skip],
        v_field.T[::skip, ::skip],
        color="white",
        alpha=0.75,
        scale=20,
    )
    axes[0].set_title(f"Velocity magnitude and vectors at z={z_idx + 1}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], label="speed")

    vmax = max(abs(div_grid.min()), abs(div_grid.max()))
    im1 = axes[1].contourf(x_mesh, y_mesh, div_grid, levels=24, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Divergence map (max |div|={np.abs(div_grid).max():.2e})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], label="du/dx + dv/dy")

    axes[2].hist(div_grid.ravel(), bins=50, edgecolor="black", alpha=0.7)
    axes[2].axvline(0, color="red", ls="--", lw=2, label="target=0")
    axes[2].set_title("Divergence histogram over sampled x-y points")
    axes[2].set_xlabel("divergence")
    axes[2].set_ylabel("count")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

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
        "Single-Realization Divergence Verification (3D Streamfunction RFF Field)\n"
        f"{subtitle}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, "divergence_single.png")
    print(f"Saved to: {out}")
    print(
        f"Stats: max|div|={np.abs(div_grid).max():.2e}, "
        f"mean|div|={np.abs(div_grid).mean():.2e}, std={div_grid.std():.2e}"
    )
    plt.show()

# %%
visualize_divergence_single_field(sigma=4.0, lengthscale=4.0, nu=1.5)
# %% [markdown]
# ## 2) Multiple Realizations: Distribution of Divergence
# For each realization we sample random continuous points and compute divergence.

# %%
def visualize_divergence_multiple_realizations(
    n_realizations: int = 20,
    n_test_points: int = 50,
    sigma: float = 1.0,
    lengthscale: float = 3.0,
    nu: float = 2.5,
    num_features: int = 500,
    grid_shape: tuple[int, int, int] = (12, 12, 6),
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

    all_divergences = []
    for seed in range(n_realizations):
        field.reset(jax.random.PRNGKey(seed))
        rng = np.random.default_rng(seed)
        x_pts = rng.uniform(1.5, n_x - 0.5, n_test_points)
        y_pts = rng.uniform(1.5, n_y - 0.5, n_test_points)
        z_pts = rng.uniform(1.5, n_z - 0.5, n_test_points)
        for x, y, z in zip(x_pts, y_pts, z_pts):
            all_divergences.append(compute_divergence_at_point(field, x, y, z))

    all_divergences = np.asarray(all_divergences)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_divergences, bins=60, edgecolor="black", alpha=0.7, density=True)
    ax.axvline(0, color="red", linestyle="--", lw=2, label="target=0")
    ax.set_xlabel("divergence")
    ax.set_ylabel("density")
    ax.set_title(
        "Divergence distribution across realizations\n"
        f"{n_realizations} realizations x {n_test_points} random points each"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    stats_text = (
        f"max |div|: {np.abs(all_divergences).max():.2e}\n"
        f"mean |div|: {np.abs(all_divergences).mean():.2e}\n"
        f"std: {all_divergences.std():.2e}\n"
        f"n total: {len(all_divergences)}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    subtitle = format_params(
        {
            "sigma": sigma,
            "lengthscale": lengthscale,
            "nu": nu,
            "L_features": num_features,
            "grid": f"{n_x}x{n_y}x{n_z}",
        }
    )
    fig.suptitle(f"Multi-Realization Divergence Check\n{subtitle}", fontsize=12, y=1.01)
    plt.tight_layout()
    out = save_figure(fig, "divergence_multiple.png")
    print(f"Saved to: {out}")
    plt.show()
# %%
visualize_divergence_multiple_realizations(sigma=4.0, lengthscale=4.0, nu=1.5, n_realizations=100)

# %% [markdown]
# ## 3) Autodiff vs Finite Difference
# 
# We compare two methods for computing divergence on a 2D slice at $z = z_{\text{idx}}$:
# 
# **Finite Difference (FD):**
# Given the discrete velocity field $u[i,j]$ and $v[i,j]$ on an integer grid,
# we approximate the divergence as:
# $$
# \text{div}_{\text{FD}}[i,j] \approx \frac{u[i+1,j] - u[i-1,j]}{2\Delta x} + \frac{v[i,j+1] - v[i,j-1]}{2\Delta y}
# $$
# using `numpy.gradient` with $\Delta x = \Delta y = 1.0$. This is a second-order centered
# difference scheme that introduces discretization error $\mathcal{O}(\Delta x^2)$.
# 
# **Autodiff (Analytical):**
# The velocity field is constructed from a streamfunction $\psi$ via:
# $$
# u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}
# $$
# By construction, the divergence is:
# $$
# \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = \frac{\partial^2 \psi}{\partial x \partial y} - \frac{\partial^2 \psi}{\partial y \partial x} = 0
# $$
# analytically (by equality of mixed partials). We use JAX autodiff to compute exact derivatives
# at continuous coordinates $(i+1, j+1, z_{\text{idx}}+1)$, evaluating the divergence without discretization error.
# 
# **The plots show:**
# - Left: FD divergence field
# - Middle: Autodiff divergence field (should be numerically $\approx 0$)
# - Right: Scatter plot comparing the two methods point-by-point

# %%
def compare_analytical_vs_finite_diff(
    sigma: float = 1.0,
    lengthscale: float = 3.0,
    nu: float = 2.5,
    num_features: int = 500,
    grid_shape: tuple[int, int, int] = (20, 20, 8),
    seed: int = 42,
) -> None:
    n_x, n_y, n_z = grid_shape
    field = build_3d_field(n_x, n_y, n_z, sigma, lengthscale, nu, num_features, seed)
    mean_field = field.velocity_field()
    z_idx = n_z // 2
    u_field = mean_field[:, :, z_idx, 0]
    v_field = mean_field[:, :, z_idx, 1]

    # Finite differences on discrete grid
    div_fd = np.gradient(u_field, 1.0, axis=0) + np.gradient(v_field, 1.0, axis=1)

    # Analytical autodiff divergence at same integer points
    div_analytical = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            div_analytical[i, j] = compute_divergence_at_point(field, float(i + 1), float(j + 1), float(z_idx + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax_fd = max(abs(div_fd.min()), abs(div_fd.max()))
    vmax_an = max(abs(div_analytical.min()), abs(div_analytical.max()))

    im0 = axes[0].imshow(div_fd.T, origin="lower", cmap="RdBu_r", vmin=-vmax_fd, vmax=vmax_fd)
    axes[0].set_title(f"Finite-difference divergence (max |div|={vmax_fd:.2e})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(div_analytical.T, origin="lower", cmap="RdBu_r", vmin=-vmax_an, vmax=vmax_an)
    axes[1].set_title(f"Autodiff divergence (max |div|={vmax_an:.2e})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].scatter(div_analytical.ravel(), div_fd.ravel(), alpha=0.5, s=10)
    axes[2].axhline(0, color="gray", ls="--", alpha=0.6)
    axes[2].axvline(0, color="gray", ls="--", alpha=0.6)
    axes[2].set_xlabel("autodiff divergence")
    axes[2].set_ylabel("finite-difference divergence")
    axes[2].set_title("Point-wise comparison")
    axes[2].grid(alpha=0.3)

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
        "Divergence: Analytical (Autodiff) vs Finite Difference\n"
        f"{subtitle}",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out = save_figure(fig, "divergence_comparison.png")
    print(f"Saved to: {out}")
    print(
        f"Analytical max |div|={np.abs(div_analytical).max():.2e}, "
        f"finite-diff max |div|={np.abs(div_fd).max():.2e}"
    )
    plt.show()

# %%
compare_analytical_vs_finite_diff(sigma=4.0, lengthscale=4.0, nu=1.5)
