"""2D Navigation Arena with RFF GP Field Visualization."""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
from src.env.field import SyntheticFlowField


def run_2d_visualization_rff():
    """Run 2D navigation with RFF GP Field."""
    
    print("=" * 70)
    print("2D NAVIGATION - RFF GP FIELD")
    print("=" * 70)
    
    # Configuration: 2D grid
    config = GridConfig.create(n_x=1000, n_y=1000)
    d_max = 100
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {config.ndim}D")
    print(f"  Size: {config.n_x} x {config.n_y}")
    print(f"  Max displacement: {d_max}")
    
    # GP Field parameters
    sigma = 20
    lengthscale = 50
    nu = 2.5
    
    print(f"\nRFF GP Field parameters:")
    print(f"  sigma: {sigma} (amplitude)")
    print(f"  lengthscale: {lengthscale} (correlation)")
    print(f"  nu: {nu} (smoothness)")
    
    # Positions
    initial_position = GridPosition(300, 300, None)
    target_position = GridPosition(700, 700, None)
    vicinity_radius = 50.0
    
    print(f"\nNavigation task:")
    print(f"  Start: ({initial_position.i}, {initial_position.j})")
    print(f"  Target: ({target_position.i}, {target_position.j})")
    
    # Create RFF GP field (2D = scalar field for single ambient axis)
    field = SyntheticFlowField(
        config,
        sigma=sigma, lengthscale=lengthscale, nu=nu,
        num_features=500,
    )
    actor = GridActor(noise_std=2, scale=50)
    reward_fn = NavigationReward(
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        peak_reward=10.0,
        step_cost=0.1,
        proximity_scale=0.05,
    )
    arena = NavigationArena(
        realized_field=field,
        observed_field=field,
        actor=actor,
        config=config,
        initial_position=initial_position,
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        max_displacement=d_max,
        boundary_mode='terminal',
        reward_fn=reward_fn,
        terminate_on_reach=False,
        process_noise_std=2,
        obs_noise_std=2,
    )
    
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=1000,
        height=1000,
        field=field,
        show_field=True
    )
    
    env = GridEnvironment(
        arena=arena,
        max_steps=100,
        seed=42,
        renderer=renderer
    )
    
    # Run episode
    print("\n" + "-" * 70)
    print("Running episode with random policy (50 steps)...")
    
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    
    print(f"  Final position: ({info['position'].i}, {info['position'].j})")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Target reached: {info['target_reached']}")
    print("-" * 70)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "2d")
    os.makedirs(output_dir, exist_ok=True)
    animated_html_path = os.path.join(output_dir, "viz_2d_rff_output_animated.html")
    
    renderer.save_animated_html(animated_html_path)
    print(f"\nSaved to: {animated_html_path}")
    
    env.close()


if __name__ == "__main__":
    run_2d_visualization_rff()
