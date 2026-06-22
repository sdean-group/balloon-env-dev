"""3D Navigation Arena with RFF GP Field Visualization."""

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


def run_3d_visualization_rff():
    """Run 3D navigation with RFF GP Field (divergence-free)."""
    
    print("=" * 70)
    print("3D NAVIGATION - RFF GP FIELD (Divergence-Free)")
    print("=" * 70)
    
    # Configuration: 3D grid
    config = GridConfig.create(n_x=50, n_y=50, n_z=20)
    d_max = 10
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {config.ndim}D")
    print(f"  Size: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"  Ambient axes: x, y (size {config.n_x} x {config.n_y})")
    print(f"  Controllable axis: z (size {config.n_z})")
    print(f"  Max displacement: {d_max}")
    
    # GP Field parameters
    sigma = 2.5
    lengthscale = 4.0
    nu = 2.5
    
    print(f"\nRFF GP Field parameters:")
    print(f"  sigma: {sigma} (amplitude)")
    print(f"  lengthscale: {lengthscale} (correlation)")
    print(f"  nu: {nu} (smoothness)")
    print(f"  Method: Streamfunction (divergence-free)")
    
    # Positions
    initial_position = GridPosition(15, 15, 10)
    target_position = GridPosition(35, 35, 16)
    vicinity_radius = 5.0
    
    print(f"\nNavigation task:")
    print(f"  Start: ({initial_position.i}, {initial_position.j}, {initial_position.k})")
    print(f"  Target: ({target_position.i}, {target_position.j}, {target_position.k})")
    print(f"  Vicinity radius: {vicinity_radius}")
    
    # Create RFF GP field (3D = streamfunction method for divergence-free field)
    field = SyntheticFlowField(
        config,
        sigma=sigma, lengthscale=lengthscale, nu=nu,
        num_features=500,
    )
    actor = GridActor(noise_std=0.5, scale=2.5)
    reward_fn = NavigationReward(
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        peak_reward=5.0,
        step_cost=0.1,
        proximity_scale=0.3,
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
        process_noise_std=0.5,
        obs_noise_std=0.5,
    )
    
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=1024,
        height=768,
        camera_eye={'x': 1.8, 'y': -1.8, 'z': 1.2},
        field=field,
        show_field=True  # Show GP mean displacement arrows
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
    
    pos = info['position']
    print(f"  Final position: ({pos.i}, {pos.j}, {pos.k})")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Target reached: {info['target_reached']}")
    print("-" * 70)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "3d")
    os.makedirs(output_dir, exist_ok=True)
    animated_html_path = os.path.join(output_dir, "viz_3d_rff_output_animated.html")
    
    renderer.save_animated_html(animated_html_path)
    print(f"\nSaved to: {animated_html_path}")
    
    env.close()


def run_3d_station_keeping_rff():
    """Run 3D station-keeping with RFF GP Field."""
    
    print("\n")
    print("=" * 70)
    print("3D STATION-KEEPING - RFF GP FIELD")
    print("=" * 70)
    
    # Configuration
    config = GridConfig.create(n_x=10, n_y=10, n_z=6)
    d_max = 2
    
    # Start at target for station-keeping
    target_position = GridPosition(5, 5, 3)
    initial_position = target_position
    vicinity_radius = 2.5
    
    print(f"\nStation-keeping task:")
    print(f"  Start/Target: ({target_position.i}, {target_position.j}, {target_position.k})")
    print(f"  Vicinity radius: {vicinity_radius}")
    print(f"  Agent tries to stay within vicinity despite GP field perturbations")
    
    field = SyntheticFlowField(
        config,
        sigma=1.0, lengthscale=3.0, nu=2.5,
        num_features=500,
    )
    actor = GridActor(noise_std=0.1)
    reward_fn = NavigationReward(
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        peak_reward=10.0,
        step_cost=0.2,
        proximity_scale=0.5,
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
        boundary_mode='clip',
        reward_fn=reward_fn,
        terminate_on_reach=False,
        process_noise_std=0.1,
        obs_noise_std=0.1,
    )
    
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=1024,
        height=768,
        field=field,
        show_field=True
    )
    
    env = GridEnvironment(
        arena=arena,
        max_steps=100,
        seed=123,
        renderer=renderer
    )
    
    # Run episode with "stay" action (action=1)
    print("\n" + "-" * 70)
    print("Running station-keeping episode (30 steps, action=stay)...")
    
    obs, info = env.reset(seed=123)
    
    total_reward = 0.0
    in_vicinity_count = 0
    
    for step in range(30):
        # Station-keeping: always try to stay
        action = 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Check if still in vicinity
        pos = info['position']
        dist = ((pos.i - target_position.i)**2 + 
                (pos.j - target_position.j)**2 + 
                (pos.k - target_position.k)**2) ** 0.5
        if dist <= vicinity_radius:
            in_vicinity_count += 1
    
    print(f"  Steps in vicinity: {in_vicinity_count}/30")
    print(f"  Total reward: {total_reward:.2f}")
    print("-" * 70)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "3d")
    os.makedirs(output_dir, exist_ok=True)
    animated_html_path = os.path.join(output_dir, "viz_3d_rff_station_keeping_animated.html")
    
    renderer.save_animated_html(animated_html_path)
    print(f"\nSaved to: {animated_html_path}")
    
    env.close()


if __name__ == "__main__":
    run_3d_visualization_rff()
    
    # Optionally run station-keeping demo
    response = input("\nRun station-keeping visualization? [y/N]: ")
    if response.lower() == 'y':
        run_3d_station_keeping_rff()
