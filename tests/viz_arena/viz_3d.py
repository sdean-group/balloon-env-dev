"""3D Navigation Arena Visual Verification.
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    ConstantDriftField,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)


def run_3d_visualization():
    """Run 3D navigation episode and display visualization."""
    
    print("=" * 70)
    print("3D NAVIGATION - VISUAL VERIFICATION")
    print("=" * 70)
    
    # Configuration: 3D grid
    config = GridConfig.create(n_x=12, n_y=12, n_z=8)
    d_max = 3
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {config.ndim}D")
    print(f"  Size: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"  Ambient axes: x, y (size {config.n_x} x {config.n_y})")
    print(f"  Controllable axis: z (size {config.n_z})")
    print(f"  Max displacement: {d_max}")
    
    # Positions
    initial_position = GridPosition(2, 2, 2)
    target_position = GridPosition(10, 10, 6)
    vicinity_radius = 2.0
    
    print(f"\nNavigation task:")
    print(f"  Start: ({initial_position.i}, {initial_position.j}, {initial_position.k})")
    print(f"  Target: ({target_position.i}, {target_position.j}, {target_position.k})")
    print(f"  Vicinity radius: {vicinity_radius}")
    
    # Create components
    field = ConstantDriftField(config, drift=[0.0, 0.0])
    actor = GridActor(noise_std=0.1)
    reward_fn = NavigationReward(
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        peak_reward=10.0,
        step_cost=0.1,
        proximity_scale=0.1,
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
    )
    
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=1024,
        height=768,
        camera_eye={'x': 1.8, 'y': -1.8, 'z': 1.2},  # Good 3D viewing angle
        field=field,
        show_field=True  # Show field mean displacement arrows (zero for SimpleField)
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
    
    # Save and display
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "3d")
    os.makedirs(output_dir, exist_ok=True)
    animated_html_path = os.path.join(output_dir, "viz_3d_output_animated.html")
    gif_path = os.path.join(output_dir, "viz_3d_output.gif")
    mp4_path = os.path.join(output_dir, "viz_3d_output.mp4")
    
    renderer.save_animated_html(animated_html_path)
    #renderer.save_gif(gif_path)
    #renderer.save_mp4(mp4_path)
    
    # try:
    #     env.render(mode='human')
    #     print("\nVisualization displayed in browser.")
    # except Exception as e:
    #     print(f"\nCould not open browser: {e}")
    #     print(f"Open the HTML file manually: {animated_html_path}")
    
    env.close()


def run_3d_station_keeping():
    """Run 3D station-keeping scenario (start at target)."""
    
    print("\n")
    print("=" * 70)
    print("3D STATION-KEEPING - VISUAL VERIFICATION")
    print("=" * 70)
    
    # Configuration
    config = GridConfig.create(n_x=10, n_y=10, n_z=6)
    d_max = 1
    
    # Start at target for station-keeping
    target_position = GridPosition(5, 5, 3)
    initial_position = target_position
    vicinity_radius = 2.5
    
    print(f"\nStation-keeping task:")
    print(f"  Start/Target: ({target_position.i}, {target_position.j}, {target_position.k})")
    print(f"  Vicinity radius: {vicinity_radius}")
    print(f"  Agent tries to stay within vicinity despite field perturbations")
    
    field = ConstantDriftField(config, drift=[0.0, 0.0])
    actor = GridActor(noise_std=0.1)
    reward_fn = NavigationReward(
        target_position=target_position,
        vicinity_radius=vicinity_radius,
        peak_reward=10.0,
        step_cost=0.2,
        proximity_scale=0.1,
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
    )
    
    renderer = NavigationRenderer(
        config=config,
        show_grid_points=True,
        width=1024,
        height=768,
        field=field,
        show_field=True  # Show field mean displacement arrows
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
    animated_html_path = os.path.join(output_dir, "viz_3d_station_keeping_animated.html")
    gif_path = os.path.join(output_dir, "viz_3d_station_keeping.gif")
    mp4_path = os.path.join(output_dir, "viz_3d_station_keeping.mp4")
    
    renderer.save_animated_html(animated_html_path)
    #renderer.save_gif(gif_path)
    #renderer.save_mp4(mp4_path)

    # try:
    #     env.render(mode='human')
    # except Exception as e:
    #     print(f"Open manually: {animated_html_path}")
    
    env.close()


if __name__ == "__main__":
    run_3d_visualization()
    
    # Optionally run station-keeping demo
    response = input("\nRun station-keeping visualization? [y/N]: ")
    if response.lower() == 'y':
        run_3d_station_keeping()
