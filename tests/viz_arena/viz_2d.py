"""2D Navigation Arena Visual Verification.
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


def run_2d_visualization():
    """Run 2D navigation episode and display visualization."""
    
    print("=" * 70)
    print("2D NAVIGATION - VISUAL VERIFICATION")
    print("=" * 70)
    
    # Configuration: 2D grid (no n_z)
    config = GridConfig.create(n_x=150, n_y=120)
    d_max = 10
    
    print(f"\nGrid configuration:")
    print(f"  Dimensions: {config.ndim}D")
    print(f"  Size: {config.n_x} x {config.n_y}")
    print(f"  Ambient axis: x (size {config.n_x})")
    print(f"  Controllable axis: y (size {config.n_y})")
    print(f"  Max displacement: {d_max}")
    
    # Positions
    initial_position = GridPosition(30, 30, None)
    target_position = GridPosition(120, 100, None)
    vicinity_radius = 20.0
    
    print(f"\nNavigation task:")
    print(f"  Start: ({initial_position.i}, {initial_position.j})")
    print(f"  Target: ({target_position.i}, {target_position.j})")
    print(f"  Vicinity radius: {vicinity_radius}")
    
    # Create components
    field = ConstantDriftField(config, drift=[0.0])
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
        width=900,
        height=700,
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
    print("Running episode with random policy (40 steps)...")
    
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    for step in range(40):
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
    
    # Save and display
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "2d")
    os.makedirs(output_dir, exist_ok=True)
    animated_html_path = os.path.join(output_dir, "viz_2d_output_animated.html")
    gif_path = os.path.join(output_dir, "viz_2d_output.gif")
    mp4_path = os.path.join(output_dir, "viz_2d_output.mp4")
    
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


if __name__ == "__main__":
    run_2d_visualization()
