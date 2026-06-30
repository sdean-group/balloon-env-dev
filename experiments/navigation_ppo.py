"""PPO baseline for balloon navigation tasks.

This is intentionally small and self-contained: a Flax MLP policy chooses one of
the discrete altitude levels, then the wind field moves the balloon. It supports
the same two tasks as the MPC script:

- cross-country: get from start to target
- max-distance: get as far from start as possible
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import GridConfig, GridPosition, ReanalysisFlowField
from src.env.field.era5_data import load_era5


class ActorCritic(nn.Module):
    n_actions: int
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(self.hidden)(x))
        x = nn.tanh(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.n_actions)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value


def _config_from_cache(path: str) -> GridConfig:
    winds = load_era5(path).winds
    return GridConfig.create(*winds.shape[1:-1])


def _clip(x: float, y: float, config: GridConfig) -> tuple[float, float]:
    return float(np.clip(x, 1.0, config.n_x)), float(np.clip(y, 1.0, config.n_y))


def _obs(x, y, z, step, config, args):
    target_x, target_y = args.target
    return np.array(
        [
            x / config.n_x,
            y / config.n_y,
            z / config.n_z,
            target_x / config.n_x,
            target_y / config.n_y,
            step / max(args.steps, 1),
        ],
        dtype=np.float32,
    )


def _distance(x, y, point):
    return float(np.hypot(x - point[0], y - point[1]))


def _reward(prev_x, prev_y, x, y, args):
    if args.task == "cross-country":
        target = (float(args.target[0]), float(args.target[1]))
        return _distance(prev_x, prev_y, target) - _distance(x, y, target) - 0.01
    start = (float(args.start[0]), float(args.start[1]))
    return _distance(x, y, start) - _distance(prev_x, prev_y, start) - 0.005


def _make_field(config, args):
    field = ReanalysisFlowField(
        config,
        args.data,
        scale=args.scale,
        slice_mode="fixed",
        fixed_index=float(args.time_index),
    )
    field.reset(jax.random.PRNGKey(args.seed))
    return field


def _run_episode(params, model, field, config, args, rng, greedy=False):
    x, y, z = map(float, args.start)
    altitude_levels = np.linspace(1.0, float(config.n_z), args.altitude_candidates)
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    path = []

    for step in range(args.steps):
        obs = _obs(x, y, z, step, config, args)
        logits, value = model.apply(params, jnp.asarray(obs[None, :]))
        logits_np = np.asarray(logits[0])
        value_np = float(np.asarray(value[0]))
        if greedy:
            action = int(np.argmax(logits_np))
        else:
            rng, subkey = jax.random.split(rng)
            action = int(jax.random.categorical(subkey, logits[0]))
        log_prob = float(logits_np[action] - logsumexp(logits_np))
        z = float(altitude_levels[action])
        time_index = min(float(args.time_index) + step * args.time_delta, field._T - 1)
        u, v = field.velocity_at_time(GridPosition(x, y, z), time_index)
        prev_x, prev_y = x, y
        x, y = _clip(x + float(u), y + float(v), config)
        reward = _reward(prev_x, prev_y, x, y, args)

        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value_np)
        path.append(dict(step=step, x=x, y=y, z=z, reward=reward))

        if args.task == "cross-country" and _distance(x, y, args.target) <= args.target_radius:
            break

    final_obs = _obs(x, y, z, len(path), config, args)
    _, final_value = model.apply(params, jnp.asarray(final_obs[None, :]))
    return dict(
        obs=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        bootstrap=float(np.asarray(final_value[0])),
        path=path,
    ), rng


def _advantages(rewards, values, bootstrap, gamma, lam):
    adv = np.zeros_like(rewards)
    lastgaelam = 0.0
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        adv[t] = lastgaelam
        next_value = values[t]
    returns = adv + values
    return adv, returns


def _loss_fn(params, model, obs, actions, old_log_probs, advantages, returns, clip_eps, vf_coef, ent_coef):
    logits, values = model.apply(params, obs)
    log_probs_all = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[:, None], axis=1).squeeze(1)
    ratio = jnp.exp(log_probs - old_log_probs)
    policy_loss = -jnp.mean(
        jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages,
        )
    )
    value_loss = jnp.mean((returns - values) ** 2)
    entropy = -jnp.mean(jnp.sum(jnp.exp(log_probs_all) * log_probs_all, axis=1))
    return policy_loss + vf_coef * value_loss - ent_coef * entropy


def _train(args):
    config = _config_from_cache(args.data)
    if config.ndim != 3:
        raise ValueError("PPO demo expects a 3D wind cache")
    field = _make_field(config, args)
    model = ActorCritic(n_actions=args.altitude_candidates, hidden=args.hidden)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, jnp.zeros((1, 6), dtype=jnp.float32))
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(_loss_fn)(
            params,
            model,
            batch["obs"],
            batch["actions"],
            batch["old_log_probs"],
            batch["advantages"],
            batch["returns"],
            args.clip_eps,
            args.value_coef,
            args.entropy_coef,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for iteration in range(args.updates):
        episodes = []
        for _ in range(args.episodes_per_update):
            episode, rng = _run_episode(params, model, field, config, args, rng)
            episodes.append(episode)

        obs, actions, old_log_probs, advantages, returns = [], [], [], [], []
        for episode in episodes:
            adv, ret = _advantages(
                episode["rewards"],
                episode["values"],
                episode["bootstrap"],
                args.gamma,
                args.gae_lambda,
            )
            obs.append(episode["obs"])
            actions.append(episode["actions"])
            old_log_probs.append(episode["log_probs"])
            advantages.append(adv)
            returns.append(ret)

        batch = dict(
            obs=jnp.asarray(np.concatenate(obs)),
            actions=jnp.asarray(np.concatenate(actions)),
            old_log_probs=jnp.asarray(np.concatenate(old_log_probs)),
            advantages=jnp.asarray(np.concatenate(advantages)),
            returns=jnp.asarray(np.concatenate(returns)),
        )
        adv = np.asarray(batch["advantages"])
        batch["advantages"] = jnp.asarray((adv - adv.mean()) / (adv.std() + 1e-8))
        for _ in range(args.epochs):
            params, opt_state, loss = update(params, opt_state, batch)
        if (iteration + 1) % max(args.updates // 5, 1) == 0:
            mean_return = np.mean([ep["rewards"].sum() for ep in episodes])
            print(f"update {iteration + 1:04d}: mean_return={mean_return:.3f} loss={float(loss):.3f}")

    eval_episode, _ = _run_episode(params, model, field, config, args, rng, greedy=True)
    return config, eval_episode["path"]


def _write_html(config, path, args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        shape=list(config.shape),
        start=list(args.start),
        target=list(args.target),
        task=args.task,
        path=path,
        title=f"{args.task}: PPO altitude policy",
    )
    output.write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{payload["title"]}</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><div id="plot" style="width:100%;height:92vh;"></div>
<script>
const data = {json.dumps(payload)};
const xs = data.path.map(p => p.x);
const ys = data.path.map(p => p.y);
const zs = data.path.map(p => p.z);
Plotly.newPlot("plot", [
  {{x: xs, y: ys, mode: "lines+markers", line: {{color: "#7c3aed", width: 4}}, marker: {{size: 6, color: zs, colorscale: "Viridis", colorbar: {{title: "z"}}}}, name: "PPO path"}},
  {{x: [data.start[0]], y: [data.start[1]], mode: "markers", marker: {{size: 14, color: "#f59e0b"}}, name: "start"}},
  ...(data.task === "cross-country" ? [{{x: [data.target[0]], y: [data.target[1]], mode: "markers", marker: {{size: 16, color: "#dc2626", symbol: "x"}}, name: "target"}}] : [])
], {{
  title: data.title,
  paper_bgcolor: "#f8fafc",
  plot_bgcolor: "#eef3f8",
  xaxis: {{title: "x", range: [0.5, data.shape[0] + 0.5]}},
  yaxis: {{title: "y", range: [0.5, data.shape[1] + 0.5], scaleanchor: "x"}},
  margin: {{l: 60, r: 20, t: 60, b: 50}}
}});
</script></body></html>
""",
        encoding="utf-8",
    )
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cross-country", "max-distance"], default="cross-country")
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", type=float, nargs=3, default=[50.0, 35.0, 1.0])
    parser.add_argument("--target", type=float, nargs=2, default=[20.0, 20.0])
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--time-index", type=float, default=15.0)
    parser.add_argument("--time-delta", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--altitude-candidates", type=int, default=7)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/output/navigation_ppo.html")
    args = parser.parse_args()

    config, path = _train(args)
    _write_html(config, path, args)
    final = path[-1]
    print(f"final=({final['x']:.2f}, {final['y']:.2f}, z={final['z']:.2f}) steps={final['step']}")


if __name__ == "__main__":
    main()

