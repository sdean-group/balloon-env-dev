# Current Branch From First Principles

This document explains the current `data-driven-gp` branch from the ground up. It is meant to be readable before a meeting even if you are not fully comfortable with Gaussian processes, weather data, or reinforcement learning yet.

## 1. What Problem Are We Solving?

We are building a balloon environment.

A balloon lives at some position:

```text
(x, y, z)
```

where:

- `x` is horizontal position east-west-ish in our grid.
- `y` is horizontal position north-south-ish in our grid.
- `z` is altitude/pressure-level index.

The balloon is pushed by wind.

The wind field is a function:

```text
position -> velocity
```

In code:

```python
velocity_at(GridPosition(x, y, z)) -> (u, v)
```

where:

- `u` moves the balloon in x.
- `v` moves the balloon in y.
- we currently do not model vertical wind velocity.

So one passive step is:

```text
x_next = x + u
y_next = y + v
z_next = z
```

If we allow control, the agent can choose altitude `z`, then the wind at that altitude moves it horizontally.

## 2. What Are The Main Things In This Branch?

This branch now has:

1. Synthetic wind fields.
2. Real ERA5-backed wind fields.
3. Data-driven GP-style wind fields.
4. Helmholtz vector-field versions.
5. Passive drift demos.
6. A cross-country navigation task.
7. A max-distance task.
8. A simple MPC baseline.
9. A PPO deep-policy baseline.

## 3. The Core Interface: `FlowField`

Main file:

```text
src/env/field/flow_field.py
```

Every wind field follows the same basic interface:

```python
reset(rng_key)
velocity_at(position)
velocity_field()
```

### `reset`

Prepares the field for an episode.

Examples:

- Synthetic GP samples a new random field.
- ERA5 chooses a time slice.
- Data-driven GP chooses fitted weights / mean field.

### `velocity_at`

This is the key method.

Given a continuous position, return wind:

```python
(u, v)
```

### `velocity_field`

Returns the whole wind field on grid points, mostly for visualization.

## 4. The Wind Field Types

### 4.1 Legacy Synthetic GP

Main file:

```text
src/env/field/synthetic.py
```

This was the original synthetic GP field from the repo.

It uses random Fourier features, or RFF.

Plain English:

```text
make a smooth random wind field by adding many random smooth waves
```

This is useful because it gives us a controllable fake wind field without downloading data.

However, the meeting feedback was that for vector fields like wind, we should use a more physically meaningful vector-field kernel. That led to the Helmholtz work below.

In the demo script, the old synthetic field is now available as:

```bash
--field legacy-synthetic
```

### 4.2 Helmholtz Synthetic GP

Main file:

```text
src/env/field/helmholtz.py
```

Class:

```python
HelmholtzSyntheticFlowField
```

This is now the synthetic field we should use.

In the demo script:

```bash
--field synthetic
```

now maps to the Helmholtz synthetic field.

You can also explicitly use:

```bash
--field helmholtz-synthetic
```

## 5. Why Helmholtz?

Wind is a vector field.

That means the output is not just one number. It is a direction and magnitude:

```text
(u, v)
```

A naive GP approach might model:

```text
u(x, y, z) separately
v(x, y, z) separately
```

That is easy, but physically weak. It treats the two wind components as mostly unrelated scalar functions.

The Helmholtz idea says a vector field can be decomposed into structured pieces:

1. **Divergence-free / rotational part**
   - Think swirling flow.
   - Horizontal air does not appear or disappear locally.

2. **Curl-free / potential part**
   - Think source/sink or expanding/contracting flow.

In this branch, `HelmholtzSyntheticFlowField` creates wind as a combination of:

```text
curl-free component + divergence-free component
```

This gives us a better synthetic test bed than independent scalar GPs.

## 6. How The Helmholtz Synthetic Field Works

File:

```text
src/env/field/helmholtz.py
```

Class:

```python
HelmholtzSyntheticFlowField
```

It samples random Fourier features.

Conceptually:

```text
random smooth waves -> scalar potentials -> vector wind field
```

It has two sets of random weights:

```python
self._div_weights
self._curl_weights
```

These control:

- curl-free component
- divergence-free component

The important method is:

```python
velocity_at_point(x, y, z)
```

It evaluates the vector field at a continuous position.

Then:

```python
velocity_at(GridPosition(...))
```

returns ordinary Python floats for the simulator.

## 7. Real ERA5 With Linear Interpolation

Main files:

```text
experiments/field_estimation/scripts/fetch_era5.py
src/env/field/era5_data.py
src/env/field/reanalysis.py
```

### `fetch_era5.py`

Downloads or builds wind caches.

It supports:

- `cds`: real pressure-level ERA5 through Copernicus.
- `openmeteo`: keyless ERA5-derived surface winds.
- `demo`: synthetic cache for testing.

### `era5_data.py`

Loads `.npz` wind caches and validates shape.

For 3D data, the shape is:

```text
(T, n_x, n_y, n_z, 2)
```

where:

- `T` is number of time slices.
- `n_x`, `n_y`, `n_z` are grid dimensions.
- `2` is `(u, v)`.

### `reanalysis.py`

Class:

```python
ReanalysisFlowField
```

This directly uses cached wind data.

Given a continuous position, it interpolates between nearby grid points.

Before this branch, it selected one time slice.

Now it supports fractional time interpolation:

```python
field.velocity_at_time(position, 1.5)
```

That means:

```text
halfway between time slice 1 and time slice 2
```

This is the first step toward treating weather as changing over time.

## 8. Data-Driven GP: What Happened?

Main file:

```text
src/env/field/data_driven.py
```

Class:

```python
DataDrivenFlowField
```

This was our first data-driven GP-like model.

It trains from wind samples, usually from ERA5 cache data.

It learns:

```text
(x, y, z) -> (u, v)
```

using random Fourier features and Bayesian/ridge-style linear regression.

But after the meeting, we clarified something important:

```text
This is a learned mean-field baseline, not the final uncertainty-aware data-driven GP.
```

Why?

Because the meeting concern was:

```text
RFF may not approximate the posterior variance from data well enough.
```

So we should not claim that `DataDrivenFlowField` is the final data-driven probabilistic model.

It is still useful because:

- it learns a smooth approximation from ERA5 samples
- it can predict continuously between grid points
- it gives us a baseline

But it is not the final answer for uncertainty.

## 9. Helmholtz Data-Driven GP

Main file:

```text
src/env/field/helmholtz.py
```

Class:

```python
HelmholtzDataDrivenFlowField
```

This is a data-driven model that uses Helmholtz vector features.

Instead of learning `u` and `v` as separate scalar functions, it uses vector-valued basis functions derived from Helmholtz structure.

This is closer to the meeting recommendation:

```text
Use Helmholtz kernel for both synthetic and data-driven.
```

Important caveat:

This is still not a full exact GP posterior. It is a structured mean-field baseline.

So the correct framing is:

```text
We now have a Helmholtz-structured data-driven baseline, but the full data-conditioned posterior problem is still future work.
```

## 10. The Demo Script

Main file:

```text
experiments/viz_passive_drift.py
```

This script creates passive drift demos.

Supported fields:

```bash
--field synthetic
--field legacy-synthetic
--field helmholtz-synthetic
--field era5
--field data-driven-gp
--field helmholtz-data-driven-gp
--field all
```

Important:

```bash
--field synthetic
```

now uses Helmholtz synthetic.

The old version is:

```bash
--field legacy-synthetic
```

### Interactive top-down demos

The top-down demos are simple browser HTML files.

They support:

- click to deploy
- type x/y/z
- pause
- resume
- reset
- speed slider

They use continuous interpolation while the balloon moves.

### Cross-section demos

Cross-section demos are in:

```text
experiments/output/passive_era5_ithaca/y_cross_section/
```

These show x-z slices at a fixed y level.

Important limitation:

Our wind model has horizontal velocity `(u, v)`, not vertical velocity. So an x-z cross-section cannot show real vertical arrows unless we add vertical wind.

## 11. Current Demo Files

### Passive direction-variation demos

```bash
open experiments/output/passive_north_america_surface/passive_drift_synthetic.html
open experiments/output/passive_north_america_surface/passive_drift_era5.html
open experiments/output/passive_north_america_surface/passive_drift_data_driven_gp.html
```

These use a larger North America surface-wind cache, so the arrows have more varied directions than the small Ithaca cache.

### Cross-section demos

```bash
open experiments/output/passive_era5_ithaca/y_cross_section/passive_drift_synthetic.html
open experiments/output/passive_era5_ithaca/y_cross_section/passive_drift_era5.html
open experiments/output/passive_era5_ithaca/y_cross_section/passive_drift_data_driven_gp.html
```

### Helmholtz smoke demos

```bash
open experiments/output/helmholtz_smoke/passive_drift_helmholtz_synthetic.html
open experiments/output/helmholtz_smoke/passive_drift_helmholtz_data_driven_gp.html
```

## 12. Navigation Tasks

We added two tasks.

### Task 1: Cross-country navigation

Goal:

```text
start at A, reach target B
```

The controller can choose altitude.

Wind moves the balloon horizontally.

### Task 2: Max distance

Goal:

```text
get as far away from the starting point as possible
```

This is simpler than A-to-B navigation.

It is useful because it rewards exploiting wind instead of targeting a specific destination.

## 13. MPC Baseline

Main file:

```text
experiments/cross_country_navigation_mpc.py
```

MPC means Model Predictive Control.

The idea:

```text
try possible actions in simulation, pick the one that looks best
```

Here the action is altitude choice.

At every step:

1. Try candidate altitude levels.
2. Simulate forward for a short horizon.
3. Score each rollout.
4. Pick the altitude with the best score.
5. Move one actual step.
6. Repeat.

For cross-country:

```text
score = negative distance to target
```

For max-distance:

```text
score = distance from start
```

### Run MPC cross-country

```bash
.pixi/envs/default/bin/python experiments/cross_country_navigation_mpc.py \
  --task cross-country \
  --data data/era5_north_america_surface_3d.npz \
  --start 50 35 1 \
  --target 20 20 \
  --time-index 15 \
  --time-delta 0.25 \
  --steps 40 \
  --horizon 6 \
  --scale 0.75 \
  --output experiments/output/cross_country_mpc_north_america.html
```

### Run MPC max-distance

```bash
.pixi/envs/default/bin/python experiments/cross_country_navigation_mpc.py \
  --task max-distance \
  --data data/era5_north_america_surface_3d.npz \
  --start 50 35 1 \
  --time-index 15 \
  --time-delta 0.25 \
  --steps 40 \
  --horizon 6 \
  --scale 0.75 \
  --output experiments/output/max_distance_mpc_north_america.html
```

Open outputs:

```bash
open experiments/output/cross_country_mpc_north_america.html
open experiments/output/max_distance_mpc_north_america.html
```

## 14. PPO Deep Policy Baseline

Main file:

```text
experiments/navigation_ppo.py
```

PPO means Proximal Policy Optimization.

It is a standard reinforcement learning algorithm.

In our setup:

- observation = balloon state + target + time progress
- action = choose altitude level
- reward depends on task

For cross-country:

```text
reward = progress toward target - small step cost
```

For max-distance:

```text
reward = increase in distance from start - small step cost
```

The policy is a small neural network:

```text
MLP actor-critic
```

It is implemented with:

- Flax
- Optax
- JAX

No new dependency stack was added.

### PPO caveat

This PPO implementation is a first baseline, not a tuned final RL solution.

It runs end-to-end and produces a policy rollout, but beating MPC will require more tuning and likely better observations/tasks.

### Run PPO

```bash
.pixi/envs/default/bin/python experiments/navigation_ppo.py \
  --task cross-country \
  --data data/era5_ithaca_3d.npz \
  --start 8 20 4 \
  --target 35 35 \
  --time-index 0 \
  --time-delta 0.25 \
  --steps 60 \
  --scale 0.03 \
  --updates 15 \
  --episodes-per-update 4 \
  --epochs 2 \
  --altitude-candidates 7 \
  --output experiments/output/navigation_ppo_ithaca.html
```

Open:

```bash
open experiments/output/navigation_ppo_ithaca.html
```

## 15. What Was Pushed Most Recently?

Latest pushed commit:

```text
4889623 Add PPO navigation baseline and Helmholtz synthetic default
```

It added:

- Helmholtz synthetic as default `synthetic` in demo
- `legacy-synthetic` option for old synthetic field
- MPC support for both cross-country and max-distance
- PPO baseline
- PPO output demo
- max-distance MPC output demo

Previous pushed commit:

```text
59b81db Add Helmholtz fields and navigation MPC baseline
```

It added:

- Helmholtz synthetic/data-driven field classes
- ERA5 time interpolation
- cross-country MPC baseline
- docs
- tests
- selected demo outputs

## 16. Tests

Last full field/arena test run:

```text
502 passed, 1 skipped
```

The skipped test is an optional ERA5 golden snapshot test.

Also checked:

```bash
python -m py_compile experiments/cross_country_navigation_mpc.py experiments/navigation_ppo.py experiments/viz_passive_drift.py
```

and smoke-ran:

- MPC cross-country
- MPC max-distance
- PPO smoke
- PPO 7-altitude Ithaca run

## 17. What To Say In A Meeting

Short version:

> We made Helmholtz synthetic the default synthetic test bed because wind is a vector field and Helmholtz structure is more physically meaningful than independent scalar GPs. We also added a Helmholtz data-driven baseline, but we are treating the current data-driven models as mean-field baselines rather than full posterior uncertainty models. ERA5 now supports fractional time interpolation. For tasks, we added cross-country navigation and max-distance, implemented first-order MPC, and added a PPO actor-critic baseline.

Even shorter:

> Synthetic is now Helmholtz. Data-driven is still a baseline, not the final posterior model. ERA5 has time interpolation. We have MPC and PPO baselines for navigation tasks.

## 18. Honest Current Limitations

1. The data-driven GP is not a full exact GP posterior.
2. PPO is implemented but not heavily tuned.
3. The North America surface cache has only one vertical level, so altitude control is trivial there.
4. The Ithaca pressure-level cache has real altitude levels but a small geographic region, so wind direction can look coherent.
5. Cross-sections do not show vertical wind because vertical velocity is not modeled.
6. Larger ERA5 pressure-level downloads still need better data-system work.

## 19. Best Next Steps

1. Use Helmholtz synthetic for controlled algorithm experiments.
2. Use ERA5 interpolation with time for real-data baseline.
3. Compare MPC vs PPO on the same task.
4. Improve data loading for larger pressure-level ERA5 regions.
5. Revisit data-driven uncertainty with a better posterior/conditioning approach.

