# GP Wind Fields Explainer

This is a practical guide for explaining the synthetic GP and data-driven GP code in this repo. It assumes you know the project goal: we want wind fields that a passive balloon can drift through. It does not assume you already know Gaussian processes.

## The One-Minute Version

We currently have three wind-field options:

1. **Synthetic GP**
   - File: `src/env/field/synthetic.py`
   - Makes up a smooth wind field from scratch.
   - Uses no real data.
   - Good for controlled synthetic experiments.

2. **Real ERA5 with linear interpolation**
   - Files: `experiments/field_estimation/scripts/fetch_era5.py`, `src/env/field/era5_data.py`, `src/env/field/reanalysis.py`
   - Downloads real ERA5 wind data, stores it on our grid, and interpolates between grid points.
   - This is the direct real-data baseline.

3. **Data-driven GP**
   - File: `src/env/field/data_driven.py`
   - Learns a smooth GP-like function from real data samples, currently from the ERA5 cache.
   - It is not just interpolating. It fits a model that predicts wind smoothly at any point.

The passive drift demo is:

`experiments/viz_passive_drift.py`

That script can visualize a balloon drifting through any of the three fields.

## Core Vocabulary

### Wind field

A wind field is a function:

```text
position -> wind velocity
```

In code, that means:

```python
velocity_at(position) -> (u, v)
```

where:

- `u` is wind movement in the x direction.
- `v` is wind movement in the y direction.
- In 3D, the input position is `(x, y, z)`.
- In our demos, `z` is usually fixed, so the balloon drifts horizontally through one altitude/pressure level.

### Passive balloon

Passive means the balloon does not control itself. It just goes wherever the wind pushes it.

The update rule is basically:

```text
new_x = old_x + u
new_y = old_y + v
z stays fixed
```

The demo clips or wraps positions at the grid boundary depending on the chosen boundary mode.

### Gaussian process

A Gaussian process, or GP, is a way to describe a smooth random function.

Instead of saying "the wind at every point is independent random noise", a GP says nearby points should have related wind. So if the wind is strong eastward at one location, nearby locations are likely to also have similar eastward wind.

For this project, the GP idea is useful because real wind fields are spatially smooth. They do not usually jump wildly from one grid point to the next.

### Kernel

A kernel says how related two points are.

Simple intuition:

```text
nearby points -> highly related
far points -> less related
```

The most important parameter is `lengthscale`.

- Small `lengthscale`: wind changes quickly over space.
- Large `lengthscale`: wind changes slowly and smoothly.

### Random Fourier features

Exact GP inference can be expensive. Random Fourier features are an approximation trick.

Instead of directly storing a huge GP covariance matrix, we build the field from many random smooth waves:

```text
smooth wave 1 + smooth wave 2 + ... + smooth wave L
```

`num_features` controls how many waves/features we use.

- More features: better approximation, slower.
- Fewer features: faster, rougher approximation.

In the code, these are called:

- `_omegas`: random wave frequencies
- `_phases`: random wave shifts
- `_weights`: how much each wave contributes

## Shared Interface

All field classes follow the same `FlowField` shape:

`src/env/field/flow_field.py`

The important methods are:

```python
reset(rng_key)
velocity_at(position)
velocity_field()
```

### `reset(rng_key)`

Sets up the field for one episode.

For synthetic GP, this draws a new random wind field.

For ERA5 interpolation, this picks a time slice.

For data-driven GP, this chooses either the learned mean field or a sampled posterior field.

### `velocity_at(position)`

This is the key method used by the simulator.

Given a position, return the wind velocity there.

### `velocity_field()`

Returns the whole field on grid points. This is mostly useful for visualization.

The balloon drift itself uses `velocity_at`.

## Synthetic GP

Main file:

`src/env/field/synthetic.py`

Main class:

```python
SyntheticFlowField
```

### What it does

Synthetic GP creates a fake but smooth wind field. There is no data loading. There is no ERA5. It samples a random function from a GP-like prior.

The phrase "prior" means: before seeing any data, this is the kind of wind field we think is plausible.

### Why we use it

Synthetic GP is useful when we want a controlled test environment:

- no download needed
- no real-world dataset dependency
- easy to generate many random wind fields
- smooth enough to look like a real flow field

### Constructor

```python
SyntheticFlowField(
    config,
    sigma=1.0,
    lengthscale=1.0,
    nu=2.5,
    num_features=500,
)
```

Important arguments:

- `config`: grid size, like `GridConfig.create(40, 40, 7)`.
- `sigma`: overall wind strength.
- `lengthscale`: how smooth the field is over space.
- `nu`: Matern kernel smoothness. Higher means smoother.
- `num_features`: number of random Fourier features.

### Important internal variables

```python
self._omegas
self._phases
self._weights
```

These define the random Fourier feature field.

Think of them like this:

- `_omegas`: directions/frequencies of random waves
- `_phases`: where each wave starts
- `_weights`: how much each wave matters

Together they define one smooth random wind field.

### Reset flow

The key method is:

```python
reset(self, rng_key)
```

It does three things:

1. Samples random frequencies with `_sample_matern_frequencies`.
2. Samples random phases.
3. Samples random weights.

Then it calls:

```python
_precompute_field()
```

That precomputes wind values at every integer grid point for visualization.

### Continuous velocity lookup

The important lookup method is:

```python
velocity_at_point(x, y, z)
```

This computes the synthetic GP wind directly at a continuous point.

Then:

```python
velocity_at(position)
```

wraps that method for the repo's `GridPosition` object.

This matters because the balloon can be at fractional positions like:

```text
(12.3, 19.7, 4.0)
```

not just integer grid points.

### 2D vs 3D behavior

The code has two modes:

#### 2D synthetic mode

In 2D, the field returns only one component:

```python
(u, None)
```

This matches older repo conventions where 2D means one ambient motion component.

#### 3D synthetic mode

In 3D, it returns:

```python
(u, v)
```

The input is `(x, y, z)`, but the output is horizontal wind.

The balloon moves horizontally:

```text
x changes by u
y changes by v
z stays fixed in our passive demo
```

### Streamfunction idea

In 3D mode, synthetic GP uses a streamfunction.

You do not need to explain the math deeply. The practical explanation is:

The code samples one smooth scalar function, then takes derivatives of it to make a 2D horizontal velocity field. This gives a nice property: the horizontal flow is divergence-free.

Plain English:

The synthetic wind does not randomly create or destroy horizontal air mass in the x-y plane. It behaves more like swirling/incompressible flow.

Relevant code:

```python
u = scale * (sin_theta @ (self._weights * self._omega_y))
v = -scale * (sin_theta @ (self._weights * self._omega_x))
```

### What to say in a meeting

You can say:

"The synthetic GP is our fully artificial wind source. It samples a smooth random field using random Fourier features. The lengthscale controls how quickly wind changes over space, sigma controls wind strength, and num_features controls approximation quality. In 3D mode it uses a streamfunction construction so the horizontal velocity field is smooth and divergence-free. The simulator calls `velocity_at` at the balloon's continuous position to get the wind velocity."

## Data-Driven GP

Main file:

`src/env/field/data_driven.py`

Main class:

```python
DataDrivenFlowField
```

### What it does

Data-driven GP learns a smooth wind function from data.

Currently the data source is usually the ERA5 cache.

So the pipeline is:

```text
ERA5 cache -> training positions and wind velocities -> fit GP-like model -> predict wind anywhere
```

This is different from real ERA5 interpolation.

### Difference from ERA5 interpolation

ERA5 interpolation:

```text
use nearby ERA5 grid values directly
```

Data-driven GP:

```text
fit a smooth model to ERA5 samples, then use the model's prediction
```

ERA5 interpolation is local and geometric.

Data-driven GP is a learned smooth approximation.

### Constructor

```python
DataDrivenFlowField(
    config,
    positions,
    velocities,
    num_features=256,
    lengthscale=8.0,
    noise_std=0.1,
    feature_seed=0,
    sample_posterior=False,
)
```

Important arguments:

- `positions`: training input coordinates, shape `(n, ndim)`.
- `velocities`: training wind outputs, shape `(n, 1)` for 2D or `(n, 2)` for 3D.
- `num_features`: number of random Fourier features.
- `lengthscale`: smoothness scale.
- `noise_std`: how much noise/regularization we assume in training data.
- `sample_posterior`: whether to sample a plausible field instead of using the mean learned field.

### ERA5 helper constructor

Most of the time we do not manually pass positions and velocities. We call:

```python
DataDrivenFlowField.from_era5_cache(...)
```

This method:

1. Loads the ERA5 `.npz` cache using `load_era5`.
2. Selects one time slice using `time_index`.
3. Builds training coordinates for every selected grid point.
4. Pulls wind velocities from the cache.
5. Optionally subsamples using `training_stride` or `max_training_points`.
6. Fits `DataDrivenFlowField`.

### Shapes to understand

For 3D ERA5:

```text
winds shape = (T, n_x, n_y, n_z, 2)
```

Meaning:

- `T`: number of time slices
- `n_x`: x grid size
- `n_y`: y grid size
- `n_z`: vertical/pressure levels
- `2`: horizontal wind components `(u, v)`

For our real Ithaca ERA5 cache:

```text
(4, 40, 40, 7, 2)
```

That means:

- 4 time slices
- 40 by 40 horizontal grid
- 7 vertical levels
- 2 wind components

### Training step, plain English

The model is trying to learn:

```text
(x, y, z) -> (u, v)
```

For each ERA5 grid point, we know:

```text
position = (x, y, z)
velocity = (u, v)
```

The GP model learns weights over random smooth features so that its predictions match the training velocities.

### Feature construction

The method:

```python
_features(positions)
```

turns raw positions into random Fourier features:

```python
phase = positions @ self._omegas.T + self._phases
return sqrt(2 / num_features) * cos(phase)
```

Plain English:

Every position gets converted into a vector of smooth wave values. The model learns how to combine those waves to predict wind.

### Standardization

The code standardizes the target wind:

```python
self._target_mean = y.mean(axis=0)
self._target_std = y.std(axis=0)
y_standardized = (y - self._target_mean) / self._target_std
```

Plain English:

Before fitting, it rescales wind values so the model sees something numerically easier. After prediction, it converts back to the original scale.

This prevents one component or large wind values from making the solve unstable.

### Bayesian linear regression part

After converting positions into features, the code fits weights.

Important code:

```python
phi = self._features(x)
regularizer = self.noise_std**2
precision = phi.T @ phi + regularizer * np.eye(self.num_features)
```

Plain English:

- `phi` is the training feature matrix.
- The model solves for weights that map features to wind.
- `noise_std` adds regularization so the fit does not become numerically unstable or overfit too hard.

The code uses Cholesky factorization:

```python
self._precision_cholesky = np.linalg.cholesky(precision)
```

Plain English:

Cholesky is a stable way to solve the linear algebra problem.

### Posterior mean vs posterior sample

During reset:

```python
reset(self, rng_key)
```

If:

```python
sample_posterior=False
```

then it uses the learned average/best field:

```python
self._weights = self._posterior_mean.copy()
```

If:

```python
sample_posterior=True
```

then it samples a plausible alternative field:

```python
self._weights = self._posterior_mean + perturbation
```

Plain English:

The posterior mean is the model's best guess. A posterior sample is one possible wind field consistent with the data and the model uncertainty.

### Prediction

The important method is:

```python
velocity_at(position)
```

It converts the position into features, applies the learned weights, then returns `(u, v)`.

Internally:

```python
_predict(positions)
```

does:

```python
features @ weights
```

then unstandardizes the answer.

### Training error

The class stores:

```python
self.training_rmse
```

RMSE means root mean squared error. It measures how close the model predictions are to the training wind values.

Small RMSE means the model fits the training data well.

But be careful: low training RMSE does not automatically mean it generalizes well to totally new data. It just means it matched the training slice.

### What to say in a meeting

You can say:

"The data-driven GP uses the same broad idea of smooth functions, but instead of sampling a field from scratch, it fits a smooth function to observed wind samples. The current helper trains from an ERA5 cache: it builds coordinate/velocity pairs from one time slice, maps coordinates into random Fourier features, and solves a Bayesian linear regression problem. Then `velocity_at` predicts wind at continuous positions. This gives us a learned smooth approximation to the ERA5 field rather than direct linear interpolation."

## Real ERA5 With Linear Interpolation

You asked mostly about synthetic GP and data-driven GP, but the real interpolation baseline is useful for contrast.

Files:

- `experiments/field_estimation/scripts/fetch_era5.py`
- `src/env/field/era5_data.py`
- `src/env/field/reanalysis.py`

### `fetch_era5.py`

Downloads ERA5 from Copernicus and converts it to the repo's grid cache.

It handles pressure levels and metadata. The result is a `.npz` file with:

```text
winds
meta
```

### `era5_data.py`

Loads and validates the `.npz` cache.

It checks that:

- the wind array shape makes sense
- the number of wind components matches 2D or 3D
- there are no NaN/Inf values

### `reanalysis.py`

Implements:

```python
ReanalysisFlowField
```

This does linear interpolation using:

```python
RegularGridInterpolator
```

Plain English:

If the balloon is between ERA5 grid points, it estimates the wind by blending nearby grid values.

## Demo Script

Main file:

`experiments/viz_passive_drift.py`

This is the easiest entry point for showing the work.

### What it supports

Fields:

```text
--field synthetic
--field era5
--field data-driven-gp
--field all
```

Views:

```text
--view topdown
--view y-cross-section
```

The top-down view shows x-y motion at a fixed z level.

The y-cross-section view shows an x-z slice at one fixed y level.

### Important functions

```python
_config_from_args(args)
```

Builds the grid config. If data is provided, it reads the cache shape.

```python
_position(args.start, config)
```

Creates the starting balloon position.

```python
_build_field(name, config, args)
```

Creates one of:

- `SyntheticFlowField`
- `ReanalysisFlowField`
- `DataDrivenFlowField`

```python
_simulate(field, config, start, args)
```

Runs passive drift:

```text
look up wind -> update position -> repeat
```

```python
_build_topdown_figure(...)
```

Creates the x-y animation.

```python
_build_y_cross_section_figure(...)
```

Creates the x-z cross-section animation.

### Command for top-down all-fields demo

```bash
.pixi/envs/default/bin/python experiments/viz_passive_drift.py \
  --field all \
  --data data/era5_ithaca_3d.npz \
  --start 8 20 4 \
  --time-index 0 \
  --steps 30 \
  --scale 0.01 \
  --num-features 128 \
  --training-stride 2 \
  --output-dir experiments/output/passive_era5_ithaca
```

### Command for y cross-section demo

```bash
.pixi/envs/default/bin/python experiments/viz_passive_drift.py \
  --field all \
  --data data/era5_ithaca_3d.npz \
  --view y-cross-section \
  --cross-section-y 20 \
  --start 8 20 4 \
  --time-index 0 \
  --steps 30 \
  --scale 0.01 \
  --num-features 128 \
  --training-stride 2 \
  --output-dir experiments/output/passive_era5_ithaca
```

## How The Three Fields Differ

| Field | Uses real data? | Learns a model? | Main file | What `velocity_at` does |
|---|---:|---:|---|---|
| Synthetic GP | No | No, it samples from a prior | `src/env/field/synthetic.py` | Evaluates a sampled RFF GP field |
| ERA5 interpolation | Yes | No | `src/env/field/reanalysis.py` | Linearly interpolates cached ERA5 grid values |
| Data-driven GP | Yes | Yes | `src/env/field/data_driven.py` | Predicts with a fitted RFF GP regression model |

## The Most Important Distinction

Synthetic GP:

```text
"Make me a plausible smooth random wind field."
```

ERA5 interpolation:

```text
"Use the real ERA5 values directly and interpolate between them."
```

Data-driven GP:

```text
"Learn a smooth model from ERA5 samples, then use that model to predict wind."
```

## What The Parameters Mean

### `sigma`

Used by synthetic GP.

Controls wind strength.

Higher `sigma` means larger velocities.

### `lengthscale`

Used by synthetic and data-driven GP.

Controls smoothness.

Higher `lengthscale` means slower spatial variation.

### `num_features`

Used by synthetic and data-driven GP.

Controls the number of random Fourier features.

Higher is more expressive but slower.

### `noise_std`

Used by data-driven GP.

Controls regularization/noise assumption.

Higher means the model trusts training points less exactly and smooths more.

### `training_stride`

Used by data-driven GP when training from ERA5 cache.

Example:

```text
training_stride=2
```

means use every second grid point along each dimension.

This speeds up training and reduces the number of points.

### `scale`

Used by ERA5 and data-driven GP.

ERA5 wind is in meters per second. The environment moves in grid cells per simulation step.

`scale` converts:

```text
m/s -> grid cells per step
```

For the demo, we used `scale=0.01` so the balloon does not instantly hit the boundary.

## Tests

Relevant test files:

- `tests/test_field/test_synthetic.py`
- `tests/test_field/test_data_driven.py`
- `tests/test_field/test_reanalysis.py`
- `tests/test_field/test_reanalysis_realdata.py`
- `tests/test_field/test_fetch_era5.py`
- `tests/test_arena/`

The key data-driven GP tests check:

- fitting works in 2D
- fitting works in 3D
- posterior sampling is reproducible with the same seed
- posterior sampling changes with a different seed
- ERA5 cache fitting works
- invalid inputs are rejected

The command we have been using:

```bash
ERA5_CACHE=data/era5_ithaca_3d.npz .pixi/envs/default/bin/pytest tests/test_field tests/test_arena -q
```

Last known result:

```text
498 passed, 1 skipped
```

The skipped test is an optional ERA5 golden snapshot test.

## Meeting Questions You Should Be Ready For

### Why do we need synthetic GP if we have ERA5?

Because synthetic GP gives controlled fake wind fields. It is useful for algorithm development without needing real data or dealing with real-world messiness.

### Why do we need data-driven GP if we already have ERA5 interpolation?

Because interpolation is just a direct lookup/blending method. Data-driven GP is a model. It can smooth, generalize, and eventually support uncertainty/posterior sampling in a way interpolation does not.

### Is the data-driven GP trained from ERA5 or radiosonde?

Right now, the implemented helper trains from the ERA5 cache. The constructor itself can train from any `positions` and `velocities`, so radiosonde support would mainly require a loader that converts radiosonde observations into the same arrays.

### Does the balloon control altitude?

In the passive demo, no. `z` is fixed. The balloon only drifts horizontally according to the wind.

### Are coordinates continuous?

Yes for field lookup. `velocity_at` accepts continuous position values through `GridPosition`, so the balloon can be at fractional coordinates.

### Are we doing exact GP inference?

No. We use random Fourier features to approximate a GP-like kernel. This is much cheaper and easier for larger grids.

### Is synthetic GP data-driven?

No. Synthetic GP samples from a prior. Data-driven GP fits to data.

### Is data-driven GP the same as ERA5 interpolation?

No. ERA5 interpolation blends nearby grid values. Data-driven GP fits a global smooth function to samples and uses that function for prediction.

## Good Short Explanation

If you only have 30 seconds:

"We have three wind sources. The synthetic GP makes smooth fake wind fields using random Fourier features, so it does not need data. The ERA5 reanalysis field uses real downloaded ERA5 wind and linearly interpolates between grid points. The data-driven GP trains a smooth RFF GP regression model from ERA5 samples, then predicts wind at continuous positions. The passive demo drops a balloon at a chosen coordinate, repeatedly calls `velocity_at`, and updates x-y position by the returned wind while keeping altitude fixed."

## File Map

Synthetic GP:

- `src/env/field/synthetic.py`

Data-driven GP:

- `src/env/field/data_driven.py`
- `src/env/field/era5_data.py` when training from ERA5 cache

Real ERA5 interpolation:

- `experiments/field_estimation/scripts/fetch_era5.py`
- `src/env/field/era5_data.py`
- `src/env/field/reanalysis.py`

Demo:

- `experiments/viz_passive_drift.py`

Exports:

- `src/env/field/__init__.py`
- `src/env/__init__.py`

Tests:

- `tests/test_field/test_synthetic.py`
- `tests/test_field/test_data_driven.py`
- `tests/test_field/test_reanalysis.py`
- `tests/test_field/test_reanalysis_realdata.py`
- `tests/test_field/test_fetch_era5.py`

