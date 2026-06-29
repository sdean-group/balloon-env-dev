# Data-Driven GP Line-by-Line Guide

This document explains:

`src/env/field/data_driven.py`

The goal of that file is to define `DataDrivenFlowField`, a wind field that learns a smooth function from data. In practice, the data is usually ERA5 wind data saved in an `.npz` cache.

The model learns:

```text
(x, y, z) -> (u, v)
```

where:

- `(x, y, z)` is position in the environment grid.
- `u` is horizontal wind in the x direction.
- `v` is horizontal wind in the y direction.

For 2D fields, it learns:

```text
(x, y) -> (u)
```

## Big Picture

Data-driven GP is different from the synthetic GP.

Synthetic GP:

```text
sample a random smooth wind field from scratch
```

Data-driven GP:

```text
fit a smooth wind field to observed data
```

It is also different from direct ERA5 interpolation.

ERA5 interpolation:

```text
blend nearby ERA5 grid values directly
```

Data-driven GP:

```text
learn a smooth model from ERA5 samples, then predict with the model
```

## Lines 1-13: Imports And Setup

```python
1  """Data-driven wind source fitted with random Fourier feature GP regression."""
```

This file implements a data-driven wind source. "Random Fourier feature GP regression" means we approximate a Gaussian-process-like smooth function using random wave features.

```python
3  from __future__ import annotations
```

This makes type annotations easier to use. It lets Python delay evaluating type hints.

```python
5  from typing import Optional, Sequence, Tuple
```

These are type hints:

- `Optional[X]`: either `X` or `None`
- `Sequence`: list-like or tuple-like
- `Tuple`: fixed-size tuple

```python
7  import jax
8  import jax.numpy as jnp
9  import numpy as np
```

The file uses both JAX and NumPy.

- NumPy is used for fitting/training.
- JAX is used for random keys during episode reset, matching the rest of the environment.

```python
11 from .era5_data import load_era5
```

Imports the ERA5 cache loader. This is used by `from_era5_cache`.

```python
12 from .flow_field import FlowField
```

Imports the base class that all wind fields follow.

```python
13 from ..utils.types import GridConfig, GridPosition
```

Imports environment types:

- `GridConfig`: describes grid size and dimensionality.
- `GridPosition`: stores positions like `(x, y)` or `(x, y, z)`.

## Lines 16-23: Class Definition

```python
16 class DataDrivenFlowField(FlowField):
```

Defines a wind field class. It inherits from `FlowField`, so it follows the same interface as synthetic and ERA5 fields.

That means it has:

```python
reset(...)
velocity_at(...)
velocity_field(...)
```

```python
17-23 """Continuous GP approximation fitted to measured or reanalysis winds..."""
```

The class docstring says the important idea:

- It is continuous, so it can predict at non-integer positions.
- It is fitted to measured/reanalysis wind.
- It uses random Fourier features.
- It fits each wind component separately, but with the same spatial features.
- `reset` chooses the posterior mean field or a sampled plausible field.

## Lines 25-36: Constructor Signature

```python
25 def __init__(
26     self,
27     config: GridConfig,
28     positions: np.ndarray,
29     velocities: np.ndarray,
30     *,
31     num_features: int = 256,
32     lengthscale: float = 8.0,
33     noise_std: float = 0.1,
34     feature_seed: int = 0,
35     sample_posterior: bool = False,
36 ):
```

This is how you directly create a data-driven GP from training arrays.

Inputs:

- `config`: grid dimensions.
- `positions`: training coordinates.
- `velocities`: training wind values.
- `num_features`: number of random wave features.
- `lengthscale`: how smooth the learned field should be.
- `noise_std`: regularization/noise assumption.
- `feature_seed`: seed for random features.
- `sample_posterior`: whether reset should sample a plausible field instead of using the best-fit mean field.

Usually, we do not manually build `positions` and `velocities`. We use:

```python
DataDrivenFlowField.from_era5_cache(...)
```

## Line 37: Initialize Parent Class

```python
37 super().__init__(config)
```

This calls the base `FlowField` constructor.

It stores the config and gives the object properties like:

```python
self.config
self.ndim
```

`self.ndim` is either:

- `2` for 2D
- `3` for 3D

## Lines 38-43: Validate Hyperparameters

```python
38 if num_features <= 0 or num_features != int(num_features):
39     raise ValueError("num_features must be a positive integer")
```

The number of features must be a positive integer.

Bad examples:

```text
0
-10
10.5
```

```python
40 if lengthscale <= 0:
41     raise ValueError("lengthscale must be positive")
```

The lengthscale must be positive because it is used as a scale in the random feature frequencies.

```python
42 if noise_std <= 0:
43     raise ValueError("noise_std must be positive")
```

The noise/regularization value must be positive. Zero would make the linear algebra less stable.

## Lines 45-47: Decide Output Shape And Convert Data

```python
45 expected_components = 1 if self.ndim == 2 else 2
```

If the field is 2D, it expects one wind output component: `u`.

If the field is 3D, it expects two wind output components: `(u, v)`.

```python
46 x = np.asarray(positions, dtype=np.float64)
47 y = np.asarray(velocities, dtype=np.float64)
```

Convert training inputs and outputs into NumPy arrays with float precision.

Here:

- `x` means training positions.
- `y` means training velocities.

This naming is standard in machine learning:

```text
x = inputs
y = targets/labels/outputs
```

## Lines 48-56: Validate Data Shapes

```python
48 if x.ndim != 2 or x.shape[1] != self.ndim:
49     raise ValueError(...)
```

The positions must have shape:

```text
(number_of_training_points, number_of_dimensions)
```

For example, in 3D:

```text
(1600, 3)
```

Each row is:

```text
[x, y, z]
```

```python
52 if y.ndim != 2 or y.shape != (x.shape[0], expected_components):
53     raise ValueError(...)
```

The velocities must match the positions.

For 3D:

```text
positions shape  = (n, 3)
velocities shape = (n, 2)
```

Each velocity row is:

```text
[u, v]
```

For 2D:

```text
positions shape  = (n, 2)
velocities shape = (n, 1)
```

## Lines 57-60: Validate Data Is Present And Finite

```python
57 if x.shape[0] == 0:
58     raise ValueError("at least one training observation is required")
```

You cannot train a data-driven model with zero data.

```python
59 if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
60     raise ValueError("training data must contain only finite values")
```

The training data cannot contain:

- `NaN`
- `Inf`
- `-Inf`

Those would break fitting.

## Lines 62-65: Validate Positions Are Inside The Grid

```python
62 lower = np.ones(self.ndim)
63 upper = np.asarray(self.config.shape, dtype=np.float64)
```

The environment uses 1-indexed coordinates.

So the valid range is:

```text
x in [1, n_x]
y in [1, n_y]
z in [1, n_z]
```

```python
64 if np.any(x < lower) or np.any(x > upper):
65     raise ValueError(...)
```

Reject training points outside the environment domain.

## Lines 67-71: Store Hyperparameters

```python
67 self.num_features = int(num_features)
68 self.lengthscale = float(lengthscale)
69 self.noise_std = float(noise_std)
70 self.feature_seed = int(feature_seed)
71 self.sample_posterior = bool(sample_posterior)
```

The class stores configuration values for later use.

These values control the GP approximation and reset behavior.

## Lines 73-78: Create Random Fourier Features

```python
73 rng = np.random.default_rng(self.feature_seed)
```

Creates a NumPy random number generator.

The seed makes the random features reproducible.

```python
74 self._omegas = rng.normal(
75     scale=1.0 / self.lengthscale,
76     size=(self.num_features, self.ndim),
77 )
```

Samples random frequencies.

Shape:

```text
(num_features, ndim)
```

If:

```text
num_features = 128
ndim = 3
```

then:

```text
_omegas shape = (128, 3)
```

Plain English:

Each row describes one random wave direction/frequency in space.

The `scale=1.0 / lengthscale` part means:

- larger lengthscale -> smaller frequencies -> smoother functions
- smaller lengthscale -> larger frequencies -> more rapidly changing functions

```python
78 self._phases = rng.uniform(0.0, 2.0 * np.pi, size=self.num_features)
```

Samples random phase shifts for each feature.

Plain English:

This shifts each wave left/right so all features are not aligned.

## Lines 80-85: Standardize Wind Outputs

```python
80 # Standardizing each output makes one noise_std meaningful across u/v and
81 # avoids poorly conditioned solves when raw winds have a large mean.
```

The comment explains why standardization exists.

```python
82 self._target_mean = y.mean(axis=0)
```

Computes the average wind for each output component.

For 3D, this gives:

```text
mean_u, mean_v
```

```python
83 self._target_std = y.std(axis=0)
```

Computes the standard deviation of each wind component.

```python
84 self._target_std = np.where(self._target_std < 1e-8, 1.0, self._target_std)
```

If a component has nearly zero variation, avoid dividing by a tiny number.

```python
85 y_standardized = (y - self._target_mean) / self._target_std
```

Convert raw wind into standardized wind.

Plain English:

Instead of fitting raw wind values directly, fit values centered around 0 with scale about 1.

This makes the math more stable.

## Lines 87-89: Build Training Feature Matrix

```python
87 phi = self._features(x)
```

Convert every training position into random Fourier features.

If:

```text
x shape = (1600, 3)
num_features = 128
```

then:

```text
phi shape = (1600, 128)
```

Each row is the feature representation of one training point.

```python
88 regularizer = self.noise_std**2
```

Use `noise_std` as regularization.

```python
89 precision = phi.T @ phi + regularizer * np.eye(self.num_features)
```

This builds the matrix for Bayesian linear regression / ridge regression.

Plain English:

The model is solving:

```text
which feature weights best predict wind?
```

The regularizer prevents the weights from becoming too extreme.

## Lines 90-93: Cholesky Factorization

```python
90 try:
91     self._precision_cholesky = np.linalg.cholesky(precision)
92 except np.linalg.LinAlgError as exc:
93     raise ValueError("could not factor GP posterior precision") from exc
```

Cholesky factorization is a numerically stable way to solve this matrix problem.

If the matrix is broken or not positive definite, fitting fails with a clear error.

## Lines 95-99: Solve For Posterior Mean Weights

```python
95 rhs = phi.T @ y_standardized
```

Build the right-hand side of the linear system.

```python
96 intermediate = np.linalg.solve(self._precision_cholesky, rhs)
```

First triangular solve.

```python
97 self._posterior_mean = np.linalg.solve(
98     self._precision_cholesky.T, intermediate
99 )
```

Second triangular solve.

Together, these compute the best-fit weights.

The result:

```text
_posterior_mean shape = (num_features, output_components)
```

For 3D:

```text
(128, 2)
```

Plain English:

This is the learned model. It says how much each random feature contributes to `u` and `v`.

## Lines 101-104: Training RMSE And Runtime State

```python
101 fitted = self._unstandardize(phi @ self._posterior_mean)
```

Predict wind back on the training points.

```python
102 self.training_rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
```

Compute training RMSE.

RMSE means root mean squared error.

Plain English:

This tells us how close the model's fitted wind values are to the training wind values.

```python
103 self._weights: Optional[np.ndarray] = None
104 self._precomputed: Optional[np.ndarray] = None
```

These are not filled until `reset`.

- `_weights`: the active weights for this episode.
- `_precomputed`: the full grid field for visualization.

## Lines 106-118: ERA5 Cache Constructor

```python
106 @classmethod
107 def from_era5_cache(...)
```

This is a helper constructor.

Instead of manually creating `positions` and `velocities`, this method loads an ERA5 cache and builds them for you.

```python
109 config: GridConfig,
110 data_path: str,
```

Inputs:

- `config`: expected grid shape
- `data_path`: path to ERA5 `.npz`

```python
112 time_index: int = 0,
```

Which ERA5 time slice to train on.

```python
113 scale: float = 1.0,
```

Converts ERA5 m/s wind into environment grid cells per step.

```python
114 training_stride: int = 1,
```

Use every grid point if stride is 1.

Use every other grid point if stride is 2.

```python
115 max_training_points: Optional[int] = None,
```

Optionally cap the number of training points.

```python
116 **kwargs,
```

Passes extra arguments like:

- `num_features`
- `lengthscale`
- `noise_std`
- `feature_seed`
- `sample_posterior`

## Lines 119-133: Load And Validate ERA5 Cache

```python
119 bundle = load_era5(data_path)
120 winds = bundle.winds
```

Load the ERA5 cache.

`winds` is the raw wind array.

For 3D:

```text
winds shape = (T, n_x, n_y, n_z, 2)
```

```python
121 expected_rank = config.ndim + 2
```

Expected array rank:

- 2D config: `4`, meaning `(T, n_x, n_y, 1)`
- 3D config: `5`, meaning `(T, n_x, n_y, n_z, 2)`

```python
122 if winds.ndim != expected_rank:
123-125     raise ValueError(...)
```

Reject cache if dimensionality does not match config.

```python
126 if tuple(winds.shape[1 : 1 + config.ndim]) != tuple(config.shape):
127     raise ValueError("cache grid shape does not match config")
```

Reject cache if the grid shape does not match.

```python
128 if not 0 <= time_index < winds.shape[0]:
129-131     raise ValueError(...)
```

Reject invalid time index.

```python
132 if training_stride <= 0:
133     raise ValueError("training_stride must be positive")
```

Stride must be positive.

## Lines 135-142: Turn ERA5 Grid Into Training Arrays

```python
135 axes = [np.arange(1, n + 1, dtype=np.float64) for n in config.shape]
```

Create coordinate axes using the repo's 1-indexed convention.

Example for `n_x=40`:

```text
[1, 2, 3, ..., 40]
```

```python
136 mesh = np.meshgrid(*axes, indexing="ij")
```

Create full coordinate grids.

For 3D, this creates arrays for all `(x, y, z)` combinations.

```python
137 positions_grid = np.stack(mesh, axis=-1)
```

Combine those coordinate grids into one array where each grid cell stores its coordinate.

For 3D, each entry is:

```text
[x, y, z]
```

```python
138 slices = tuple(slice(None, None, training_stride) for _ in config.shape)
```

Build slicing rules for subsampling.

If `training_stride=2`, use every second grid point along each dimension.

```python
139 positions = positions_grid[slices].reshape(-1, config.ndim)
```

Flatten grid coordinates into a training table.

For 3D:

```text
positions shape = (n_training_points, 3)
```

```python
140 velocities = (winds[time_index][slices] * float(scale)).reshape(
141     positions.shape[0], -1
142 )
```

Pick the ERA5 time slice, subsample it, scale velocities, and flatten into a training table.

For 3D:

```text
velocities shape = (n_training_points, 2)
```

## Lines 144-153: Optional Random Subsampling

```python
144 if max_training_points is not None:
```

Only run this block if the caller asked for a cap.

```python
145 if max_training_points <= 0:
146     raise ValueError("max_training_points must be positive")
```

Validate cap.

```python
147 if positions.shape[0] > max_training_points:
```

Only subsample if we have too many points.

```python
148 rng = np.random.default_rng(int(kwargs.get("feature_seed", 0)))
```

Create reproducible random generator.

```python
149 chosen = rng.choice(
150     positions.shape[0], size=max_training_points, replace=False
151 )
```

Choose random training rows without duplicates.

```python
152 positions = positions[chosen]
153 velocities = velocities[chosen]
```

Keep only the chosen positions and velocities.

## Lines 155-159: Fit Model And Attach Metadata

```python
155 model = cls(config, positions, velocities, **kwargs)
```

This creates the actual `DataDrivenFlowField`, using the arrays we just built.

This line triggers the whole fitting process from lines 37-104.

```python
156 model.data_metadata = bundle.meta
```

Store ERA5 metadata on the model.

```python
157 model.time_index = int(time_index)
```

Store which time slice was used.

```python
158 model.training_points = int(positions.shape[0])
```

Store how many training points were used.

```python
159 return model
```

Return the fitted field.

## Lines 161-164: Feature Map

```python
161 def _features(self, positions: np.ndarray) -> np.ndarray:
```

This method converts raw coordinates into random Fourier features.

```python
162 positions = np.asarray(positions, dtype=np.float64)
```

Make sure positions are a NumPy float array.

```python
163 phase = positions @ self._omegas.T + self._phases
```

Compute the wave phase for each position and feature.

Shape example:

```text
positions shape = (1600, 3)
_omegas.T shape = (3, 128)
phase shape = (1600, 128)
```

```python
164 return np.sqrt(2.0 / self.num_features) * np.cos(phase)
```

Return cosine features.

Plain English:

Each position becomes a vector of smooth wave values.

## Lines 166-167: Convert Back To Original Wind Scale

```python
166 def _unstandardize(self, values: np.ndarray) -> np.ndarray:
167     return values * self._target_std + self._target_mean
```

The model predicts standardized wind. This converts it back to real scaled wind units.

## Lines 169-185: Reset Selects Active Field Weights

```python
169 def reset(self, rng_key: jnp.ndarray) -> None:
170     """Select the posterior mean or draw one coherent posterior field."""
```

`reset` prepares the field for one simulation episode.

```python
171 if self.sample_posterior:
```

If posterior sampling is enabled, generate a plausible random field around the learned mean.

```python
172-179 z = np.asarray(jax.random.normal(...), dtype=np.float64)
```

Draw standard random noise with the same shape as `_posterior_mean`.

```python
180 perturbation = self.noise_std * np.linalg.solve(
181     self._precision_cholesky.T, z
182 )
```

Transform the random noise into a posterior weight perturbation.

Plain English:

This creates one plausible alternate set of model weights.

```python
183 self._weights = self._posterior_mean + perturbation
```

Use the perturbed weights this episode.

```python
184 else:
185     self._weights = self._posterior_mean.copy()
```

If posterior sampling is disabled, use the best-fit mean weights.

In our usual demo runs, this is the normal path.

## Lines 187-192: Precompute Full Grid Field

```python
187 axes = [np.arange(1, n + 1, dtype=np.float64) for n in self.config.shape]
188 mesh = np.meshgrid(*axes, indexing="ij")
189 positions = np.stack(mesh, axis=-1).reshape(-1, self.ndim)
```

Build every integer grid position.

```python
190 values = self._predict(positions)
```

Predict wind at every grid point.

```python
191 components = 1 if self.ndim == 2 else 2
```

Decide whether the output has one or two wind components.

```python
192 self._precomputed = values.reshape(*self.config.shape, components)
```

Store the grid-shaped wind field.

This is mostly used for visualization.

## Lines 194-199: Predict At Arbitrary Positions

```python
194 def _predict(self, positions: np.ndarray) -> np.ndarray:
```

Internal prediction method.

```python
195 if self._weights is None:
196-198     raise RuntimeError(...)
```

You must call `reset` before predicting, because `reset` chooses active weights.

```python
199 return self._unstandardize(self._features(positions) @ self._weights)
```

This is the actual model prediction.

Step by step:

1. Convert positions to features.
2. Multiply features by learned weights.
3. Convert standardized output back to wind scale.

## Lines 201-212: Public `velocity_at`

```python
201 def velocity_at(
202     self, position: GridPosition
203 ) -> Tuple[float, Optional[float]]:
```

This is the method the simulator calls.

Input:

```text
GridPosition(x, y, z)
```

Output:

```text
(u, v)
```

```python
204 coords: Sequence[float]
```

Type hint for the coordinate tuple.

```python
205 if self.ndim == 2:
206     coords = (position.i, position.j)
```

For 2D, use `(x, y)`.

```python
207 else:
208     coords = (position.i, position.j, position.k)
```

For 3D, use `(x, y, z)`.

```python
209 value = self._predict(np.asarray([coords], dtype=np.float64))[0]
```

Predict wind for this one position.

The array wrapper makes the shape:

```text
(1, ndim)
```

Then `[0]` extracts the single predicted velocity.

```python
210 if self.ndim == 2:
211     return (float(value[0]), None)
```

For 2D, return one component.

```python
212 return (float(value[0]), float(value[1]))
```

For 3D, return both horizontal components.

## Lines 214-219: Public `velocity_field`

```python
214 def velocity_field(self) -> np.ndarray:
```

Returns the full precomputed grid field.

```python
215 if self._precomputed is None:
216-218     raise RuntimeError(...)
```

You must call `reset` first.

```python
219 return np.asarray(self._precomputed)
```

Return the full field as a NumPy array.

This is used by visualization code to draw arrows.

## End-To-End Data Flow

When we run the data-driven GP demo, this happens:

```text
1. Load ERA5 cache
2. Build coordinate table
3. Build velocity table
4. Convert coordinates into random Fourier features
5. Fit weights with Bayesian linear regression
6. Reset field
7. Predict wind at balloon position
8. Move balloon
9. Repeat
```

## One Good Meeting Explanation

Say this:

"The data-driven GP takes wind samples from a dataset, currently ERA5, and fits a smooth function from position to wind velocity. The code converts each position into random Fourier features, which are basically random smooth waves. Then it solves a regularized linear regression problem to learn how to combine those waves to match the observed wind. At runtime, the simulator calls `velocity_at`, which evaluates that learned function at the balloon's continuous position."

## Important Caveats

1. This is not exact GP inference.
   - It is an RFF approximation to GP regression.

2. This currently trains on one time slice.
   - `time_index` chooses which ERA5 time slice.

3. This currently uses ERA5 cache data.
   - The class could train from radiosonde data too, but we would need a loader that creates `positions` and `velocities`.

4. The passive demo keeps altitude fixed.
   - The field is 3D, but the balloon does not move vertically in this demo.

5. Training RMSE is not evaluation.
   - It only says how well the model matches its training samples.

