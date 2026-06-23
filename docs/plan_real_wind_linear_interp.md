# Plan: Real Wind Field via Linear Interpolation

## 0. One-paragraph summary

Add a new leaf `FlowField` — `ReanalysisFlowField` — that answers `velocity_at(p)` by
**(bi/tri)linear interpolation of real ERA5 winds** pre-resampled onto the env grid.
Real-world data acquisition and regridding happen **offline** (a script → cached `.npz`),
so the field class stays pure and fast: it just loads an array, picks a time slice on
`reset()`, and interpolates. It drops into `NavigationArena(realized_field=…, observed_field=…)`
exactly where `SyntheticFlowField` does today.

## 1. Design principles (carried from the existing code)

- **The field stays pure.** Per `src/env/field/flow_field.py`, a `FlowField` only answers
  "what's the wind at `p`?". No data download, no CDS API calls, no unit policy decisions
  inside `velocity_at`. Those live offline or in `__init__`.
- **`reset(rng_key)` draws *a realization*.** For the GP that's new RFF weights; for real
  data it's **which historical time slice** to use this episode (mirrors `UniformDriftField.reset`
  consuming the key).
- **Deterministic after reset.** Same `rng_key` → same slice → identical trajectories
  (the contract the `tests/test_field/test_synthetic.py` tests enforce).
- **Separation into 3 layers** so each is independently testable:

| Layer | Where | Responsibility |
|---|---|---|
| Acquisition + regrid | `experiments/.../fetch_era5.py` (offline, run once) | CDS download → resample to `(n_x,n_y[,n_z],T,components)` → save `.npz` |
| Loader | `src/env/field/era5_data.py` | Load `.npz`, expose array + metadata, no env coupling |
| Field | `src/env/field/reanalysis.py` | `FlowField` subclass: interpolate + `reset` slice selection + unit scaling |

## 2. The data array contract (the linchpin)

Everything keys off one cached array with a fixed shape. **Pre-resample ERA5 onto the env
grid offline** so that grid index == array index and `velocity_at` becomes pure interpolation
on a unit grid (no lat/lon math in the hot path).

Cached `.npz` contains:

- **2D:** `winds` shape `(T, n_x, n_y, 1)` — component `u` only (one ambient axis).
- **3D:** `winds` shape `(T, n_x, n_y, n_z, 2)` — components `(u, v)`; `z` axis = vertical level.
- `meta`: dict — source region (SF box), lon/lat/level edges used for regridding, native units
  (`m/s`), timestamps for each `T`, and the `(n_x,n_y,n_z)` it was built for.

This is deliberately the **same layout `velocity_field()` already returns**
(`(n_x,n_y,1)` / `(n_x,n_y,n_z,2)`), so rendering is free.

## 3. Coordinate & unit conventions — the part to get right

This is where real fields bite you. Three mappings, all resolved **offline** during regrid so
the field doesn't think about them:

1. **Axis assignment** (from `src/env/utils/types.py`): ambient `(i,j)` → `(lon, lat)`;
   controllable `k` → **vertical level**. This matches the balloon: the agent actuates altitude,
   wind carries it horizontally.
2. **Spatial regridding:** ERA5's native lon/lat/pressure grid → uniform `1..n_x`, `1..n_y`,
   `1..n_z`. Do this with `scipy.interpolate.RegularGridInterpolator` at *build time*, sampling
   at the env grid-cell centers. After this, env grid index maps linearly to physical coordinate.
3. **Unit scaling:** ERA5 winds are **m/s**; the GP `sigma` is dimensionless grid-cells/step.
   Convert with `scale = dt_seconds / dx_meters_per_cell`, applied per component. Make `dt`, `dx`
   explicit constructor args (or bake into the offline step). **Get this wrong and drift
   magnitude is meaningless vs. the GP runs** — validate by comparing typical `|u|` to a GP run
   with `sigma≈3`.

> ⚠️ **Vertical-axis subtlety:** altitude↔pressure is nonlinear. Decide the vertical coordinate
> at regrid time (geopotential height is most physical for a balloon) and grid on *that*; then
> linear interp in `k` is honest. Document the choice in `meta`.

## 4. File-by-file implementation

### 4a. `experiments/field_estimation/scripts/fetch_era5.py` (offline, run once)
- Use `cdsapi` to pull ERA5 `u`/`v` over the SF box, a chosen pressure-level set, and a time
  range (e.g. N days hourly → that's your `T` realizations).
- Regrid each timestep onto `(n_x,n_y[,n_z])` via `RegularGridInterpolator`.
- Store **raw m/s** + `dx`,`dt` in meta (so `dt`/`dx` can be retuned without re-downloading).
- `np.savez_compressed("data/era5_sf_<grid>.npz", winds=…, meta=…)`.
- This is the only network-touching code; keep it out of `src/`.

### 4b. `src/env/field/era5_data.py`
- `load_era5(path) -> Era5Bundle` returning the array + parsed meta.
- Validate shape against an expected `GridConfig`. No JAX, no env imports beyond types.

### 4c. `src/env/field/reanalysis.py` — `ReanalysisFlowField(FlowField)`

```python
class ReanalysisFlowField(FlowField):
    def __init__(self, config, data_path, *, scale=1.0,
                 slice_mode="random"):  # "random" | "fixed"
        super().__init__(config)
        bundle = load_era5(data_path)          # (T, n_x, n_y[, n_z], C)
        assert bundle.winds.shape[1:1+config.ndim] == config.shape
        self._winds = bundle.winds * scale     # to cells/step
        self._T = self._winds.shape[0]
        self._slice_mode = slice_mode
        self._interp_u = None
        self._interp_v = None  # 3D only

    def reset(self, rng_key):
        # pick which real weather snapshot is "this episode"
        if self._slice_mode == "fixed":
            t = 0
        else:
            t = int(jax.random.randint(rng_key, (), 0, self._T))
        self._build_interpolators(self._winds[t])   # RegularGridInterpolator(linear)

    def velocity_at(self, position):
        # NOTE: interpolate over the FULL spatial position, not just the
        # ambient axes. In 2D the wind varies over BOTH grid axes (x, y) but
        # has a single component u -- this mirrors SyntheticFlowField, which
        # evaluates the GP over (x, y) and returns (u, None).
        if self.ndim == 2:
            u = float(self._interp_u((position.i, position.j)))
            return (u, None)
        pt = (position.i, position.j, position.k)
        return (float(self._interp_u(pt)), float(self._interp_v(pt)))

    def velocity_field(self):
        return self._current_slice  # already (n_x,n_y[,n_z],C)
```

- Build one `RegularGridInterpolator(method="linear", bounds_error=False, fill_value=None)` per
  component over axes `arange(1, n+1)` so it matches the **1-indexed continuous domain** and
  **clamps/extrapolates at the boundary** (consistent with the arena's `boundary_mode="clip"`).
- `bounds_error=False, fill_value=None` → linear extrapolation at edges; safer than NaN given
  fractional positions can sit at `n_x` exactly.

### 4d. Register in `src/env/field/__init__.py`
Add `ReanalysisFlowField` to imports + `__all__`.

## 5. Testing & success criteria

The implementation is "done" when the acceptance criteria below hold and the test suite is
green. Tests mirror the contract style of `tests/test_field/test_synthetic.py`.

### 5.1 Acceptance criteria (the measurable "done" definition)

| # | Criterion | Measurable threshold |
|---|---|---|
| 1 | **Interface contract** | Subclasses `FlowField`; 2D `velocity_at` → `(float, None)`, 3D → `(float, float)`; `velocity_field()` shape `(n_x,n_y,1)` / `(n_x,n_y,n_z,2)` |
| 2 | **Interpolation is exactly linear** | On an affine ground-truth field, `velocity_at` matches the analytic value at random fractional points to `< 1e-5` |
| 3 | **Node-exactness** | At integer positions, `velocity_at` equals the stored grid cell to `< 1e-6` |
| 4 | **Two access paths agree** | `velocity_field()[i-1,j-1]` == `velocity_at(GridPosition(i,j))` for every node |
| 5 | **Deterministic after reset** | Same `rng_key` → byte-identical field & trajectory; `fixed` mode always picks `t=0` |
| 6 | **Realization diversity** | Over many keys in `random` mode, ≥ 2 distinct slices selected, all in `[0, T)` |
| 7 | **Unit scaling linear & correct** | Output scales exactly with `scale`; real-data magnitudes land in a sane cells/step band vs. GP `sigma≈3` |
| 8 | **No NaN/Inf in-domain** | Every query in `[1,n]` per axis is finite; boundary-exact queries finite |
| 9 | **Data validation fails loudly** | Shape/ndim mismatch vs `GridConfig` and NaN-laden data raise `ValueError` at load/construct, not silently |
| 10 | **Drops into the arena** | An inert `STAY` agent shows non-zero net drift; trajectory finite and in-bounds under `clip` |
| 11 | **Physical plausibility (real data)** | Magnitudes within reanalysis ranges; spatially smooth; consecutive time slices correlated |
| 12 | **Regression-stable** | A golden `(slice, position) → velocity` snapshot is unchanged across refactors |

> Explicitly **not** a criterion: divergence-free flow. Trilinear interpolation is *not*
> divergence-free (unlike the streamfunction GP). Don't test for it; optionally *bound*
> divergence under real data (criterion 11).

### 5.2 The keystone trick — affine ground truth

Bilinear/trilinear interpolation reproduces an affine field (`u = a·x + b·y [+ c·z] + d`)
**exactly**. So a fixture whose data is an affine function gives a closed-form ground truth:
assert `velocity_at` to float tolerance instead of eyeballing. This single idea covers criteria
2, 3, and 4 with **no real data**.

### 5.3 Test files (hermetic unless marked)

| File | Covers criteria | Notes |
|---|---|---|
| `tests/test_field/conftest.py` | — | Affine `.npz` fixtures (2D + 3D), arena builder helper |
| `tests/test_field/test_era5_data.py` | 9 | Loader: shape/ndim/NaN validation, component count |
| `tests/test_field/test_reanalysis.py` | 1–8 | Core contract, affine-exactness, reset/determinism, scaling, boundary |
| `tests/test_field/test_reanalysis_integration.py` | 10 | Drops into `NavigationArena`; inert-agent drift; shared-field correlation |
| `tests/test_field/test_reanalysis_realdata.py` | 11, 12 | `@pytest.mark.era5`, opt-in; plausibility + golden snapshot |

Register the opt-in marker in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["era5: requires a real ERA5 cache; skipped in CI"]
```

Run:
```bash
pixi run pytest tests/test_field/ -v            # hermetic, fast (CI)
pixi run pytest tests/test_field/ -m era5 -v    # real-data, local only
```

> Loader/constructor contract these tests assume: `load_era5(path) -> Era5Bundle` (with
> `.winds`, `.meta`); the **field constructor raises `ValueError`** (not bare `assert`) on a
> `GridConfig` shape/ndim mismatch, so validation is testable under `python -O`.

## 6. Integration / demo

Clone `experiments/viz_wind_drift.py` → `viz_real_wind_drift.py`, swapping
`SyntheticFlowField(...)` for `ReanalysisFlowField(config, data_path, scale=...)`. Everything
downstream (arena, renderer, `velocity_field`) is unchanged — that's the payoff of the shared
interface. Visually confirm the drift looks like coherent real weather, and that magnitude is
comparable to the GP demo.

## 7. Forecast/reality wiring (sets up later work, ~free)

Because of `src/env/field/composite.py`: `realized = ReanalysisFlowField(analysis.npz)` and
`observed = ReanalysisFlowField(forecast.npz)` (or a degraded/lagged slice). Pass both to
`NavigationArena(realized_field=…, observed_field=…)`. **Do not build this yet** — but shaping
`reset` around slice-index selection now means a shared time index can later keep the two
correlated, matching the two-field design notes.

## 8. Dependencies

Add to `pixi.toml`:
- `scipy` (explicit — currently only transitive via scikit-learn) → interpolation.
- Offline-only (could be a separate `feature`): `cdsapi`, `xarray`, `cfgrib` (or `netCDF4`)
  for ERA5. Keep these out of the default env if you want fetch to be opt-in.

## 9. Build order (checklist)

1. `fetch_era5.py` → produce one real `.npz` for the SF box (and a tiny fixture for tests).
2. `era5_data.py` loader + validation.
3. `ReanalysisFlowField` (2D first — simplest, one component).
4. Tests (fixture-based) → green.
5. `viz_real_wind_drift.py` demo; eyeball drift + magnitude vs GP.
6. Extend to 3D (vertical level) + tune `scale` (`dt`/`dx`).
7. (Later) forecast/reality two-field experiment.

---

**Two decisions to lock before coding:**
(a) pre-resample offline onto the env grid — recommended, keeps `velocity_at` trivial;
(b) vertical coordinate for the `k` axis (geopotential height vs. pressure level).
