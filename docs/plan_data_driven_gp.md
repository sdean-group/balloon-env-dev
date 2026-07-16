# Plan: Data-Driven GP Wind Field (Train the GP)

## The one-line idea

Source #3 is **source #1 with the prior learned from real data instead of guessed.**

Same RFF GP sampler as `SyntheticFlowField` — but we first fit its knobs (amplitude,
lengthscale, smoothness, mean) to real ERA5 winds offline, then sample from that fitted
prior at runtime.

## Why it's worth building (where it sits between the two existing fields)

| Field | What it gives | What it lacks |
|---|---|---|
| #1 `SyntheticFlowField` | Infinite novel realizations | Made-up statistics — wrong magnitude/smoothness |
| #2 `ReanalysisFlowField` | Real, realistic winds | Only `T` fixed replays of the archive |
| **#3 `FittedGPFlowField`** | **Infinite novel realizations whose statistics match real SF winds** | Matches *statistics*, not specific weather (see approach choice) |

## Three things "train the GP" could mean

| Approach | Meaning | Verdict |
|---|---|---|
| **(a) Fitted prior** | Learn Matérn (σ, ℓ, ν) + mean from ERA5; still sample the existing RFF prior | **Recommended.** Cheapest, reuses #1 almost verbatim, trivially testable |
| (b) Conditioned posterior | Condition on real ERA5 → realization = real-ish mean + random residual | Data-grounded, fits the forecast/reality design — but heavier, and largely *is* `mean + (a)` |
| (c) Learned generative model | A richer spatio-temporal / non-stationary model | Overkill; big project, hard to validate |

### Recommendation: do **(a)**, built so **(b)** is a later one-liner.

1. **It's literally the three-source story:** #1 is a guessed prior, #3 is the *same sampler
   with the prior fitted*. Clean contrast, almost no new code.
2. **Maximal reuse.** `SyntheticFlowField` already does Matérn-RFF sampling (incl. the 3D
   divergence-free streamfunction). The *fit* reuses a gpjax recipe already in this repo
   (`passive_estimation_uniform.py`): `Matern52` → `-conjugate_mll` → `gpx.fit` + `optax`.
3. **(a) is the hard part of (b).** A data-grounded realization is `mean + residual_GP` =
   `SumField(mean_field, fitted_gp)` using the *existing* `composite.py`. So (b) later needs
   **no new class** — just swap the mean for a real-data field. We get the forecast/reality
   wiring nearly for free.

> **Honest caveat:** (a) matches statistics, not specific weather — every draw is independent
> of the real archive. If the forecast/reality experiment is urgent, jump straight to (b) by
> composing our fitted GP with a real-data mean. The architecture makes that a one-line change.

## Architecture: the same 3-layer split as the linear-interp plan

| Layer | File | Job |
|---|---|---|
| **Offline fit** (run once) | `experiments/field_estimation/scripts/fit_gp.py` | Load ERA5 cache → fit Matérn + mean → save a tiny param file |
| **Loader** | `src/env/field/gp_model_data.py` | Load + validate that param file (mirrors `era5_data.py`) |
| **Field** | `src/env/field/fitted_gp.py` | `FlowField` that configures the RFF sampler from fitted params |

The field stays **pure**: no training in `velocity_at` — it just reads the saved params in
`__init__`, exactly like `ReanalysisFlowField` reads the `.npz` cache.

## The pieces, one at a time

### 1. Offline fit — `fit_gp.py`

- Load the existing ERA5 cache with `load_era5` (reuse the contract, invent nothing).
- Subtract the mean (store it); the kernel fits the zero-mean residual.
- Subsample ~500–1000 grid points/slice — a full GP is O(N³) and `T·n_x·n_y` is huge.
- **Objective:** maximize marginal likelihood (`-gpx.objectives.conjugate_mll`) with `optax`
  Adam — the existing recipe. Fit (σ, ℓ); **pick ν from {0.5, 1.5, 2.5} by held-out NLL**
  (gpjax has discrete Matérn classes, and ν isn't gradient-friendly).
- Save params + provenance to a small `.npz`. CLI mirrors `fetch_era5.py`.

### 2. Loader — `gp_model_data.py`

`load_gp_model(path) -> GpModelBundle(params, meta)`. Validates that σ, ℓ, ν > 0, values are
finite, and the component count matches the spatial rank — raising `ValueError` loudly, just
like `load_era5`.

### 3. Field — `FittedGPFlowField`

A thin wrapper that delegates the hot path to a `SyntheticFlowField` built from fitted params:

```python
class FittedGPFlowField(FlowField):
    def __init__(self, config, model_path, *, num_features=500):
        super().__init__(config)
        b = load_gp_model(model_path)          # validates vs config -> ValueError
        self._mean = b.mean                     # added back at query time
        self._gp = SyntheticFlowField(config, sigma=b.sigma,
                                      lengthscale=b.lengthscale, nu=b.nu,
                                      num_features=num_features)

    def reset(self, key):        self._gp.reset(key)        # draw RFF weights from LEARNED prior
    def velocity_at(self, p):    return self._gp.velocity_at(p) + mean   # (mean per component)
    def velocity_field(self):    return self._gp.velocity_field() + mean
    def sub_fields(self):        return (self._gp,)          # arena resets inner GP once
```

- **Trained model lives in the artifact file**, loaded once in `__init__`. Nothing fits at runtime.
- **`reset` samples** by delegating to `SyntheticFlowField.reset` — fresh RFF weights drawn from
  the *fitted* spectral density. The mean is constant, added deterministically.
- It's morally `SumField(ConstantDriftField(mean), SyntheticFlowField(fitted))`. Swap that mean
  for a `ReanalysisFlowField` and you have approach (b).

Then register it in `src/env/field/__init__.py`.

## Three modeling choices `SyntheticFlowField`'s defaults get wrong for real winds

Each is resolved offline at fit time. These are also the main open decisions (below):

1. **Mean.** ERA5 has a strong westerly; a zero-mean GP is wrong → fit and store a mean.
2. **Anisotropy (3D).** One isotropic ℓ conflates horizontal correlation (100s of km) with
   vertical (across pressure levels). Prefer a per-axis lengthscale `(ℓ_x, ℓ_y, ℓ_z)` — a small
   RFF-sampler extension.
3. **Divergence-free (3D).** #1's streamfunction is divergence-free; real ERA5 (u,v) isn't.
   Simplest honest fit = two independent component GPs (not divergence-free, like #2).

## Testing — same style as the existing suite

**Keystone trick** (the analogue of linear-interp's "affine ground truth"):
**draw data from a GP with *known* (σ, ℓ, ν, mean), run the fit, check the recovered values.**
Fully hermetic — no ERA5 download.

| File | Covers |
|---|---|
| `conftest.py` (extend) | `known_gp_npz` fixture + small saved-artifact fixtures |
| `test_gp_model_data.py` | Loader validation raises `ValueError` |
| `test_fitted_gp.py` | Field contract, mean/variance recovery, determinism, diversity |
| `test_fit_gp.py` | Keystone hyperparameter recovery + reproducibility (`@pytest.mark.slow`) |
| `test_fitted_gp_integration.py` | Drops into `NavigationArena`; inert agent drifts |
| `test_fitted_gp_realdata.py` | `@pytest.mark.era5`: fit real SF cache; plausibility + golden snapshot |

Acceptance bar (the measurable "done"):

- Interface: 2D → `(float, None)`, 3D → `(float, float)`; correct `velocity_field` shapes.
- Recovers known (σ, ℓ, mean) within ~15–20%; selects correct ν.
- Realization mean ≈ fitted mean; variance ≈ σ².
- Deterministic after `reset`; diverse across keys; finite everywhere in-domain.
- Bad artifacts raise `ValueError`; fit is seed-reproducible and beats a misspecified baseline.
- Drops into the arena (inert agent drifts on the fitted mean).

## Dependencies

None new — `gpjax`, `optax`, `jax`, `numpyro` are already under `feature.gp` in `pixi.toml`.
Keep `gpjax`/`optax` in the *offline* path only; the field imports just `jax` (via `SyntheticFlowField`).

## Build order

1. `fit_gp.py`: load cache → demean → subsample → MLL-fit (σ, ℓ) for one ν → save artifact.
2. `gp_model_data.py` loader + validation.
3. `FittedGPFlowField`, 2D first (single component, verbatim `SyntheticFlowField` reuse).
4. Hermetic tests green (loader, contract, keystone recovery).
5. Add ν selection by held-out NLL + fitted mean.
6. Extend to 3D — decide independent-component vs streamfunction, ARD lengthscale if chosen.
7. Demo: clone `viz_real_wind_drift.py`, swap in `FittedGPFlowField`; eyeball drift vs the others.
8. *(Later, ~free)* Approach (b): `SumField(real_mean, fitted_gp)` as the realized/observed pair.

---

## Decisions I need from you before coding

1. **Approach for v1 — confirm (a)?** Calibrated prior now, with (b) data-grounded posterior as
   a later `SumField` — or is the forecast/reality experiment urgent enough to build (b) now?
   *(This drives everything else.)*

2. **3D divergence-free?** Keep #1's streamfunction (physical, but a harder fit) or fit two
   independent (u, v) GPs (simplest, honest to ERA5, like #2)?

3. **Anisotropy?** Isotropic Matérn (verbatim reuse) or per-axis lengthscale (small extension,
   but matters a lot in 3D)?

4. **Region scope?** One SF-specific artifact (your RL-HAB note says ERA5 fits generalize poorly
   across regions), or bake a region argument in now?
