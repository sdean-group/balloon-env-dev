# Wind fields & simulator — clean design

## Goal

A minimal, single-responsibility design for wind in the balloon simulator. Two ideas only:

1. A **`FlowField`** is a pure function over space: "what is the wind at point `p`?" It is the
   *source* of velocities (synthetic GP, real ERA5, learned model). It knows nothing about
   forecasts, noise, clipping, or agents.
2. The **simulator (arena)** holds *two* fields — the **realized** wind `W` (moves the
   balloon) and the **observed** wind `Ŵ` (what the agent sees) — and turns them into
   per-step motion and observations.

Everything the old `AbstractField` did beyond "give me the wind" (per-step noise, clipping,
displacement PMFs, `disp_levels`) either moves to the simulator or is **deleted**.

> **We are dropping the PMF / DP-oracle machinery entirely.** It existed to give a
> discrete dynamic-programming agent an analytical transition distribution. We're continuous
> now and not targeting that agent, so `get_displacement_pmf*`, `disp_levels`, and the
> clipped-normal PMF math all go away. This removes a large amount of code and an entire
> "does the oracle use W or Ŵ?" design question.

---

## Why this is the clean split

The current `AbstractField` does three unrelated jobs:

1. **Where velocities come from** — the GP/RFF math. *(a spatial source)*
2. **How a per-step displacement happens** — `sample_displacement`: add noise, clip to
   `d_max`. *(a transition process)*
3. **The analytical transition distribution** — `get_displacement_pmf*`. *(an oracle model)*

Clean design: a `FlowField` is only #1. #2 belongs to the simulator (it's about how *this
simulation* turns wind into motion). #3 is deleted. Single responsibility per object, and
the two-field framework drops out for free.

---

## Layer 1: `FlowField` — a wind source

```python
class FlowField(ABC):
    """A deterministic continuous velocity field, fixed after reset(). Nothing else.

    Subclasses differ ONLY in where velocities come from.
    """

    def __init__(self, config: GridConfig):
        self.config = config

    @property
    def ndim(self) -> int:
        return self.config.ndim

    @abstractmethod
    def reset(self, rng_key: jnp.ndarray) -> None:
        """Draw one realization of the field (called once per episode)."""

    @abstractmethod
    def velocity_at(self, position: GridPosition) -> Tuple[float, Optional[float]]:
        """Deterministic (u, v) at a continuous position. v is None in 2D."""

    def velocity_field(self) -> Optional[np.ndarray]:
        """(n_x, n_y[, n_z], ndim) grid of velocities — for plotting. Optional."""
        return None

    # Composition sugar (see below)
    def __add__(self, other):   return SumField(self, other)
    def __mul__(self, scalar):  return ScaledField(self, scalar)
    __rmul__ = __mul__
```

Note what's **gone** versus the old `AbstractField`: no `d_max`, no `disp_levels`, no
`sample_displacement`, no `_clip_displacement`, no `get_displacement_pmf*`. A field is now a
small, obvious object.

### Built-in sources

- `SyntheticFlowField` — the current `RFFGPField`, stripped to just the GP/RFF math
  (`reset`, `velocity_at`, `velocity_field`).
- `GriddedFlowField` — real ERA5 data with interpolation. *(later)*
- `DataDrivenFlowField` — a fitted/generative model. *(later)*

### Composition: coupling and forecast error, with no extra classes

Three tiny combinators give you the whole forecast/reality relationship without any
dedicated "coupling" class:

```python
class SumField(FlowField):     # add fields:  a + b
class ScaledField(FlowField):  # scale a field: 0.3 * a
class ZeroField(FlowField):    # always (0, 0) — an explicit "nothing"
```

You express the relationship between `W` and `Ŵ` by **how you build the two fields**, and
correlation comes from **sharing a field object** between them:

```python
shared = SyntheticFlowField(config, sigma=5, lengthscale=10)
error  = SyntheticFlowField(config, sigma=1, lengthscale=2)

observed = shared                 # the forecast
realized = shared + error         # truth = forecast + structured error  (reuses `shared`!)
```

Because `observed` and `realized` reference the **same** `shared` object, after `reset()`
they share its realization → they're correlated exactly as much as `shared` contributes.
Table cells become wiring choices:

```python
# Good forecast:      realized = observed + small_error          (share observed)
# Useless baseline:   observed, realized = two separate fields    (no sharing)
# Partial info:       s = gp(); observed = s + biasA; realized = s + biasB
# Real truth:         realized = GriddedFlowField(...); observed = gp(...)
```

The **structured forecast error is itself a `FlowField`** (correlated, fixed per episode) —
this is the right model for forecast error, and it lives in composition, not in the field
base class.

---

## Layer 2: the simulator (arena)

The arena holds the two fields and owns everything about *how* wind becomes motion.

```python
class GridArena(AbstractArena):
    def __init__(self, realized_field, observed_field, actor, config,
                 max_displacement, process_noise_std=0.0, obs_noise_std=0.0, ...):
        self.realized_field = realized_field    # W: moves the balloon
        self.observed_field = observed_field    # Ŵ: what the agent observes
        self.max_displacement = max_displacement
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std
        ...

    def reset(self, rng_key):
        # Reset each UNIQUE field once → shared sub-fields are drawn once → stay correlated.
        for f, k in zip(_unique_fields(self.realized_field, self.observed_field),
                        jax.random.split(rng_key, _n_unique)):
            f.reset(k)
        ...

    def step(self, action):
        kt, ko, ka, self._rng = jax.random.split(self._rng, 4)
        true_disp = self._displacement(self.realized_field, kt, self.process_noise_std)
        obs_disp  = self._displacement(self.observed_field, ko, self.obs_noise_std)

        self._move(true_disp)                                  # balloon follows W
        self.position = self.actor.step_controllable(self.position, action, ka)
        self.position, self._out_of_bounds = self._enforce_boundaries(self.position)
        self.last_displacement = obs_disp                       # agent observes Ŵ
        ...

    def _displacement(self, field, key, noise_std):
        u, v = field.velocity_at(self.position)
        if noise_std:                                           # optional per-step jitter
            u = u + noise_std * jax.random.normal(key)
            ...
        return self._clip(u, v)                                 # clamp to ±max_displacement
```

Responsibilities that **moved out of the field and into the arena**:

| Concern | Was | Now |
| --- | --- | --- |
| clip bound (`d_max`) | field property | `max_displacement` on the arena (clip + obs-space bounds) |
| per-step noise | inside `sample_displacement` | `process_noise_std` / `obs_noise_std` on the arena (separate for W and Ŵ; default 0) |
| PMF / `disp_levels` | field methods | **deleted** |

### Two kinds of randomness, cleanly separated

- **Structured error** (`W` vs `Ŵ` gap): a `FlowField` added in composition. Spatially
  correlated, fixed per episode. This is the forecast-error model.
- **Per-step noise** (optional): white jitter the arena adds at sample time, separately for
  realized and observed. Off by default; use only if you want measurement/process jitter on
  top of the structured fields.

---

## Changes to make to the existing code

### Add

- `field/flow_field.py` — `FlowField` ABC (above). Replaces `abstract_field.py`.
- `field/composite.py` — `SumField`, `ScaledField`, `ZeroField`.
- (arena) `_unique_fields(...)` helper — dedupe fields reachable from realized + observed by
  `id()` so shared sub-fields reset exactly once.

### Modify

- `field/rff_gp_field.py` → `field/synthetic.py` (`SyntheticFlowField`):
  - Keep: `reset`, the RFF precompute, `velocity_at_point` → rename `velocity_at`,
    `get_mean_displacement_field` → rename `velocity_field`.
  - Delete: `sample_displacement`, `noise_std`, `get_displacement_pmf`,
    `get_displacement_pmf_grid`, the module-level `_compute_1d_pmf_grid`, `d_max`,
    `disp_levels`, `get_mean_displacement`.
- `field/simple_field.py`:
  - Today it samples a fresh uniform displacement *every call* — that's per-step noise, not a
    field, and it violates "deterministic after reset." Replace with a real field, e.g.
    `ConstantDriftField(config, drift)` or a `UniformDriftField` that draws one random drift
    vector at `reset()`. (Or drop it if unused.)
- `arena/grid_arena.py`:
  - `__init__`: take `realized_field`, `observed_field`, `max_displacement`,
    `process_noise_std`, `obs_noise_std` (drop the single `field` / `d_max` / `disp_levels`).
  - `reset`: reset unique fields once.
  - `step`: sample realized → move; sample observed → `last_displacement`.
  - `observation_space`: use `self.max_displacement` for the `±` bounds.
  - Move the clip helper here (from the old field `_clip_displacement`).
- `arena/navigation_arena.py`, `arena/dynamic_sg_arena.py`:
  - `__init__`/`super().__init__`: pass both fields + `max_displacement`.
  - `dynamic_sg_arena.py:step` (it overrides): apply the same two-field sampling.
- `eval/experiment_config.py` and all `tests/viz_*` / `tests/test_*` construction sites:
  - Build two fields (often sharing one), pass `max_displacement` to the arena.
  - Remove `disp_levels` / `noise_std` args from field construction; move `noise` (if wanted)
    to the arena's `*_noise_std`.

### Delete

- All PMF / oracle code: `get_displacement_pmf`, `get_displacement_pmf_grid`,
  `_compute_1d_pmf_grid`, and `disp_levels` everywhere (field base, synthetic, simple).
- Any agent code paths that consumed the PMF (e.g. `dp_agent` / `ap_ssp_agent` PMF calls) —
  out of scope now; leave the agents aside per "not worrying about agents right now," but
  note they will no longer find `arena.field.get_displacement_pmf_grid()`.

---

## Proposed design, after the changes (summary)

- A wind **source** is a `FlowField`: `reset(key)` draws a realization; `velocity_at(p)`
  returns the wind there. That's the entire contract.
- You build forecast/reality relationships by **composing and sharing** fields:
  `realized = observed + error` (correlated, since `observed` is reused), or two independent
  fields (uncorrelated baseline), or different sources for each (e.g. ERA5 truth, synthetic
  forecast). No coupling class, no special cases.
- The **arena** owns the *dynamics*: it samples the realized field to move the balloon and
  the observed field to produce the observation, clamps to `max_displacement`, and optionally
  adds per-step noise. The two-field semantics live in one obvious place: `step`.
- The system has exactly two abstractions to learn (`FlowField`, arena), no PMF, no
  `disp_levels`, and a field is a tiny object — so adding a new wind source (e.g. ERA5) is
  "write one `velocity_at`."

---

## Deferred / open questions

- **`max_displacement`** — confirm it lives on the arena (it bounds the clip and the
  observation space). Could later be a small `StepSpec` config if the arena grows.
- **`SimpleField` replacement** — pick `ConstantDriftField` vs `UniformDriftField` vs delete.
- **Per-step noise** — keep the optional `*_noise_std` knobs, or rely solely on structured
  error fields? Default 0 either way.
- **Correlation ergonomics** — optionally a helper that, given a target correlation, sizes
  the shared vs private field magnitudes for you.
- **Temporal axis** — `velocity_at(position, t)` once time-varying ERA5 lands.
- **Agents / oracle** — out of scope now. If a model-based baseline is ever wanted again,
  reintroduce a transition/PMF helper as a standalone function over a field, not on the field.
