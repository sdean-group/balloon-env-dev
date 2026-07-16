# Plan: Time-Varying Flow Fields (weather evolves *within* an episode)

> **Status (2026-06-24): implemented through reanalysis.** Contract `velocity_at(p, t)` +
> `time_varying` shipped; synthetic GP temporal axis (`lengthscale_t`), renderer threading, and
> reanalysis temporal interpolation (`steps_per_slice`) all done and tested (21 new tests; full
> field+arena suite green at 526 passed). Not yet built: Phase-2 forecast-error-grows-with-`t`,
> data-driven temporal features, and the agent-observation change. See "Decisions" for what's left.

## The one-line idea

Today a `FlowField` is a pure function of **space**, frozen for the whole episode. Make it a
pure function of **space *and* time** — `velocity_at(p, t)` — and let the arena drive `t` forward
each step. The wind the balloon flies through then *evolves during the flight*, like real weather.

## Why

1. **Real flights outlive the weather.** A station-keeping balloon flies for hours/days; the wind
   genuinely changes underneath it. A frozen-per-episode field is the least realistic part of the
   current sim. (Loon's BLE varies wind in-episode for exactly this reason.)
2. **Forecast error is fundamentally temporal.** The whole observed/realized (`Ŵ`/`W`) design exists
   to model forecast error — and forecasts degrade *with lead time*. With no time axis, the
   forecast/reality gap is fixed for the episode; it can never *grow*. Time is the axis that unlocks
   the feature the two-field architecture was built for.

> **Two distinct things, kept separate.** (a) the *field* evolving (weather changes) and (b) the
> *forecast diverging from truth* over time. Both are temporal but different mechanisms. This plan
> builds (a) — the foundation. (b) is Phase 2, and it *needs* (a) to exist first.

## The core design decision: how `t` enters

The `FlowField` contract is `velocity_at(position)` — no time. The clean, in-keeping move is to make
the field a pure function of `(space, time)`:

```python
def velocity_at(self, position: GridPosition, t: float = 0.0) -> (u, v|None): ...
```

- **`t` is elapsed simulation steps since `reset`** (a float, starts at 0). The arena already owns
  `step_count` ([grid_arena.py:123](../src/env/arena/grid_arena.py#L123)); it passes that in.
- **Default `t=0.0`** keeps the migration painless: every existing call site and test that says
  `velocity_at(pos)` still type-checks and reproduces today's behavior exactly.
- **Each field owns its own time-scale** (`lengthscale_t` for the GP, `steps_per_slice` for ERA5).
  The arena passes raw step count and stays ignorant of weather timescales — minimal coupling.
  (The arena doesn't know the horizon anyway; `max_steps` lives in `GridEnvironment`.)

**Rejected alternative — a stateful `tick()`** that mutates the field mid-episode. It breaks
"deterministic *function* of position," wrecks reproducibility-as-a-function, and makes tests
stateful. Keep fields pure: same `(key, p, t)` → same velocity, always.

**Why raw steps, not normalized phase `[0,1]`:** normalization needs the horizon, which the field
(and arena) don't have. Raw steps keep `ℓ_t` meaningful regardless of episode length. (If we later
want physical units, multiply by a per-field `dt` — but not needed for v1.)

## What changes, file by file

| Layer | File | Change |
|---|---|---|
| **Contract** | [flow_field.py](../src/env/field/flow_field.py) | add `t=0.0` to `velocity_at` / `velocity_field`; add `time_varying` property (default `False`) |
| Leaf: GP | [synthetic.py](../src/env/field/synthetic.py) | add temporal frequency `ω_t` (`lengthscale_t`); thread `t` into `θ` |
| Leaf: ERA5 | [reanalysis.py](../src/env/field/reanalysis.py) | interpolate over a **time axis** between slices (`steps_per_slice`) |
| Leaf: fitted | [data_driven.py](../src/env/field/data_driven.py) | v1: accept & ignore `t` (static); document |
| Leaf: simple | [simple_field.py](../src/env/field/simple_field.py) | accept & ignore `t` (time-invariant) |
| Composites | [composite.py](../src/env/field/composite.py) | thread `t` through; `time_varying = any(child)` |
| Arena | [grid_arena.py](../src/env/arena/grid_arena.py) | pass `t=step_count` into `_displacement` → `velocity_at` |
| Renderer | [navigation_renderer.py](../src/env/rendering/navigation_renderer.py) | pass current `t` into `velocity_field` / `velocity_at` |

### 1. The contract — [flow_field.py](../src/env/field/flow_field.py)
```python
@abstractmethod
def velocity_at(self, position, t: float = 0.0): ...
def velocity_field(self, t: float = 0.0): return None
@property
def time_varying(self) -> bool: return False   # leaves override to True when they evolve
```
`sub_fields()` and `reset()` are unchanged. `time_varying` is a cheap signal for tests, the renderer
(should it re-draw the quiver each frame?), and future optimizations.

### 2. Synthetic GP — the elegant case ([synthetic.py](../src/env/field/synthetic.py))
Time becomes **one extra frequency axis**. Today (`_precompute_field`, `velocity_at_point`):
```
θ = x·ωₓᵀ + φ
```
Add a temporal frequency `ω_t` (shape `(L,)`), drawn at `reset` from a 1-D spectral density with its
own lengthscale `ℓ_t` (RBF: `ω_t ~ N(0, 1/ℓ_t²)`; or reuse the Matérn sampler in 1-D):
```
θ = x·ωₓᵀ + t·ω_t + φ
```
- A draw still fixes `ω, φ, w` **once** at `reset`; `t` is just another input to the same cosine sum.
  Result: a sample from a **spatiotemporal** GP, continuous and reproducible, with **one new knob
  `ℓ_t`** = how fast weather evolves (large = slow drift). `ℓ_t = ∞` recovers today's frozen field.
- **3D stays divergence-free.** The curl `u=-∂ψ/∂y, v=∂ψ/∂x` is over *spatial* derivatives; `t` only
  rides inside `θ`, so `∂/∂y, ∂/∂x` still bring down `ω_y, ω_x` exactly as now
  ([synthetic.py:160-171](../src/env/field/synthetic.py#L160)). Divergence-free holds at every fixed `t`.
- `_precompute_field` can no longer cache the whole episode. Options: (a) cache the `t=0` grid for a
  cheap default and recompute when `t` differs, or (b) make it `_field_at(t)` and compute on demand.
  `velocity_at_point` already recomputes per call — just add `t` to `r`/`θ`.

### 3. Reanalysis — temporal interpolation for free ([reanalysis.py](../src/env/field/reanalysis.py))
The cached array `winds (T, n_x, n_y[, n_z], C)` **already has a time axis** — we currently throw it
away by freezing on `winds[t0]` at reset. Instead, build **one `RegularGridInterpolator` over
`(time, *space)`** and let episode time index into it:
```python
# reset: pick start slice t0 (random/fixed, as today), then build a time-aware interpolator
time_axis = np.arange(self._T, dtype=float)                       # data-slice coordinate
self._interp_u = RegularGridInterpolator((time_axis, *self._axes), self._winds[..., 0], ...)

# velocity_at(p, t): map episode step -> fractional data-slice, then interpolate
s = t0 + t / self.steps_per_slice                                # fractional slice position
s = min(s, self._T - 1)                                          # clamp at end of window (decision)
pt = [[s, p.i, p.j(, p.k)]]
u = self._interp_u(pt)
```
- New param **`steps_per_slice`**: how many env steps span one ERA5 slice interval (sets evolution
  speed). One interpolator does **bilinear/trilinear in space AND linear in time** in a single call.
- Reset still selects the start slice `t0` (the realization) — `slice_mode` semantics unchanged.
- **End-of-window policy** is a decision (below): clamp on the last slice, wrap, or require
  `T ≥ t0 + horizon/steps_per_slice`.

### 4. Fitted & simple fields
- **`DataDrivenFlowField`**: v1 accept and ignore `t` (`time_varying = False`); note "static in time,
  temporal features TODO" — the same `ω_t` trick applies later.
- **`ConstantDriftField` / `UniformDriftField`**: time-invariant by definition — accept and ignore
  `t`, `time_varying = False`. No behavior change.

### 5. Composites — pure plumbing ([composite.py](../src/env/field/composite.py))
```python
def velocity_at(self, position, t=0.0):
    ua, va = self.a.velocity_at(position, t)
    ub, vb = self.b.velocity_at(position, t)
    ...
@property
def time_varying(self): return self.a.time_varying or self.b.time_varying
```
`reset` stays a no-op; `sub_fields`/`unique_fields` reset walk is untouched. Sharing-for-correlation
still works — now the shared leaf is correlated *across space and time*.

### 6. Arena — drive the clock ([grid_arena.py](../src/env/arena/grid_arena.py))
`step` samples both fields at the current position; add the current time:
```python
def step(self, action):
    ...
    t = float(self.step_count)              # 0 on first step; both fields see the same t
    true_disp = self._displacement(self.realized_field, true_key, self.process_noise_std, t)
    obs_disp  = self._displacement(self.observed_field, obs_key,  self.obs_noise_std,  t)
    ...
    self.step_count += 1

def _displacement(self, field, key, noise_std, t):
    u, v = field.velocity_at(self.position, t)
    ...
```
That's the entire arena change — `step_count` already exists and is already part of `GridArenaState`,
so nothing else in state/serialization moves.

### 7. Renderer ([navigation_renderer.py](../src/env/rendering/navigation_renderer.py))
`renderer.step(state)` receives a state carrying `step_count`. Thread it into the quiver builder so
each frame shows the field *at that time*: `velocity_field(t=state.step_count)` (line 319) and the
pointwise fallback `velocity_at(pos, t=state.step_count)` (line 336). For time-varying fields the
quiver should be rebuilt per frame; for static fields (`time_varying == False`) it can be cached.

## ⚠️ The non-obvious correctness issue: does the agent need to *see* `t`?

Once the field is non-stationary, the optimal action depends on **when** you are, not just where.
A Markov agent whose observation is `[i, j, k, u_obs, v_obs]`
([environment.py:65](../src/env/environment.py#L65)) can no longer be optimal in principle — the
environment became time-inhomogeneous. This is a real RL-correctness consequence, not a detail.

Options (decide before training on this): (a) add normalized episode time to the observation; (b) add
a short history/frame-stack; (c) accept partial observability (often fine for station-keeping with a
good local forecast). **Recommendation: expose episode time in the observation behind a flag**, so we
can A/B it. This is an *environment* change, separate from the field work, but flag it now.

## Testing

Back-compat first: because `t` defaults to `0.0`, **the entire existing suite stays green unchanged** —
that's the migration safety net. Then add temporal tests (style mirrors the existing field tests):

| File | Covers |
|---|---|
| `test_synthetic.py` (extend) | `t=0` matches today (golden); continuity in `t`; **divergence-free preserved at `t>0`** (3D); determinism `(key,t)`; `ℓ_t→∞` ≈ frozen |
| `test_reanalysis.py` (extend) | velocity at fractional data-time = exact linear blend of the two bracketing slices; `t=0` = slice `t0`; clamp/wrap at window end |
| `test_composite.py` | `time_varying` propagates; `t` threads through Sum/Scaled |
| `test_field_temporal_integration.py` (new) | passive balloon in `NavigationArena`: trajectory **evolves over the episode** and is reproducible across two identical seeds |

Acceptance bar:
- `velocity_at(p)` (no `t`) and `velocity_field()` behave exactly as today (golden snapshots).
- A passive balloon's per-step displacement changes over an episode and is seed-reproducible.
- GP 3D divergence-free at arbitrary `t`. Reanalysis time-interp is exact linear blend.
- Time-invariant fields are provably unaffected by `t`.

## Build order (each step keeps the suite green)

1. **Contract + plumbing, no behavior change.** Add `t=0.0` to `velocity_at`/`velocity_field` on the
   base and *every* impl + composite + arena threading; all fields ignore `t`. Suite stays green.
   (Pure mechanical change — lands first, de-risks everything.)
2. **Synthetic GP temporal axis** (`ω_t`, `ℓ_t`). First real in-episode variation. New tests.
3. **Renderer** threads `t`; eyeball an evolving quiver (clone `viz_passive_drift.py`).
4. **Reanalysis** time-axis interpolation (`steps_per_slice`, end-of-window policy).
5. *(optional)* **Data-driven** temporal features (same `ω_t` trick).
6. *(Phase 2 — the payoff)* **Forecast error that grows with `t`.** A time-dependent combinator —
   e.g. `ScaledField` accepting a schedule `s(t)`, or a new `TimeRampField` — so
   `realized = observed + s(t)·error` widens the `Ŵ`/`W` gap with lead time. Falls out of this
   architecture once (a) exists.
7. *(separate track)* Add episode-time to the **observation** behind a flag (the correctness item).

## Decisions

**Locked (2026-06-24):**
- **Units of `t` = raw elapsed steps** (float, starts 0; arena passes `self.step_count`).
- **Time-scale ownership = per-field** (`lengthscale_t`, `steps_per_slice`).
- **Back-compat = `t=0.0` default** on the contract; existing call sites/tests untouched.
- **Scope for this pass = through reanalysis** (contract + plumbing → GP → renderer → reanalysis).
  *Not* in this pass: Phase-2 forecast-error-grows-with-`t`, data-driven temporal features, the
  agent-observation change.

**Still open (defer until the above lands):**
- **Reanalysis end-of-window** — clamp on last slice (default for now), wrap, or require enough
  slices for the horizon.
- **Agent observation** — expose episode time (behind a flag) vs accept partial observability.
  Tracked as a separate environment-level change; flagged, not built here.
