# Canonical Factor-Graph Diffusion: Design

## Objective

Define a new lazy infinite-field sampler around the existing frozen conditional wind
denoiser. The target is not numerical agreement with InfiniteDiffusion. The target is a
different random field with per-step overlap communication, fixed random-access work,
and no recursively expanding denoising pyramid.

## State And Factors

The shared normalized chart state at EDM level `s` is

```text
J_s: (channel, time, y, x).
```

A factor `i` is a standard model window. `P_i` extracts its state:

```text
x_s^i = P_i J_s.
```

The unchanged model predicts a clean local block:

```text
x0_hat^i = D_theta(x_s^i, sigma_s, condition_i).
```

The local EDM ODE direction is

```text
d_s^i = (x_s^i - x0_hat^i) / sigma_s.
```

All factor directions are placed into chart coordinates and reconciled:

```text
d_bar_s = sum_i P_i^T(W_i * d_s^i) / sum_i P_i^T(W_i).
```

The chart takes one global Heun step with `d_bar_s`. The predictor state is evaluated by
all factors again at the next sigma before the corrected state is accepted. Consequently,
all factors at the next EDM level read the same overlap values.

## Canonical Atlas

Charts are indexed by integer triples `(k_t, k_y, k_x)`. Their core and halo geometry is
fixed by `ChartConfig`. A chart's support, factor locations, noise, conditioning, weights,
and EDM schedule are pure functions of this key and the global seed.

For default geometry:

```text
core:       2 x 64 x 64
halo:       1 x 32 x 32 on each side
support:    4 x 128 x 128
model:      4 x 64 x 64
factors:    1 x 3 x 3 = 9
```

Neighboring chart supports overlap. Their clean outputs are fused by deterministic chart
weights that taper through each halo. For coordinate `q`:

```text
J(q) = sum_k a_k(q) J_k(q) / sum_k a_k(q).
```

Only charts whose support intersects a query are generated. Final normalized chart states
are held in an LRU cache; eviction changes performance but not values.

## Provenance

### MultiDiffusion

Inherited mechanism:

- One shared canvas state.
- Parallel crop-level diffusion updates.
- Weighted least-squares reconciliation of overlapping updates.
- A frozen reference model with no fine-tuning.

CFGD adapts this from a bounded image to each bounded canonical space-time chart and fuses
EDM directions inside a global Heun update.

Reference: Bar-Tal et al., *MultiDiffusion: Fusing Diffusion Paths for Controlled Image
Generation*, ICML 2023, https://proceedings.mlr.press/v202/bar-tal23a.html.

### DiffCollage

Inherited abstraction:

- Large content represented by local generative factors.
- Overlap values represented as shared variables.
- Parallel rather than autoregressive factor evaluation.

The first implementation uses MultiDiffusion's closed-form consensus. It does not claim
to reproduce the complete DiffCollage score-composition method.

Reference: Zhang et al., *DiffCollage: Parallel Generation of Large Content with Diffusion
Models*, CVPR 2023, https://arxiv.org/abs/2303.17076.

### InfiniteDiffusion

Inherited procedural interface:

- Integer coordinates over an unbounded domain.
- Coordinate-keyed initial noise.
- Lazy query evaluation.
- Weighted overlap accumulation.
- Caching, exact revisits, and seed consistency.

Replaced mechanism:

- InfiniteDiffusion recursively queries previous diffusion layers and can expand the
  required window pyramid.
- CFGD materializes one fixed chart state and performs every diffusion level inside that
  bounded state.

Reference: Goslin, *InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility
for Open-World Terrain Generation*, SIGGRAPH 2026,
https://xandergos.github.io/terrain-diffusion/.

## Mechanical Invariants

The implementation and toy tests enforce:

1. Equal coordinates receive equal initial noise in every chart.
2. Factor weights cover every chart variable with positive total weight.
3. Chart selection depends only on query coordinates.
4. Chart generation depends only on chart key, seed, geometry, model, and conditioning.
5. Atlas accumulation uses a stable sorted chart order.
6. Repeating or recomputing a chart returns the same tensor.
7. A subquery equals the corresponding crop from a larger query.
8. Model work per chart is fixed:

```text
number_of_factors * (2 * number_of_EDM_steps - 1).
```

Model batching changes wall time, not this logical evaluation count.

## Claims Not Yet Established

The following require real-checkpoint experiments and should not be stated as results:

- Better ERA5 realism than InfiniteDiffusion.
- Better seam metrics.
- Better wall time on MPS or CUDA.
- Preservation of the base model's exact local marginal distribution.
- Long-range meteorological coherence beyond chart support.
- A formal continuous-field seam theorem for discrete wind arrays.

CFGD is a locally finite atlas, not an exact solution of one globally connected infinite
factor graph. Chart support fixes its maximum interaction range.
