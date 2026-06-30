# Wind-Field Generation for Balloon RL — Benchmark + Trained Generator

A reference-free, two-axis **benchmark** for wind generators, and the first model to clear it: a
trained **InfiniteDiffusion** generator producing unbounded, seamless, reproducible stratospheric
winds. Scores ∈ [0,1], higher = better; **N/A** = capability not declared (not a penalty).

## Headline — two axes

| | ERA5 (truth) | **InfiniteDiffusion (trained)** | phase-shuffle | BLE-VAE | toy | white noise |
|---|---|---|---|---|---|---|
| **Axis-1** — field quality | 0.93 | **0.66** | 0.57 | 0.50 | 0.48 | 0.00 |
| **Axis-2** — procedure | N/A | **0.88** | N/A | N/A | 0.99 | N/A |

The trained model beats every baseline except real ERA5 on field quality, and is the only generator
that satisfies the infinite-generation procedural guarantees (the toy is a no-ML stand-in used to
validate the machinery).

---

## Axis-1 — field quality (reference-free, grid-independent)

Each field is scored against *physically-motivated ideals*, no ground truth needed. **COMPOSITE =
mean(score: spectrum, score: intermittency)** — both reference-free, in [0,1].

| Component | ERA5 | trained | toy | phase-shuffle | BLE-VAE | noise |
|---|---|---|---|---|---|---|
| **score: spectrum** *(in composite)* | 1.00 | 1.00 | 0.84 | 1.00 | 0.00 | 0.00 |
| **score: intermittency** *(in composite)* | 0.85 | 0.32 | 0.12 | 0.13 | 1.00 | 0.00 |
| **= COMPOSITE** | **0.93** | **0.66** | **0.48** | **0.57** | **0.50** | **0.00** |
| spectrum slope (≈ −3) · *diagnostic* | −3.92 | −3.53 | −2.52 | −3.92 | 1.67 | 0.05 |
| increment kurtosis (>3) · *diagnostic* | 5.56 | 3.96 | 3.35 | 3.39 | 13.9 | 3.00 |
| vort/div ratio (>1) · *diagnostic* | 0.77 | 0.93 | 3.76 | 0.97 | 59.5 | 0.96 |
| amplitude vs peer · *diagnostic, ref-gated* | 1.00 | 0.48 | 0.70 | 1.00 | 0.52 | 1.00 |

- **score: spectrum** — energy-spectrum slope vs the ≈ −3 synoptic-turbulence ideal (turned into a
  [0,1] score). Trained: solved (slope −3.53).
- **score: intermittency** — velocity-increment kurtosis vs ERA5's (the fat-tailed gusts/fronts real
  wind has). Trained's main remaining gap (0.32 vs 0.85): right scale structure, too-smooth extremes.
- **Diagnostics (reported, NOT in the composite):** *vort/div ratio* (rotational-vs-divergent
  balance — real wind is rotation-dominated; this and kurtosis are what a **calibration** showed
  actually discriminate, since phase-shuffled noise fakes the spectrum); *amplitude* (RMS wind speed
  vs the peer — kept out of the composite because it needs a reference, and the structural scores stay
  reference-free). ERA5 sits as a strong *peer row* on the same scale, not "the answer."

## Axis-2 — the procedure (capability-gated)

The claims only *some* generators make. **PROC COMPOSITE = mean of the four capability scores below**,
computed only for generators that declare these capabilities; bounded generators (ERA5, VAE, anchors)
are **N/A**, which is what lets fixed-crop and infinite generators share one leaderboard honestly.

| Component (all in composite) | trained | toy |
|---|---|---|
| **seam** (1 = seamless tiling) | 0.87 | 0.97 |
| **revisit** (1 = exact seed-determinism; max\|Δ\| = 0) | 1.00 | 1.00 |
| **budget / O(1)** (far-query ≈ near-query cost) | 0.94 | 0.98 |
| **extent** (1 = no quality drift as the crop grows) | 0.69 | 0.99 |
| **= PROC COMPOSITE** | **0.87** | **0.99** |

- **seam** — divergence/value continuity across tile-stitch boundaries (no artificial edges).
- **revisit** — same (seed, coords) re-derives bit-identical winds (Δ = 0): reproducible worlds.
- **budget / O(1)** — querying a far location costs the same as a near one (lazy, constant-time).
- **extent** — field quality stays flat as you sample larger regions (no large-scale repetition/drift).

The point this proves: swapping a *learned* denoiser into the unmodified machinery **keeps** all four
guarantees — the model owns Axis-1, the machinery owns Axis-2, and they're independently testable.

## Axis-3 — temporal realism (in progress)

The wind field must also **evolve like weather** over a multi-day episode. The metric is **`score:
temporal realism` = mean(persistence-match, tendency-match)**, *peer-matched* to ERA5 — real evolution
lives in a band, so **too-frozen *and* too-chaotic are both penalized** (tendency-match catches frozen;
persistence-match catches incoherent) — plus **`drift`** (spatial COMPOSITE vs lead time; should stay
flat). Validated ranking so far: ERA5 1.00 > kinematic-toy 0.53 (too frozen) > shuffled 0.34.

> Key physical finding the benchmark surfaced: **wind patterns aren't passive tracers** — the jet is
> quasi-stationary (air flows *through* it), so advecting ERA5 by its mean wind predicts the next
> frame *worse* than persistence. So the advective baseline is a deliberately-naive *floor*, not a target.

**Methods being compared (both learned routes are built and training on the GPU cluster):**
- **M1 kinematic/advective** — carry the static field by the mean wind. No learning; the naive floor.
- **M3 autoregressive** (GenCast-style) — learn `p(frame_{t+1} | frame_t)` and roll forward. Standard
  for weather; cheaper; cost = O(t) rollout + drift accumulation.
- **M2 joint spacetime** — denoise an H×W×τ block jointly (factorized 2D-space + 1D-time); novel here,
  preserves the O(1)/seamless guarantees *in time*; heavier.
- *Parked:* M4 (noise/latent evolution) and M5 (advection + learned residual — a strong later option).

A 5-row temporal leaderboard (ERA5 / M3 / M2 / kinematic-toy / shuffled) is wired up; the M3/M2 rows
fill in once their training checkpoints land.

---

## Design context (the choices behind the numbers)

- **Benchmark first.** Candidate generators differ too much to rank on one number (bounded vs
  unbounded, deterministic vs stochastic), and "accuracy to one ERA5 snapshot" is the wrong target for
  a *generative* model — hence reference-free Axis-1 + capability-gated Axis-2.
- **Two separable halves.** *Machinery* (lazy MultiDiffusion blending over an infinite lattice → Axis-2)
  vs *model* (the window denoiser → Axis-1). The machinery was validated with an analytic toy denoiser
  *before* training anything, then the real EDM model swapped in with zero machinery changes.
- **The trained model.** EDM-style U-Net (pixel space) with vertical levels as channels (u,v × 18
  model levels), per-(level,variable) normalization (stratospheric wind variance swings ~10× with
  altitude); trained on ERA5 model-level crops (NE Pacific, levels 49–66) on the Cooper Union "Kahan"
  cluster.
- **Generator-family choice.** Of the three families — autoregressive outpainting (drifts), joint
  diffusion (MultiDiffusion → **InfiniteDiffusion**, chosen for seamless/seed-consistent/O(1)), and
  hierarchical coarse→fine (CorrDiff-style) — we built the InfiniteDiffusion fine layer first. The
  **coarse→fine conditioning layer** (lat/lon/season + a synoptic field, à la CorrDiff/Terrain
  Diffusion) is the planned next layer and is also where location/time conditioning enters.
- **Truth ≠ observation.** The generated field is the *truth* the balloon flies through; a **separate**
  forecast-error model (what the agent *sees*) layers on top — folding them together would destroy the
  partial observability that makes the RL task interesting.

## Known gaps → next steps (each is metric-driven)
- **Intermittency 0.32 → 0.85** (the main field-quality gap) and **low amplitude** — both point to
  *more training + more diverse data* (the run was cut at 84k steps on 4 seasonal weeks), not a sampler
  change (a stochastic sampler was tested and made the blended field worse).
- **Mild extent drift (+0.063/octave)** — fixable with restart-style blending iterations.
- Then: **coarse conditioning**, and **time/advection** (the real research target — M2/M3 above).

---

## Reproduce

```bash
# from balloon-env-dev/, full pixi env (jax + torch + infinite-tensor):
PY=".pixi/envs/default/bin/python"
PYTHONPATH=. $PY -m src.eval.windeval.benchmark_m1 --ckpt runs/idiff_m1/step_84000.pt   # spatial, both axes
PYTHONPATH=. $PY -m src.eval.windeval.benchmark_temporal --m3-ckpt … --m2-ckpt …        # temporal leaderboard
```

Figures: `docs/figures/` (trained field + seam overlay, ERA5/trained/toy comparison, zoom montage).
Full reports: [`benchmark_m1_report.md`](../src/eval/windeval/benchmark_m1_report.md),
[`benchmark_temporal_report.md`](../src/eval/windeval/benchmark_temporal_report.md). Build log +
design decisions: `infinite-diffusion-progress.md`.
