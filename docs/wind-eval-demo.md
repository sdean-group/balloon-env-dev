# Generating Wind Fields for Balloon RL — a Benchmark, and a First Trained Generator

*A wind-field **evaluation harness** plus the first generator to clear it end-to-end: a trained
**InfiniteDiffusion** model that produces unbounded, seamless, reproducible stratospheric winds.*

---

## TL;DR

- **The hard part isn't building a wind generator — it's knowing which one is *good*.** I built a
  reference-free, two-axis **benchmark** that scores any wind generator without needing it to match
  a single "true" field, and that fairly compares generators with very different capabilities.
- I then built **InfiniteDiffusion**: a diffusion model that generates an *infinite, seamless*
  wind field you can query anywhere, trained on real ERA5 reanalysis (on the Cooper Union GPU cluster).
- **Result:** the trained model **beats every baseline except real ERA5** on field quality, while
  being the only generator that satisfies the "infinite-generation" procedural guarantees.

| | ERA5 (truth) | **InfiniteDiffusion (trained)** | phase-shuffle | BLE-VAE | toy | white noise |
|---|---|---|---|---|---|---|
| **Axis-1** — field quality | 0.93 | **0.66** | 0.57 | 0.50 | 0.48 | 0.00 |
| **Axis-2** — procedure | N/A | **0.88** | N/A | N/A | 0.98 | N/A |

---

## 1. The problem

Training/evaluating a balloon RL agent needs *wind fields* — and lots of them. A balloon doesn't
sit still; it drifts for days across hundreds of km and rides winds that change with altitude. So a
useful wind generator has to:

- produce **many plausible** fields (for RL diversity), not one fixed map;
- cover **unbounded extent** (the balloon wanders off any fixed crop);
- be **seamless** (no artificial discontinuities the agent could exploit);
- be **reproducible** (the same seed → the same world, for fair eval).

There are several ways to build such a generator — and that's exactly the problem.

## 2. Why I built the benchmark *first* (the real design challenge)

When picking a generator, the candidates differ so much that you **can't rank them on one number**:

- Some are **bounded** (a fixed ERA5 crop, a VAE sample); some are **unbounded** (tile-based).
- Some are **deterministic** interpolations; some are **stochastic** samplers.
- "Accuracy to ERA5" is the wrong target — a *generative* model should make **plausible** weather,
  not memorize one reanalysis snapshot. Penalizing it for not matching a specific field is backwards.

So before committing to any generator, I built an **evaluation harness** with two orthogonal axes:

**Axis-1 — field quality (reference-free, grid-independent).** Each field is scored against
*physically-motivated ideals*, no ground truth needed:
- **energy spectrum slope** (≈ −3 for synoptic turbulence),
- **rotational vs divergent balance** (real wind is rotation-dominated),
- **velocity-increment kurtosis** (intermittency — the fat-tailed gusts/fronts real wind has).
ERA5 is just a strong *peer row* on the same scale, not "the answer."

> A calibration finding shaped this: the obvious metrics (power spectrum, autocorrelation) are
> **fooled by phase-shuffled noise** — it has the right spectrum but no real structure. Only the
> vorticity/divergence balance and increment kurtosis (which probe *cross-structure* and
> *non-Gaussianity*) actually discriminate. Those gate the score.

**Axis-2 — the procedure (capability-gated).** The claims only *some* generators make: **seamless**
tiling, **seed-determinism**, **O(1) random access**, and **no quality drift with extent**. Bounded
generators simply don't declare these capabilities, so they score **N/A — not penalized**. This is
what lets a fixed-crop VAE and an infinite tiler live on the same leaderboard honestly.

## 3. The generator landscape the benchmark lets me compare

The benchmark turns "which generator should I build?" into an evidence-based decision. The field:

| Generator | Type | Extent | Status |
|---|---|---|---|
| **ERA5 reanalysis** | real data | bounded | ceiling / truth peer |
| **BLE-VAE** | learned (VAE) | bounded | baseline (team) |
| Real-wind **linear interpolation** | interpolation | bounded | planned ([plan](plan_real_wind_linear_interp.md)) |
| Data-driven **Gaussian process** | probabilistic | bounded | planned ([plan](plan_data_driven_gp.md)) |
| **Temporal** evolving fields | — | — | planned ([plan](plan_temporal_fields.md)) |
| **InfiniteDiffusion** | diffusion | **unbounded** | **implemented + trained (this week)** |
| phase-shuffle / white-noise | anchors | bounded | lower-bound controls |

InfiniteDiffusion is the candidate I bet on for the "unbounded + seamless + reproducible"
requirements — the benchmark is what told me whether that bet paid off.

## 4. InfiniteDiffusion — what it is and how I built it

**The idea** (after Goslin's *Terrain Diffusion / InfiniteDiffusion*): run a diffusion model as a
**blending wrapper** over an *infinite lattice*. A normal diffusion model denoises a fixed-size
window; InfiniteDiffusion tiles overlapping windows across an unbounded grid and fuses them with a
weighted average, evaluating *lazily* — only the windows your query touches. That's what makes the
field infinite, seamless, and O(1) to sample anywhere.

**The design decision that made it tractable — two independent halves:**

- **Machinery** (the blending wrapper) → owns the **Axis-2** properties (seamless, deterministic, O(1)).
- **Model** (the window denoiser) → owns the **Axis-1** field quality.

These are **separable and independently testable**. So I validated the machinery *first* with an
**analytic toy denoiser** (a divergence-free field with a tuned spectrum — no ML), proving the
infinite-generation guarantees before training anything. Then I trained the **real** denoiser and
**swapped it in with zero changes to the machinery**.

**The trained model:** an EDM-style U-Net (Karras et al.) in pixel space, with the vertical levels
carried as channels (u,v × 18 model levels) and **per-(level,variable) normalization** (stratospheric
wind variance swings ~10× with altitude). Trained on **ERA5 model-level crops** (4 seasonal weeks of
2023 over the NE Pacific, levels 49–66 ≈ 50–140 hPa) on the **Cooper Union "Kahan" GPU cluster**
(RTX PRO 6000), to 84k steps.

## 5. Results

**The trained field looks like real wind, and it's seamless.** Left: speed + direction. Right:
vorticity with the tile-stitch grid overlaid — *a broken tiler shows streaks along those dashed
lines; ours shows none.*

![Trained InfiniteDiffusion wind field](figures/fig_trained_field.png)

**Side-by-side structure.** The trained field shares ERA5's smooth, large-scale character; the
analytic toy's divergence-free filaments look obviously synthetic by comparison. *(Each panel scaled
to its own range — see the amplitude caveat below.)*

![ERA5 vs trained vs toy](figures/fig_compare.png)

**It's genuinely infinite.** Same seed, zooming in — multi-scale coherence and free random access
into an unbounded field (the red box marks the next zoom):

![Zoom montage](figures/fig_zoom_montage.png)

**Reading the leaderboard:**

- **Axis-1 (0.66):** beats the toy, BLE-VAE, and even phase-shuffle; trails only the ERA5 ceiling
  (0.93). The **energy spectrum is solved** (score 1.00, slope −3.53). The remaining gap is almost
  entirely **intermittency** (0.32 vs ERA5's 0.85) — the model gets the scale structure right but
  under-produces sharp gusts/fronts.
- **Axis-2 (0.88):** swapping a *learned* denoiser into the machinery **keeps** the guarantees —
  **exact seed-determinism** (repeat query Δ = 0), seamless tiling (0.87), O(1) random access. This
  validates the two-halves design end-to-end: a real model inherits the infinite/seamless/deterministic
  properties for free.

## 6. Known limitations → and where this goes next

The benchmark didn't just rank the model — it pointed at exactly what to fix:

- **Amplitude is low — now measured and decomposed.** I added a (scale-relative) **amplitude
  metric** to Axis-1 — RMS wind speed vs the peer, kept out of the reference-free COMPOSITE.
  Investigating the apparent "≈10 vs ≈30 m/s" gap showed it's **two effects**: (1) the eval peer
  (`era5_real`, 33–42°N) is a genuinely jet-stronger region than the model's *training* climatology
  (`era5_train`, 25–55°N + seasonal averaging) — a perfect model would still only score **0.70** vs
  the peer (an unavoidable climatology offset); and (2) a real **generation defect** — the
  deterministic ODE sampler recovers only ~42% of the anomaly variance (under-dispersion), so the
  model hits ~0.69 of its *own* target amplitude. I tested the obvious no-retrain fix (a
  window-seeded **stochastic sampler**): in isolation it helps, but on the *blended* generator the
  MultiDiffusion average smooths the per-window stochasticity away and amplitude/intermittency get
  *worse* — so the real fix is **more training + more diverse data** (the run was cut off at 84k on
  4 weeks), not a sampler change. The stochastic sampler is kept off by default (it does improve
  **extent drift** 0.69→0.90, a separate gap).
- **Intermittency gap (0.32 → 0.85).** The biggest field-quality gap; the candidate fix is a loss
  re-weighting / noise-schedule change (or simply more, more-diverse data — this used only 4 weeks).
- **Mild extent drift (+0.063/octave).** Quality varies slightly with crop size; fixable by adding
  blending iterations with a restart-style refinement.

Each of these is a **metric-driven** next step — which is the whole point of building the benchmark
first. After that: coarse weather conditioning, then **time/advection** (the real research target).

---

## Reproduce

```bash
# from balloon-env-dev/, full pixi env (jax + torch + infinite-tensor):
PY=".pixi/envs/default/bin/python"

# two-axis leaderboard (trained model):
PYTHONPATH=. $PY -m src.eval.windeval.benchmark_m1 --ckpt runs/idiff_m1/step_84000.pt

# these figures:
PYTHONPATH=. $PY -m src.eval.windeval.scripts.make_demo_figures --ckpt runs/idiff_m1/step_84000.pt
```

Full reports: [`benchmark_m1_report.md`](../src/eval/windeval/benchmark_m1_report.md) (trained,
both axes) and [`benchmark_m0_report.md`](../src/eval/windeval/benchmark_m0_report.md) (machinery
validation with the toy). Staged build log + design decisions: `infinite-diffusion-progress.md`.
