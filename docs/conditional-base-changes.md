# Conditional 4D base model — change log & design decisions

*Tracking doc for the Phase-5 conditional base-diffusion-model work (started 2026-07-10).
Same contract as `benchmark-v2-changes.md`: every design decision is recorded with who made
it; small decisions Claude takes alone are flagged for push-back. The goal: a good
**conditional space-time (4D) base diffusion model**, trained first, wired into the
InfiniteDiffusion machinery second. Success = moving the two v2 board numbers:
L_eff 673 km → toward 56 km, tail err 0.1% 15.6 → toward 2.9 m/s.*

## Decisions made by Shaurya (2026-07-10)

1. **Temporal route = M2 space-time blocks.** The conditional model denoises whole H×W×τ
   blocks jointly (existing `spacetime.py` factorized architecture); long sequences come
   from tiling in time, extending InfiniteDiffusion's O(1)/lazy/seamless guarantees to the
   time axis. M3 (autoregressive rollout) rejected for drift over day-long episodes.
2. **Conditioning = location + time** (NOT a coarse synoptic field — that stays a future
   option if the tail number doesn't move). This deliberately **supersedes the Phase-2
   "never show the model absolute coordinates" decision**, which was made for the
   unconditional model: the conditional model is *supposed* to learn location-dependent
   climatology so the RL env can request winds at a place and date.
3. **Time encoding = continuous cyclic harmonics, not bins.** Years treated as
   exchangeable ("the same month year to year would have the same weather"). Within the
   year: sin/cos of annual (+ semiannual) phase; within the day: sin/cos of diurnal phase.
   Granularity is *learned*, not imposed — no season/month enum anywhere. (GraphCast /
   GenCast-standard encoding; low-order harmonics also structurally prevent the model from
   memorizing individual days of a single training year.)
4. **Location encoding = per-pixel lat/lon coordinate channels** (two extra clean
   conditioning channels). Chosen over a window-center embedding because it composes with
   InfiniteDiffusion tiling: each tile of a large canvas carries its own coordinates, so
   blended neighbors stay geographically consistent.
5. **Reflection augmentation dropped** for the conditional model. Mirroring the field
   while keeping coordinates teaches false geography; mirroring both trains on a mirrored
   Earth. The larger 2023 dataset replaces the ×4 synthetic augmentation with real weather.
6. **Training data = all of 2023, hourly, days 8–14 of every month EXCLUDED** (the
   benchmark split contract — reference is days 8–14 of Jan/Apr/Jul/Oct 2023). ~281 days
   ≈ 6.7k hourly steps on the NE-Pacific box (25–55°N × 105–135°W), ML 49–66. Multi-year
   deferred until one year plateaus (one change at a time).

## Decisions made by Claude (flagged for review — push back on any of these)

1. **Harmonic set**: annual + semiannual for the yearly cycle (the climatology-standard
   pair), single harmonic for the diurnal cycle → 6 time features per frame:
   `[sin,cos](2π·doy/365.25), [sin,cos](4π·doy/365.25), [sin,cos](2π·hour/24)`.
   Semidiurnal tide (stratospherically real) deferred — add a harmonic only if a temporal
   metric demands it.
2. **UTC hour, not local solar hour**, for the diurnal features. The model has per-pixel
   longitude channels, so local time is a learnable combination (lon/15 h offset);
   feeding UTC keeps the feature independent of the coordinate channels.
3. **Time features enter per-frame through the embedding pathway** (zero-init linear
   added to the per-frame noise embedding), matching the project's zero-init-residual
   pattern: an untrained conditional path is exactly the unconditional model. The hour
   advances frame-to-frame within a τ-block, so features are per-frame, not per-block.
4. **Coordinate channels normalized by training-domain center/half-width** (stored in the
   checkpoint, applied identically at inference; a ±180° branch guard maps inference lon
   into the training convention). Raw lat/90-style scaling would give a ~0.3 dynamic range
   on a regional box — poorly conditioned.
5. **Coordinate channels are concatenated clean (un-scaled by EDM c_in)**, exactly like
   M3's previous-frame conditioning — only the noisy target gets preconditioning math.
6. **No warm-start from the static M1 checkpoint** for the first run — the conditional
   model trains from scratch (M1 was trained with augmentation on 24× less data; a clean
   from-scratch run is the defensible baseline). Warm-start is a recorded option if
   training is too slow.
7. **Download mechanics**: reuse `download_era5_train.py` with 24 MARS date chunks
   (two per month: days 1–7 and 15–end), hourly (`00/to/23/by/1`), grid/area/levels
   identical to the existing artifacts. Runs on the Kahan login node (has internet;
   the CDS queue dominates wall time).

8. **The gate is condition-matched for spacetime checkpoints.** `gate.py` draws N random
   τ-frame reference blocks from the training zarr and samples the model at the SAME
   (lat, lon, times) tuples, then pools the n·τ frames per side. This makes the gate a
   test of the *conditional* distribution, not just the climate. Static checkpoints keep
   the old behavior; the report format is unchanged. For the m2cond run pass
   `--ref src/eval/windeval/data/era5_2023.zarr`.

## Decisions made by Shaurya (2026-07-14) — W1_cond protocol

7. **A W1_cond condition = (location, month, hour-of-day).** Reference pool for a
   condition = that hour on each of the 7 held-out days (8–14) of the month; model pool =
   seeds sampled at those same 7 timestamps. Rationale: the cyclic harmonics are
   bandlimited below day-scale (they *cannot* resolve individual days — by design), so
   day-to-day variability at a fixed (month, hour) IS the within-condition distribution.
   Rejected: exact-timestamp conditions (n=1 reference, conflates ensemble spread with
   spatial variability) and month-climatology pooling (averages the diurnal cycle away —
   couldn't tell whether the diurnal harmonics do anything).

## Decisions made by Claude (2026-07-14, benchmark wiring — flagged for review)

9. **Condition set = 4 months × 2 hours (00/12 UTC) × 7 days × 2 seeds = 112 blocks.**
   Hours 00/12 bracket the diurnal cycle; 2 seeds × 7 days = a 14-member model pool per
   condition. ~8 min of sampling on the Mac (3.9 s/block on MPS). More hours/seeds are a
   knob if W1_cond looks noisy.
10. **Fixed 64² window centered on the reference grid** (lat 32.25–48°N, lon 232–248°E)
   as the single "location" for v1 of the protocol — one location keeps the condition
   count honest; multi-location comes with the tiling work. The window keeps the
   reference's native north-to-south lat ordering (matches training crops).
11. **`--cond-ckpt` CLI flag (default `runs/idiff_m2cond/latest.pt`) adds an
   `idiff m2cond` row** next to the static row — explicit flag, no checkpoint sniffing.
   Both models appear on one board.
12. **Main-row pooling uses seed-0 blocks only** (224 frames with unique real
   timestamps); both seeds feed W1_cond, where times don't collide. Avoids duplicate
   timestamps corrupting the temporal segment detection.
13. **W1_cond frame selection: frame 0 of each block** (the exact condition hour);
   frames 1–3 (h+1..h+3) participate only in the pooled main row.
14. **Temporal metrics for the m2cond row are N/A** — the pooled row carries real
   timestamps, but τ=4-frame segments fall below the suite's own MIN_FRAMES=16 guard
   (correctly: a 4-point periodogram isn't evidence). Temporal tiling unlocks them.
15. **Condition-matched W1_cond floor added to the self-split row** (days 8–10 vs 11–14
   at each (month, hour), same window): 2.38 m/s — the scale to read the model's
   W1_cond against. The static row's W1_cond stays the old degenerate one-condition
   version (different protocol; footnoted as not comparable).

## Decisions made by Claude (2026-07-14, structured-noise baselines — flagged for review)

Shaurya's team asked for simplex-noise (the BLE family) and Helmholtz-kernel-GP
baseline rows. Implementation in `src/eval/windeval/baselines.py`; both rows join the
board next to white noise / phase shuffle. Decisions:

16. **Fitting contract: every knob is fit on held-out half A** (the same half the other
   anchors are derived from), scored vs the full reference. Each row reads as "the best
   this noise family can do" — a trained model must beat structured noise at its best,
   not a strawman. The alternative (BLE's published noise constants for simplex) was
   rejected: those were tuned for forecast *perturbation*, not standalone generation,
   and would understate the baseline.
17. **Knob→statistic mapping (one knob, one reference statistic):** per-level mean/std
   moment-matched (as the white-noise anchor does); vertical/temporal correlation
   scales matched to half A's adjacent-level (0.982) and 4h-lag (0.920) correlations —
   analytically for the GP (SE inverse), empirically for simplex (measured fBm
   autocorrelation transect, interpolated); horizontal scale by coarse log-grid search
   minimizing SR_E (the benchmark's own objective) + one geometric refinement round.
18. **Simplex = octaved fBm** (4 octaves, persistence 0.5, lacunarity 2 — the standard
   stack; single-octave simplex is near-monochromatic and would strawman the baseline).
   4D noise over (x, y, level-input, time-input); u and v from independent instances.
   Fitted horizontal scale: 32 px.
19. **Helmholtz GP: SE spectra for both potentials, one shared horizontal length scale**
   (fitted ℓ=4 px), div/curl energy split from half A's var(div)/var(vort) (0.47 —
   levels 49–66 carry a lot of divergent gravity-wave energy). Sampled spectrally on a
   256² periodic grid, cropped to 121² (no wraparound inside the domain). Level+time
   correlation via Cholesky of K_t ⊗ K_L mixing iid spatial samples (exact for a
   separable kernel). Per-level length scales rejected (breaks separability; recorded
   extension). Isotropic in pixel units (ignores the ~25% lon/lat km anisotropy — same
   approximation the model's crops make).
20. **Dependency added: `opensimplex>=0.4.5`** (pypi, via pixi + requirements.txt).
   numba does NOT accelerate its vectorized API (~17 µs/pt measured) → octave×component
   calls fan out over a ProcessPoolExecutor (8 workers; full simplex row ≈ 6 min,
   GP row ≈ 15 s; both cached as `data/baseline_*.zarr`, `--regen` to refit).
21. **Figure colors CVD-checked by computation** (Machado-matrix simulation, Lab ΔE):
   helmholtz gp #3a1fdf (min ΔE 32 vs all existing under normal/deutan/protan),
   simplex noise #2fbf66 (ΔE 12.7, above the ≥12 floor; contrast 2.4:1 is carried by
   the legend + table). First-pick violet/dark-green FAILED the simulation (ΔE 3 and 1)
   — eyeballing colors is not a thing.

## Finding (2026-07-14): the structured-noise baselines expose what the suite measures

Fitted knobs: simplex scale 44.8 px (≈ 1100 km features), d_lev 0.043, d_4h 0.098;
GP ℓ=4 px, s_lev 5.2 levels, s_time 9.8 h, div_frac 0.47.

**The uncomfortable board:** fitted fBm simplex noise beats BOTH trained models on
every spectral row (SR_E 0.75 / SR_div 0.87 / SR_vort 0.72 vs m2cond's 1.62/1.64/1.64;
L_eff resolved-to-grid 56† vs 337) and both baselines sit at/below the floor on
marginal W1, tails, and shear (moment-matching by construction — the documented
white-noise blind spot, now shown to extend to every population statistic when the
noise also has fitted correlations). Helmholtz GP is worse than simplex spectrally
(SE spectrum can't make a power law; SR_E 3.19) but its tails are floor-level (1.42).
Simplex SR_time 1.19 vs floor 0.28; GP 5.56 (SE time kernel too smooth).

**Read:** population-statistics metrics cannot separate a trained model from noise
fitted to those same statistics — that is exactly what these rows are for. What noise
structurally CANNOT do: conditional realism (W1_cond — baselines have no conditioning),
coherent temporal evolution at scale, and actual weather structures. The models' case
now rests on W1_cond (m2cond 3.26 vs 2.38 floor, and noise can't play) + the tail gap
that conditioning was supposed to close (still open). The metric suite itself may need
a structure-sensitive metric (e.g. coherent-feature or multi-point statistics) — for
Shaurya to weigh.

**Flagged follow-up (needs Shaurya):** score the unconditional rows on W1_cond under
the new protocol (their samples are condition-independent = the climatological
baseline; crop to the center window, pool as "seeds" per condition). That would put a
number on "how much does conditioning buy beyond climatology" — m2cond's 3.26 vs
noise-climatology's X. ~~Not implemented pending his sign-off.~~ **RESOLVED 2026-07-30
(Shaurya approved): implemented as decision 27 below. The answer to "how much does
conditioning buy" is: nothing measurable — see the finding below.**

## Decisions made by Claude (2026-07-29/30, P0 metric work — flagged for review)

Shaurya (post-professor meeting): build fidelity metrics that test real weather structure,
and score simplex on W1_cond instead of leaving N/A. Both shipped. Decisions:

22. **New module `metrics/structure.py`** holding the metrics a matched spectrum does NOT
   fix. Rationale made explicit: every pre-existing row is a population statistic —
   1-point marginals (W1/tails/shear) plus the 2-point covariance (PSD) — and a field
   matched on mean+covariance reproduces all of them by construction. That is *why*
   fitted simplex sits at the floor almost everywhere; it is not a mystery to explain away.
23. **Opposing-wind (vertical decoupling) metric** = fraction of columns with a level
   pair >90° apart, both ≥5 m/s. Borrowed from RL-HAB's "Forecast Score" (Schuler et
   al., NRL, arXiv 2502.05014) rather than invented — matches the standing "defensible
   raw metrics over invented scores" rule. Two rows: `opp-wind frac` (≈ref, ERA5 = 0.201)
   and `opp-wind err` (lower). **`shear.py` provably cannot see this**: it pools
   ADJACENT-level differences per component into one histogram, never forming an angle
   between wind vectors nor spanning a large altitude gap.
24. **Common 64² window for ALL structural metrics.** Non-negotiable: scoring the model's
   64² output against the full 121² reference INVERTS the decoupling verdict (it made
   the model look better than simplex; window-matched the opposite is true). Component
   area is also domain-bounded. Regression-guarded in the test file.
25. **Speed/jet-intensity rows** (`W1 speed`, `jet speed err 99/99.9%`): moment-matching
   u and v does not match |V| or its jet-core tail.
26. **`jet area ratio` shipped as PROVISIONAL, explicitly not to be scored.** Its 1.42
   floor is genuine synoptic variability (days 11-14 hold ~2× as many, smaller
   components as 8-10: 1093 vs 2065), not estimator noise — verified by splitting one
   weather period against itself (area 1.07, range 0.80-1.42) and by checking median/p75
   (no better). `jet elong ratio` IS reliable (within-period 0.97, range 0.93-1.02;
   synoptic floor ~1.17). Fix path: condition-match the geometry as W1_cond does.
27. **Climatological W1_cond for every unconditional row** (noise baselines + `idiff
   trained`), replacing N/A: their condition-independent pool vs each condition's
   reference, window- and sample-size-matched (14 frames/condition = the model's
   2 seeds × 7 days). N/A read as "not applicable" when the truth is "cannot condition";
   the number now says how far climatology is from the conditional distribution — the
   thing conditioning must beat. `idiff trained`'s old degenerate one-condition 3.39 is
   replaced by a comparable 3.87. ble_vae stays N/A (10 pressure levels, 0.45° grid).
28. **`--cond-tag` / `--cond-churn` CLI passthrough** — the block cache is keyed by
   sampler settings, NOT by checkpoint, so scoring an alternate ckpt needs an explicit
   tag (`--cond-ckpt .../step_300000.pt --cond-tag 300k`). Without it the board silently
   re-scored the 100k cache.

## Finding (2026-07-30): W1_cond does NOT separate the model from noise

Board rerun on the **300k** conditional ckpt (no churn), all rows window-matched:

| | floor | phase shuf | white noise | simplex | helmholtz | idiff trained | **m2cond** |
|---|---|---|---|---|---|---|---|
| W1 cond | 2.38 | 7.40 | 3.13 | 3.38 | 3.22 | 3.87 | **3.13** |
| opp-wind frac (ref 0.201) | 0.20 | 1.00 | 0.78 | 0.23 | 0.39 | 0.01 | **0.03** |
| opp-wind err | 0.00 | 0.80 | 0.58 | **0.02** | 0.19 | 0.19 | 0.17 |
| W1 speed | 1.11 | 1.21 | 1.14 | 1.69 | 1.21 | 3.85 | 3.14 |
| jet elong ratio | 1.17 | 1.10 | 0.87 | 0.85 | 0.70 | 0.93 | **0.95** |

**The headline Shaurya asked about is answered, and the answer is no.** W1_cond was the
model's last standing claim; on it the conditional model (3.13) **exactly ties white
noise (3.13)** and only marginally beats simplex (3.38). Conditioning buys ~nothing over
climatology. The axis is intrinsically compressed — perfect conditioning could only go
3.13 → 2.38 — but the model captures 0% of that range.

**The new decoupling metric separates cleanly and indicts US.** ERA5 makes an opposing
level pair in 20% of columns; simplex hits 23% (err 0.02 = at floor); **our models manage
1-3% (err 0.17-0.19)** — 6-7× too few. Both trained models are drastically over-smoothed
in the vertical. Phase-shuffle at 1.00 and the 0.00 floor bracket the metric properly, so
it is well calibrated. **No sampling setting fixes it**: across every cached variant,
churn roughly doubles the fraction (0.03→0.05) then plateaus, and more steps (36, 64) do
nothing — the defect is in the learned model, not the sampler.

**Read:** the model currently has NO metric on which it beats fitted noise. The one place
it is not last is `jet elong ratio` (0.95 vs simplex 0.85), and that sits inside the
floor. The over-smoothing is the same defect the L_eff regression (281→421) hinted at,
and it is the property the downstream RL task depends on most — vertical decoupling is
what makes station-keeping possible at all. This is the strongest available argument for
coarse (vertically-resolved) conditioning: it would hand the model the vertical structure
it is failing to invent.

## Phase 5b — COARSE CONDITIONING (2026-07-30, Shaurya approved the experiment)

Kept strictly separate from the existing model: `coarse_factor=0` (every existing config
and checkpoint) leaves the architecture numerically identical, asserted by test.

29. **The coarse field = a horizontal block-mean of the target block, ALL 18 levels kept**
   (`data.coarsen`, area average via `avg_pool2d`). Area-mean, not blur-and-subsample, so
   the cell mean of the fine field IS the coarse value and "what did the model add inside
   the cell" is well posed. It commutes with the per-channel affine normalisation
   (asserted), so coarsening the normalised block == normalising the coarsened raw field.
   Vertical resolution is deliberately NOT reduced: the measured defect is that the model
   invents no vertical decoupling, so the coarse field supplies it.
30. **factor 8**: a 64² crop at 0.25° → 8×8 cells of 2° ≈ 200 km. Above the 56 km L_eff
   target (real fine-scale work left) and comparable to operational coarse products.
31. **Injected as clean concatenated input channels, per frame** — a separate `coarse`
   argument rather than reusing `cond` (which is τ-constant): the coarse field evolves
   frame to frame, so it also carries temporal information. Low-res in the dataloader,
   bilinearly upsampled inside the net (nearest would inject block edges the net must
   learn to ignore). Channel order [x, cond, coarse]; only the noisy target gets `c_in`.
32. **Everything else identical to the 300k m2cond baseline** — same 1-yr data, size,
   optimizer, schedule, seed (`configs/era5_2023_m2coarse.yaml`). One variable.
33. **No classifier-free-guidance dropout in v1** (recorded extension): it would let one
   model do both conditional and unconditional and expose a guidance knob, but it is a
   second change.
34. **MANDATORY companion row `coarse upsampled`** — bilinear upsampling of the identical
   conditioning field, no model. Scored automatically whenever `--coarse-ckpt` is given.
   Any metric where the model does not beat it is a metric where the model added nothing.
35. **The coarse row is NOT comparable head-to-head with the unconditional rows** and the
   report says so: it is handed the synoptic state the others must invent. It answers the
   simulator's actual question (forecast → realised field, per the two-field design)
   rather than "can it invent weather".

### Finding (2026-07-30, BEFORE the model finished training): the baseline is strong

Scoring `coarse upsampled` alone — no diffusion model at all — against the held-out ref:

| | floor | coarse upsampled | m2cond 300k |
|---|---|---|---|
| SR_E | 0.25 | 1.74 | **1.62** |
| SR_div | 0.25 | **1.70** | 2.02 |
| SR_vort | 0.25 | **1.82** | 2.06 |
| L_eff (km) | 56 | 481 | 481 (same k-bin) |
| W1 u | 1.07 | **1.93** | 3.83 |
| tail err 0.1% | 2.92 | **5.91** | 13.37 |
| W1 speed | 1.11 | **0.19** | 3.14 |
| opp-wind err | ~0.02 | **0.025** | 0.17 |

**Plain upsampling of a 2° field beats the 300k conditional model on 6 of 8 metrics** —
the conditioning information is worth more than everything the unconditional model
learned. It also all but closes the vertical-decoupling gap (0.025 vs 0.17), which
confirms the mechanism: coarse conditioning fixes that defect by SUPPLYING the structure,
not by teaching the model to invent it. Say this plainly in any writeup.

**W1_cond DEGENERATES for coarse rows — do not compare it across protocols.** Upsampling
scores 0.22, far BELOW the 2.38 floor, because its conditioning comes from the very days
being scored: for coarse rows W1_cond is a reconstruction error, not a test of conditional
realism. (The floor is across-day; the coarse rows are same-day.)

**⇒ SUCCESS CRITERION for the coarse experiment** (set before seeing the model, to avoid
moving the goalposts): the coarse model must **beat `coarse upsampled` on the spectral
rows — SR_E / SR_div / SR_vort and L_eff** — because that is exactly what block-averaging
destroyed and the only place a generative model can add value here, **while not degrading
the distribution and structure rows** (W1 u/speed, tails, opp-wind err) relative to
upsampling. Beating only the unconditional rows proves nothing.

### Run + tooling
- **Job 1021** `idiff-m2coarse`, gpu48, batch 16, 2.6 it/s, 300k target (~32 h),
  snapshots every 10k for an early read. Launched while the 4-yr download runs on CPU.
  **KILLED at step 7.1k on 2026-07-30 and replaced by job 1022 `idiff-m2coarse2`** — see
  decisions 36–41 below. The config `era5_2023_m2coarse.yaml` is kept for the record.
- Board: `--coarse-ckpt runs/idiff_m2coarse2/latest.pt` (adds both rows).
- Eye test: `python -m src.eval.windeval.scripts.eye_test_coarse` →
  `coarse_eye_spatial.png` (speed maps: ERA5 / coarse input / upsampled / diff rows) and
  `coarse_eye_vertical.png` (bearing-vs-level columns + the opposing-wind bar chart).

## Phase 5b-2 — RESIDUAL parameterization (2026-07-30, Shaurya: "do what you think is
## best for performance")

### Finding that forced the redesign: the conditioner gives away 97.4% of the variance

Measured on the held-out reference, normalized units, bilinear lift of the block-mean:

| factor | resolution | variance explained by upsampling | residual std |
|---|---|---|---|
| 2× | 0.5° | 99.9% | 0.022 |
| 4× | 1.0° | 99.4% | 0.064 |
| **8× (ours)** | **2.0°** | **97.4%** | **0.139** |
| 16× | 4.0° | 92.4% | 0.236 |

This explains the whole board at once: on any variance-dominated metric (W1 of marginals,
tails, opposing-wind fraction) upsampling is already near-perfect and there is nothing
left to win. The remaining 2.6% lives entirely above the coarse Nyquist — which is
exactly why the pre-registered success criterion is the spectral rows. The criterion now
has a quantitative justification, not just an argument.

It also exposes a **training-budget bug in job 1021**. EDM draws
`log σ ~ N(-1.2, 1.2²)` with `sigma_data = 1`. The signal the conditioner has *not*
supplied has amplitude 0.139, and `P(σ > 0.139) = 0.74`. So ~3/4 of training steps drew
noise that swamps everything unknown; at those steps the optimal denoiser is literally
"return the upsampled coarse field", and the step exerts no pressure toward fine
structure. Three quarters of a 31-hour run was going to an already-solved problem.

### 36. Residual parameterization — diffuse `(x − U(coarse)) / coarse_scale`
**Decision (Claude).** The diffusion models the residual, not the field. `coarse_scale`
is the measured residual RMS (~0.139), so the residual is unit-variance and `sigma_data`,
`P_mean`, `P_std`, `sigma_min/max` all keep their tuned EDM values while now acting on
the band that is actually unknown. **Rescaling the residual rather than lowering
`sigma_data` is deliberate** — it is the equivalent fix with the smaller blast radius
(nothing downstream in the sampler needs retuning).
*Side benefit:* "predict zero" is now **exactly** the `coarse upsampled` control row, by
construction and not by approximation (asserted in test F1). The model starts at the bar
it has to clear instead of having to rediscover it.

### 37. `coarse_scale` measured from the training set, not hard-coded
**Decision (Claude).** `data.measure_residual_scale` over 256 deterministic blocks at run
start; the value is written into `cfg` and therefore into every checkpoint, so sampling
reproduces the training transform exactly. Hard-coding 0.139 would silently be wrong the
moment the factor, the level set, or the year changes. A resumed leg re-measures on a
different sample, but the estimate covers ~1.5e8 values so leg-to-leg spread is
negligible.

### 38. Exact block-mean consistency projection at sampling (ILVR / replacement)
**Decision (Claude), default ON, ablatable via `--no-coarse-project`.** After each
denoiser evaluation, `full ← full + U_nn(coarse − blockmean(full))`. This turns "reproduce
the synoptic state you were handed" from a soft learned objective into an exact
constraint, frees capacity for the residual band, and removes the bilinear lift's own
cell-mean bias. The **nearest** lift is required, not bilinear: it is the adjoint of
block-mean, so it shifts each cell by a constant and provably cannot manufacture sub-cell
structure (asserted in test F4 via within-cell variance). Applied to the *denoised
estimate* inside each Heun stage, not to the noisy iterate — the numerically stable
variant. Costs nothing.

### 39. Conditioning dropout 0.1 → classifier-free guidance available at sampling
**Decision (Claude).** 10% of training steps see a zeroed coarse field. Over-smoothing is
the model's named defect and CFG is the standard sharpening knob, but it **must be baked
in before training** — hence now. Two guardrails: (a) `guidance = 1` is the default and is
*exactly* ordinary conditional sampling, so the board row can never silently become a
guided row; (b) a guided sample on a checkpoint with no unconditional branch raises rather
than silently degrading. Cost: 10% fewer conditioned steps, and 2 forward passes per step
only when guidance ≠ 1.

### 40. A flag channel accompanies the dropout
**Decision (Claude).** An all-zero coarse field in normalized units is a *legal* field
(climatological mean everywhere), so without a marker the network cannot distinguish "no
conditioning" from "a genuinely calm synoptic state" and the unconditional branch is
contaminated. One constant plane, 1.0 when conditioned and 0.0 when dropped. Present only
when `coarse_dropout > 0`, so it does not perturb any other run.

### 41. Conditioning augmentation DEFERRED — flagged for Shaurya
**Decision (Claude), and the one I most want reviewed.** Training with a *noised* coarse
field (Ho et al. cascaded-diffusion style, with the noise level passed as an embedding) is
what makes a downscaler survive a real forecast — biased, differently-spectrumed, wrong —
instead of the block-mean of truth it is trained on. I left it out of this run because it
does not serve the pre-registered spectral criterion, it cannot be ablated within one run,
and it adds bug surface. **It is not optional for deployment**, and it is the first thing
to add when this stops being a metric exercise and starts feeding the simulator.
Also deferred: multi-resolution injection (coarse enters only at `in_conv` today) and
feeding both nearest and bilinear lifts. Both are P2 and would confound this run.

### Verification before launch (2026-07-30)
- **22 checks pass** in `test_coarse_conditioning.py` (section F is new); structure-metric
  suite unchanged, ERA5 opp-wind regression still 0.2029.
- **CPU smoke train, real zarr, 3 steps:** `coarse_scale=0.1220` measured over 256 blocks
  (held-out centre window gave 0.1385, a 32-block draw 0.1286 — the spread is the crop
  sampling, and the training-set value is the correct one to train with). Loss finite,
  checkpoint written.
- **End-to-end sampling from that checkpoint**, coarse taken from the held-out reference:

  | sampler setting | block-mean(sample) vs coarse, max abs err |
  |---|---|
  | projection ON (default) | **1.67e-06** (float32 precision) |
  | projection OFF | 5.23e-01 |

  i.e. the projection does exactly what decision 38 claims. `guidance=1.5` runs, stays
  finite, and produces a genuinely different sample from `guidance=1`.
- **The residual parameterization is visible even at 3 training steps**: samples come out
  at `|u|max = 62.8 m/s` — physically sane — because the base (upsampled ERA5) dominates
  and the untrained residual is small. Under the old parameterization an equally untrained
  model produced garbage. This is decision 36's "starts at the control row" property,
  observed rather than argued.

### First read: step 10k of 300k (3.3% trained) — FAILS the criterion, informatively

Board vs `coarse upsampled` (`--coarse-tag m2coarse2_10k`). **Direction read, not a result.**

| row | upsampled | diff+coarse 10k | |
|---|---|---|---|
| SR_E / SR_div / SR_vort | **1.74 / 1.70 / 1.82** | 4.79 / 4.83 / 4.81 | ✗ much worse |
| L_eff (km) | 481 | 56† | *not* a win — see below |
| W1 u / v | 1.93 / 0.56 | **1.77 / 0.45** | ✓ |
| tail 1% / 0.1% | 4.08 / 5.91 | **3.60 / 5.09** | ✓ |
| W1 speed | 0.19 | **0.11** | ✓ |
| jet speed 99% / 99.9% | 0.82 / 1.39 | **0.26 / 0.66** | ✓ |
| W1 shear u / v | 0.87 / 0.71 | **0.37 / 0.15** | ✓ |
| opp-wind frac (ref 0.20) / err | 0.18 / 0.03 | **0.19 / 0.01** | ✓ |
| jet elong ratio (→1) | **0.88** | 0.82 | ✗ slightly |

**The criterion is failed and the failure is the opposite of the one anticipated**: the
model improved essentially every distribution and structure row and lost badly on all
three spectral ratios. **The `56†` L_eff is not a win** — `†` means the energy ratio never
crossed threshold, which together with SR_E 4.79 is the signature of *excess* high-k
energy. Upsampling's defect is a small-scale *deficit* (SR 1.74); this model overshoots
the other way. The two are not symmetric and must not be reported as "resolves to ERA5
scale".

**The eye test says what the numbers cannot** (`coarse_eye_spatial.png`): the diff+coarse
panels place the synoptic structure correctly — the Jan-08 bottom-left jet core and the
Jul-12 diagonal jet are both recovered sharply, where `coarse upsampled` blurs them and
`diff (no coarse)` misses them entirely — but the whole field carries a visible **speckle**.
The added fine-scale energy is *noise, not weather texture*. That is exactly SR_E 4.79
drawn. At 3.3% trained this is the expected undertrained-residual signature; it is also
the specific thing to re-check at 50k, because a speckle that persists would mean the
residual is being modelled as near-white and the scale/schedule needs another look.

Note this defect is genuinely fixable and not an information limit: sub-cell detail is NOT
determined by the conditioner, so a sample can never match ERA5 pointwise — but SR is a
PSD *statistic*, so a correct generative model should match it regardless.

`coarse_eye_vertical.png`: diff+coarse columns track ERA5 bearing-vs-level closely
(opp-wind 0.19 vs ERA5 0.20) while diff-no-coarse wanders (0.03). **Say plainly that this
is SUPPLIED, not learned** — all 18 levels are handed to the model (decision 29).

### VERDICT at step 300k (2026-08-01): the success criterion is MET

Converged model (`--coarse-tag m2coarse2_300k`), loss 0.104 → 0.059. The criterion —
set before seeing any of this — was: **beat `coarse upsampled` on the spectral rows
without degrading the distribution/structure rows.** Both halves hold.

| row | upsampled | **diff+coarse 300k** | 10k | verdict |
|---|---|---|---|---|
| **SR_E** | 1.74 | **1.00** | 4.79 | ✓ |
| **SR_div** | 1.70 | **1.04** | 4.83 | ✓ |
| **SR_vort** | 1.82 | **0.98** | 4.81 | ✓ |
| **L_eff (km)** | 481 | **198** | — | ✓ |
| W1 u / v | 1.93 / 0.56 | **1.77 / 0.44** | 1.77 / 0.45 | ✓ |
| tail 1% / 0.1% | 4.08 / 5.91 | **3.57 / 5.05** | 3.60 / 5.09 | ✓ |
| W1 speed | 0.19 | **0.11** | 0.11 | ✓ |
| jet speed 99% / 99.9% | 0.82 / 1.39 | **0.34 / 0.61** | 0.26 / 0.66 | ✓ |
| W1 shear u / v | 0.87 / 0.71 | **0.38 / 0.11** | 0.37 / 0.15 | ✓ |
| opp-wind frac (ref 0.20) / err | 0.18 / 0.03 | **0.19 / 0.01** | 0.19 / 0.01 | ✓ |
| jet elong ratio (→1) | 0.88 | **0.93** | 0.82 | ✓ |

**The 10k speckle was undertraining, as hypothesised.** SR_E 4.79 → 1.00 between 10k and
300k while every distribution row stayed put — i.e. the extra training went almost
entirely into fixing the small-scale spectrum, which is exactly what the residual
parameterization was designed to make the training budget do (decision 36). SR_E 1.00 is
**the best spectral row any trained model has produced on this board** (m2cond 2.04,
idiff trained 2.09). The eye test agrees independently: `coarse_eye_spatial.png` shows
coherent filaments (Jan-08 wave train, Jul-12 jet) where the 10k figure showed grain.

**Do not overclaim — four caveats that belong in any writeup:**
1. **Simplex still wins two spectral rows** (SR_E 0.75 vs 1.00, SR_vort 0.72 vs 0.98). Its
   horizontal scale was FIT by searching on SR_E, so it is near-optimal there by
   construction, and it collapses on the conditional/structural rows. Neither direction
   of that comparison is like-for-like.
2. **L_eff 198 km is still 3.5× the 56 km target.** Best on the board, not resolved.
3. **The comparison is asymmetric by construction** — this model is handed the synoptic
   state; the vertical decoupling it "fixes" (opp-wind 0.19 vs ERA5 0.20, against 0.03
   unconditional) is SUPPLIED, not learned.
4. **`jet area ratio` stays PROVISIONAL** (floor 1.42 = real synoptic variability).

Note `coarse_scale` in the final checkpoint is 0.12156, not the 0.1220 the run started
with: the post-reset leg re-measured on a shifted `data_seed`, as decision 37 says it
would. 196k steps at 0.1220, 104k at 0.12156 — a 0.3% shift, far inside the noise.

### Sampler sweeps at 300k: decision 39 earned its place, decision 38 did not

**Classifier-free guidance (decision 39) — a free, near-monotone win.**

| row | g=1.0 | **g=1.5** | upsampled |
|---|---|---|---|
| SR_E | 1.00 | **0.88** | 1.74 |
| SR_div | 1.04 | **0.93** | 1.70 |
| SR_vort | 0.98 | **0.86** | 1.82 |
| L_eff (km) | 198 | **146** | 481 |
| W1 u / v | 1.77 / 0.44 | **1.76 / 0.43** | 1.93 / 0.56 |
| tail 1% / 0.1% | 3.57 / 5.05 | **3.55 / 5.01** | 4.08 / 5.91 |
| jet speed 99% / 99.9% | **0.34** / 0.61 | 0.37 / **0.59** | 0.82 / 1.39 |
| shear u / v | 0.38 / **0.11** | **0.37** / 0.12 | 0.87 / 0.71 |
| jet elong (→1) | 0.93 | **0.95** | 0.88 |

Two trivial regressions (jet 99% 0.34→0.37, shear v 0.11→0.12) against L_eff 198→146 km
and SR_E 1.00→0.88. Costs 2 forward passes per sampler step, nothing at train time.
This retroactively justifies baking `coarse_dropout=0.1` in before training — it could not
have been added afterwards.

**Full sweep, and the knee is at 1.5:**

| row | g=1.0 | **g=1.5** | g=2.0 | upsampled | simplex | floor |
|---|---|---|---|---|---|---|
| SR_E | 1.00 | 0.88 | **0.80** | 1.74 | 0.75 | 0.25 |
| SR_div | 1.04 | 0.93 | **0.86** | 1.70 | 0.87 | 0.25 |
| SR_vort | 0.98 | 0.86 | **0.78** | 1.82 | 0.72 | 0.25 |
| L_eff (km) | 198 | 146 | **140** | 481 | 56† | 56† |
| tail 1% / 0.1% | 3.57 / 5.05 | 3.55 / 5.01 | **3.53 / 4.99** | 4.08 / 5.91 | 1.73 / 2.34 | 2.01 / 2.92 |
| jet speed 99% | **0.34** | 0.37 | 0.39 | 0.82 | 3.41 | 4.75 |
| jet speed 99.9% | 0.61 | 0.59 | **0.56** | 1.39 | 4.20 | 5.63 |
| shear v | **0.11** | 0.12 | 0.14 | 0.71 | 0.97 | 0.10 |
| jet elong (→1) | 0.93 | 0.95 | **0.96** | 0.88 | 0.85 | 1.17 |

**L_eff knees hard between 1.5 and 2.0**: 198→146 km is a 26% gain, 146→140 is 4%. The SR
rows keep improving roughly linearly, but the two monotone regressions keep growing too
(jet 99% 0.34→0.37→0.39; shear v 0.11→0.12→0.14).

**RECOMMENDATION (Claude, flagged for Shaurya): default `--coarse-guidance 1.5`.** It
captures ~60% of the total spectral gain and essentially all of the L_eff gain, for half
the degradation. g=2.0 is defensible if the spectrum is the headline — at 2.0 the model
**beats simplex on SR_div** (0.86 vs 0.87) and nearly ties it on SR_E (0.80 vs 0.75),
which is the first time anything trained here has come near the fitted-noise baseline on
its own best metric. Add "guidance default" to the pending-decisions list alongside the
churn default.

**Consistency projection (decision 38) — NULL on every metric.**

| | proj ON | proj OFF |
|---|---|---|
| SR_E / div / vort | 1.00 / 1.04 / 0.98 | 1.01 / 1.06 / 0.99 |
| L_eff (km) | 198 | 210 |
| everything else | — | identical to 2 d.p. |

All differences are inside the noise. **Say plainly that the metric justification for
decision 38 failed.** What the ablation actually measured is a property of the *model*:

| | block-mean error vs conditioner (normalised units) |
|---|---|
| untrained (3 steps), proj off | 0.523 |
| trained 300k, proj off | **0.0455** mean, 0.0757 max |
| trained 300k, proj on | **0.0000** |

By 300k the network has learned the block-mean constraint to ~4.6% of field std — an 11×
improvement over untrained — which is why forcing it exactly moves no metric. But note
0.0455 is **37% of the residual scale (0.1216)**, i.e. sizeable in the space the model
actually works in; the board is structurally blind to it because every row is
distributional or spectral.

⇒ **Keep the projection ON, on non-metric grounds**: it makes "the realised field is
consistent with the forecast it was handed" a guarantee rather than a 4.6% near-miss,
which is the simulator's actual question. It costs nothing. Do not claim it improved
the board.

### What did NOT change (so the comparison survives)
Same data (`era5_2023.zarr`, days 8–14 excluded), same architecture and size, same
optimizer, same seed, same 300k schedule, same coarse operator and factor 8, same
evaluation protocol and the same pre-registered success criterion. `coarse_factor = 0`
still yields a byte-identical pre-5b model; the m2cond 300k checkpoint still loads,
samples, and reports `guidance 1 / residual off / scale 1` (test F5).

## Findings

- **Smoke test (2026-07-10, local CPU):** 6-step tiny conditional model on
  `era5_temporal.zarr` trains (loss ≈0.9–1.1, finite), checkpoints with
  `coord_norm={lat 40±15, lon 240±15}`, and the standalone gate samples
  condition-matched blocks end-to-end (SR_E ratio ≈7 — expected garbage at step 6).
  Regression checks: time-feature phases correct at Jan-1/mid-year and 00/12 UTC; the
  ±180↔0–360 longitude branch guard maps −122.42° onto the 237.58° grid exactly;
  unconditional spacetime forward/loss and `WindSpaceTimeDataset` unchanged.
- **Kahan GPU pools (from `train_temporal.sbatch`, supersedes "one 96 GB GPU"):** MIG
  partitions `gpu24`×8 / `gpu48`×2 / `gpu96`×2; jobs MUST request a pool via
  `--gres=gpu:<pool>:1`. batch 32 × τ 4 needs `gpu96`; batch 16 fits `gpu48`.

## Reproduce

```bash
# 1. Download all-of-2023 hourly, days 8–14 excluded (run on the Kahan login node;
#    CDS queue dominates). 24 chunks = two per month:
LASTS=(31 28 31 30 31 30 31 31 30 31 30 31)   # 2023 is not a leap year
DATES=""
for i in $(seq 1 12); do
  m=$(printf "%02d" "$i")
  DATES+="2023-$m-01/to/2023-$m-07 2023-$m-15/to/2023-$m-${LASTS[$((i-1))]} "
done
python src/eval/windeval/generators/infinite_diffusion/download_era5_train.py \
  --dates $DATES --time 00/to/23/by/1 --prefix era5_2023 \
  --out src/eval/windeval/data/era5_2023.zarr

# 2. Train on Kahan (resumable; ~14 GB data RAM + gpu96 pool):
CONFIG=src/eval/windeval/generators/infinite_diffusion/configs/era5_2023_m2cond.yaml \
  sbatch --gres=gpu:gpu96:1 --requeue -J idiff-m2cond \
  src/eval/windeval/generators/infinite_diffusion/configs/train_temporal.sbatch

# 3. Mid-training gate (on the cluster, standalone):
python src/eval/windeval/generators/infinite_diffusion/gate.py runs/idiff_m2cond/latest.pt \
  --ref src/eval/windeval/data/era5_2023.zarr
```

## File-level change log

All paths under `src/eval/windeval/generators/infinite_diffusion/` unless noted.

| Change | Files |
|---|---|
| NEW tracking doc | `docs/conditional-base-changes.md` (this file) |
| NEW architecture explainer | `docs/conditional-base-model.md` — how the model works and where each idea comes from (EDM backbone, factorized space-time U-Net, coord-channel location conditioning, cyclic-harmonic time conditioning with the bandlimit argument, 14 references) |
| NEW conditioning features + dataset | `data.py`: `CoordNorm` (domain-normalized per-pixel lat/lon channels + lon branch guard), `time_features` (6 cyclic harmonics), `WindCondSpaceTimeDataset` (blocks + coords + tfeat; augmentation forced off) |
| Conditioning in the spacetime model | `spacetime.py`: `SpaceTimeUNet(cond_channels=, time_features=)` — coord channels concat clean at input, zero-init tfeat linear into the per-frame embedding; `EDMPrecondSpaceTime.forward/loss(cond=, tfeat=)`; `SpaceTimeSampler` loads conditional ckpts and takes `sample_block(..., lat=, lon=, times=)` |
| Conditional training wiring | `train.py`: `conditional` flag (requires spacetime), dataset triple unpack, `coord_norm` saved in every checkpoint |
| NEW run config | `configs/era5_2023_m2cond.yaml` (τ=4, hourly, 128-ch, batch 32 → gpu96); reuses `train_temporal.sbatch` via `CONFIG=` |
| Gate: spacetime + conditional | `gate.py`: detects `cfg["spacetime"]`, condition-matched `_ref_blocks`, pools n·τ frames; static path unchanged |
| Download hardening | `download_era5_train.py`: `--stream` constant-memory ingest (per-chunk zarr append + monotonic-time check; the in-RAM path needs ~28 GB for a full hourly year), resumable downloads (`.part` + skip-existing) |
| Fail-loud device | `era5_2023_m2cond.yaml` `device: cuda` (not auto) — the July M2 jobs silently fell back to CPU on GPU-less allocations and crawled at <0.05 it/s to the 24 h limit |

## Operational notes (2026-07-10 launch session)

- **M3 finished 100k steps on Kahan (Jul 4)** — `runs/idiff_m3/step_100000.pt` + 49 more
  ckpts (~30 GB). M2 never trained (CPU-fallback jobs). M3 is not the Phase-5 route but
  is a free temporal-board row later.
- **Kahan SSH**: pubkey auth impossible (AFS home unwritable) → local `~/.ssh/config`
  `Host kahan` with ControlMaster/ControlPersist 12h; user logs in once with password,
  subsequent `ssh kahan` commands ride the socket.
- Download runs on the Mac (proven cdsapi+cfgrib path) with `--stream` (24 GB laptop);
  zarr rsyncs to Kahan when done. zsh gotcha: `--dates ${=DATES}` (zsh doesn't word-split
  unquoted vars — the first attempt sent all 24 specs as one MARS request, which failed).
- **CDS reality (2026-07-11): `reanalysis-era5-complete` allows ~1 active request per
  user** — 23 of 24 parallel submissions were instantly `rejected`. Final shape: 14
  sequential requests (weekly leftovers + one explicit-date-list request per month),
  ~1–2.5 h each in the MARS queue, ~26 h wall total. Laptop sleep pauses the local
  poller, not the server-side job. `era5_2023.zarr`: 6744 hourly steps × 18 × 121², 8.3
  GB, verified days-8–14-free + monotonic + finite (|u| ≤ 63.7 m/s).
- macOS ships rsync 2.6.9 — no `--info=progress2` (it prints usage and copies nothing;
  verify transfers by remote file count, not exit code).

## Training launched (2026-07-12)

- `era5_2023.zarr` rsynced to Kahan (10,362 files, verified count-identical).
- **Job 941** = `idiff-m2cond`, **gpu48 slice, batch 16** (τ=4 → 64 spatial frames).
  Admin asked us off the two gpu96 slices, so batch dropped 32→16 (config updated).
  Job 935 (gpu96, batch 32) was cancelled ~1 min in; stub run dir removed.
  `--requeue` + ckpt-every-2000 rides out the admin's ongoing SLURM restarts.
- Confirmed at launch: `device=cuda` (the fail-loud config), dataset loading.
  First-checkpoint gate: `python src/eval/windeval/generators/infinite_diffusion/gate.py
  runs/idiff_m2cond/latest.pt --ref src/eval/windeval/data/era5_2023.zarr` (compute node).
- **Gate @ step 10k (n=4, vs training crops):** W1 v at floor (ratio 1.07), tails 1.25×,
  W1 u 1.45× — the conditioning already matches the local climate; SR_E/div/vort ~4×
  floor (fine scales come later, as in M1). Finite, loss descending.
- **The "SLURM keeps killing us" saga (Jul 12–14) — root cause was DISK QUOTA.** Five
  consecutive job deaths, each exactly at the first checkpoint write after a resume
  (truncated `step_*.pt` every time); then submissions started failing instantly with
  `InvalidAccount` / `Account=(null)` and were purged pre-start. All of it was the
  `/zooper2/$USER` quota (~100 GB) filling up: 63 GB was accumulated checkpoints
  (idiff_m1 25 GB + idiff_m3 30 GB + m2cond 8 GB). At the edge, each 641 MB step-file
  write died mid-stream; fully exhausted, slurmd couldn't even write the job's .out →
  instant FAILED (RaisedSignal:53) with NO error anywhere user-visible. `Account=(null)`
  is this cluster's *normal* accounting-off state — a red herring. Diagnosed by a
  minimal `sbatch` test whose script write failed with "Disk quota exceeded".
  **Cleanup:** kept idiff_m1/step_84000 (benchmarked), idiff_m3/step_100000 (final),
  m2cond latest+step_10000; deleted ~60 GB of intermediate ckpts + ingested gribs
  (101→41 GB). Job 981 (resumed step 32k, --mem=48G) survived its first checkpoint.
  **Standing rule: prune step_*.pt as runs progress; a full run adds ~21 GB.**
- **TRAINING COMPLETE (Jul 14, job 981): 100,000 steps, final loss 0.018, clean exit.**
  Pruned run dir 23→1.9 GB (kept latest.pt = step_100000 + step_10000); final ckpt
  pulled to the Mac (`runs/idiff_m2cond/latest.pt`, 641 MB, size-verified).
- **Gate @ step 100k (n=4, vs training crops) — every ratio in the healthy band:**
  SR_E 1.10, SR_div 0.98, SR_vort 1.23 (spectral collapsed from ~4× at 10k → ≈1;
  the open question from mid-training resolved in our favor), W1 u 1.19, W1 v 1.47,
  tail 1% 1.88, tail 0.1% 2.00. Tails at ~2× floor are now the largest residual gap —
  consistent with the v2 board hypothesis; the real test is the held-out benchmark.
  Caveat: n=4 gate floors are re-drawn per run, so small ratio shifts vs the 10k gate
  (e.g. W1 v 1.07→1.47) are within sampling noise — read bands, not deltas.
- **benchmark.py conditional wiring landed (Jul 14).** W1_cond protocol = Shaurya's
  (location, month, hour) decision; wiring decisions 9–15 above. Sampling cost: 112
  blocks ≈ 8 min on the Mac (3.9 s/block, MPS); blocks cached at
  `data/idiff_m2cond_blocks.npz` (--regen to resample).
- **HELD-OUT BOARD, m2cond vs static M1 (the first conditional-model board, Jul 14):**
  SR_E 2.09→1.62, SR_div 2.14→1.64, SR_vort 2.10→1.64, **L_eff 673→337 km** (target 56),
  W1 shear u 1.70→1.10, v 1.62→1.15, W1 u 4.26→3.88, W1 v 3.00→3.29 (≈flat),
  **tail err 1% 11.5→10.8, 0.1% 15.6→13.8** (target 2.9; floor 2.9),
  **W1_cond 3.26 vs condition-matched floor 2.38 (1.37×)** — the conditioning
  demonstrably matches per-condition distributions. Read: conditioning + 24× more real
  data moved SPECTRA a lot (L_eff halved) and TAILS only modestly — the gate's 2×-floor
  tails were vs training crops at matched conditions; on held-out data extremes remain
  the gap. Shaurya's recorded fallback (coarse-field conditioning) is now the live
  question, vs. cheaper knobs first (more seeds/hours in the protocol, temporal tiling,
  multi-year data). Caveat for interpretation: the m2cond row compares the 64² center
  window against the full 121² reference — extremes that live off-window (jet-edge
  events near the domain boundary) can inflate its tail error relative to a
  window-matched comparison; worth a window-matched ref variant before concluding.
- **4-year download stall + recovery (Jul 28-29).** Job 1013 hit its self-imposed 24 h
  wall limit at Jul 28 23:50 and its TERM trap tried to resubmit itself with `sbatch` —
  **which does not exist on the compute node** (SLURM clients live only on kahanctrl;
  `/usr/bin/sbatch` is absent from `kahan` entirely). The chain died silently and ~20 h
  of CDS queue time was lost. Fixes: (a) `--time=7-00:00:00` — the `debug` partition is
  `MaxTime=UNLIMITED`, so the 24 h limit was self-imposed, never a cluster rule; (b) the
  trap now reports honestly instead of pretending to resubmit; (c) safety net = a
  login-node `--dependency=afterany` job (the script is idempotent, so a spurious run
  just exits). Also recovered the in-flight Oct-2022 request directly from the CDS cache
  (it had finished server-side at 01:09 while nothing was listening) — saved ~4.5 h.
  **Status Jul 30 15:15 UTC: job 1019 healthy at 19h37m (1020 = safety net still
  pending). era5_2022.zarr DONE (8.6 GB, gribs auto-deleted); 2021 at 8/12, Sep queued;
  2020 not started => 16 of 36 months remain.** Zero errors, zero HTTP retries.
  **The CDS diurnal cycle is now measured over a full turn** and it is steep: overnight
  UTC 33-35 min/month (four back-to-back 21:30-23:13), degrading through 2 h (01:00-05:00)
  to 4 h+ by 09:00, and the 09:27 request sat `accepted` (queued, never started) for
  5h48m — verified alive via the CDS jobs API, i.e. ECMWF-side congestion, not a fault.
  **Blended rate is nonetheless a stable ~2.2-2.3 h/month across both jobs**, so use that
  for planning, not the instantaneous rate. **ETA ~Aug 1 (range Jul 31 - Aug 2).**
  Quota 41 GB of ~100 GB; projected peak ~63 GB when 2020's gribs coexist with 4 zarrs.
- **`train_temporal.sbatch` had the SAME broken self-resubmit — caught before the 4-yr
  launch (Jul 29).** Its trap also called `sbatch` from the compute node, and its wall
  limit was 24 h while a 300k-step 4-year run needs **~31 h** at 2.7 it/s (500k ≈ 51 h).
  Left alone, the big run would have hit the wall and simply stopped, with the "chain"
  never firing — silently, the way the download did. Both scripts now use
  `--time=7-00:00:00` + an honest trap; the documented safety net is a login-node
  `--dependency=afterany` job (harmless when resume=true). **Lesson: a recovery path
  that has never actually fired is not a recovery path.**

## Decision by Shaurya (2026-07-27): make the base model as strong as possible

Approved the full improvement sequence (recommendation-first list, his "lets get through
all the improvements"): (1) window-matched ref check, (2) sampler sweep (steps, then
churn), (3) longer training on 2023 data, (4) **scale training data to 4 years
(2020–2023)**, with coarse-field conditioning still the recorded fallback if tails
resist. Downloads + jobs run on Kahan.

## Decisions made by Claude (2026-07-27, executing the sequence — flagged for review)

22. **4-year download mirrors the 2023 protocol exactly** — hourly, same box/levels, one
    explicit-date-list MARS request per month, **days 8–14 excluded in ALL years** (not
    strictly needed for 2020–2022 contamination since the reference is 2023-only, but it
    preserves the option of a future multi-year eval; also keeps every year statistically
    comparable). Downloader: `configs/download_years.sh`, CPU-only sbatch on the compute
    node with wall-limit self-resubmit; per-year zarrs (`era5_2020/2021/2022.zarr`),
    gribs deleted after each year lands (quota). CDS key copied to
    `/zooper2/$USER/.cdsapirc` (chmod 600; $HOME is AFS).
23. **Resume-replay bug found + fixed in train.py**: dataset items are idx-seeded and the
    DataLoader restarts at idx 0 every leg, so each 24 h resubmit of job 981 REPLAYED the
    same crop sequence — the 100k model saw roughly the first-leg crops cycled, a large
    silent cut to effective data diversity. Fix: peek the resume step before building the
    dataset and shift the dataset seed by it (`data_seed = seed + start_step`); each leg
    now draws a fresh deterministic stream. Also `snapshot_every` (step_N.pt cadence)
    split from `ckpt_every` (latest.pt cadence) so long runs don't flood the quota.
24. **300k resume on the 2023 data launched immediately** (job 1014, gpu48, snapshots
    every 20k) — isolates the "just train longer" variable from "more data" before the
    4-yr run; measured **2.7 it/s on gpu48** → 200k extra steps ≈ 20.5 h (the old
    0.46 it/s figure from job 981's gpu96 legs was never GPU-bound, it seems).
25. **Stochastic churn added to SpaceTimeSampler** (constructor knobs s_churn/s_min/
    s_max/s_noise, default 0 = bitwise the old deterministic ODE; verified vs cached
    blocks, max Δ 1.3e-5 = cpu-vs-mps float noise). Mirrors trained.py's Alg-2 exactly;
    churn noise seeded per block (seed ^ 0x5F5E1 so it can't collide with init noise).
    `_conditional_artifacts` gained num_steps/s_churn params with per-setting cache
    files (`idiff_m2cond_blocks_s{N}[_c{γ}].npz`) — the board default is untouched.
26. **Multi-zarr + float16 data loading** (WindSpaceTimeDataset + Cond subclass only):
    `data_path` may be a list of per-year zarrs (time-concatenated; grid-equality and
    strictly-increasing-time guards), `data_dtype: float16` halves in-RAM size
    (4 yr ≈ 28.5 GB vs 57 GB > the 64 GB cap; chunk-wise dask cast so fp32 never fully
    materializes; items cast back to fp32 before normalize). Measured fp16 effect:
    3.5e-3 max delta in normalized units (~0.02 m/s). compute_stats now accumulates in
    float64 (required for fp16 storage; shifts fp32-path stats by ~1e-6 relative —
    harmless, incl. across job 1014's resubmit legs). Per-year zarrs stay separate on
    disk (no 33 GB merged copy; avoids a ~66 GB transient on the quota).
    Config: `configs/era5_4yr_m2cond.yaml` (n_steps 300000 — **Shaurya to confirm**).

## Findings (2026-07-27)

- **Window-matched reference check (open decision #1 — RUN): ~⅓ of the m2cond tail
  number was the window-vs-full-ref mismatch, the rest is real.** Same pooled samples,
  ref and floor restricted to the same 64² center window:
  tail 1% 10.77→**7.40** (floor 2.01→2.39), tail 0.1% 13.77→**9.12** (floor 2.92→2.57)
  → the true gap is ≈3.5× floor (was ≈4.7×). W1 u 3.88→2.72; W1 v ≈flat (3.29→3.45).
  **Spectra are NOT a window artifact**: SR_E 1.62→1.88, L_eff 337→356 under the
  window-matched ref. Whether the board adopts window-matched refs for the m2cond row is
  a protocol decision for Shaurya (recommend: yes, as a separate column or footnote —
  the current row understates the model against a floor it can't reach by construction).
- Kahan login node (kahanctrl) cannot import modern numpy at all (CPU predates
  x86-64-v2; wheels refuse with a RuntimeError) — stricter than the old "no AVX → no
  JAX" note. Anything numpy-touching must run on the compute node (CPU-only sbatch is
  fine and coexists with the GPU job).
- gpu48 trains this model at **2.7 it/s** (batch 16) — job-981-era wall-time estimates
  were ~6× pessimistic; a full 300k 4-yr run is ~31 h, not ~10 days.
- **Sampler step-count sweep (18→36→64, deterministic): NULL RESULT — ruled out.** All
  deltas ≤0.2 m/s / ≤0.06 SR both directions; 36 and 64 agree to the 2nd decimal
  (full-ref tails 13.95/13.97 vs 13.77 @18; window-matched 9.30/9.31 vs 9.12). The
  probability-flow ODE is already well-integrated at 18 steps — the smoothness and tail
  deficits are properties of the LEARNED SCORE, not integration error. Keep 18 (cheap).
  Churn sweep (γ=10/40 @18 steps) is the remaining sampler lever — running.
- **Churn sweep (γ = 0/4/10/40 at 18 steps): MONOTONE WIN, saturating at the EDM clamp.**
  Note γ≥7.45 all collapse to per-step γ_eff = √2−1 (the Karras clamp at 18 steps), so
  γ=10 and γ=40 are identical by construction — the tested curve is γ_eff 0→0.222→0.414.
  Every distribution metric improves monotonically, none degrade:
  W1 u 3.88→3.57→3.35 (window-matched 2.72→2.40→2.16, floor 1.42) ·
  tail 0.1% 13.77→13.04→12.65 (window 9.12→8.38→8.00, floor 2.57) ·
  **W1_cond 3.26→2.99→2.87 (floor 2.38 → now 1.21×)** · L_eff 337→337→281 km ·
  SR_E ≈flat (1.62→1.66). Mechanism: stochastic churn restores the marginal variance the
  deterministic ODE under-disperses — exactly the thin-tail signature. RECOMMENDATION
  (needs Shaurya, board numbers change): adopt s_churn=10 @18 steps as the sampling
  default for BOTH model rows (trained.py already supports it seeded-per-window, so the
  static row and the frozen tiling wrapper stay protocol-consistent); regen both rows.
  Residual after all sampler-side wins (window-matched): tails ≈3.1× floor, L_eff ≈5×
  target — now purely training-side (300k resume, 4-yr data, then coarse-field fallback).
  Minor: window-ref L_eff is pinned at 356.22 across ALL variants — k-bin quantization
  of the 64² grid; read L_eff from the full-ref column only.
- **Quota preflight for the 4-yr campaign (Shaurya-requested)**: pruned /zooper2 44→32 GB
  (deleted: idiff_m2cond/step_10000.pt [10k-gate served, numbers recorded here],
  era5_train.zarr [Phase-2 6-hourly set, superseded], .cache/openpi + .cache/uv
  [regenerable; π0.5 ckpts auto-redownload]). Kept: m1 step_84000 + m3 step_100000
  (board/comparison ckpts), era5_temporal.zarr (temporal-tiling work may reuse).
  Projected campaign peak ≈75–81 GB vs the ~100 GB silent killer. Standing policy:
  after each run is benchmarked, prune its non-final step_*.pt.
- **300k-vs-100k board comparison (2026-07-28, resume complete, final loss 0.016 vs
  0.018): longer training improved every DISTRIBUTION metric slightly and REGRESSED the
  spectra.** At matched sampler settings (churn 10): W1 v 3.14→2.96, tails 0.1%
  12.65→12.51 (window 8.00→7.86), **W1_cond 2.87→2.71 = 1.14× the 2.38 floor** — but
  L_eff (full ref) 281→421 km and the deterministic variant 337→481. Reading: more
  MSE-denoising steps on the same year calibrates conditional means (marginals,
  conditioning) while smoothing away small-scale energy — the 100k model's sharper
  spectrum was partly beneficial undertraining noise. **"Train longer" is NOT the tail
  fix** (12.51 vs floor 2.57): the remaining tail hope is 4-yr data diversity, then
  coarse-field conditioning. Churn's benefit persists at 300k (better on every metric
  than deterministic). step_200000.pt kept on Kahan if we want to bisect the spectral
  turnover. Window-ref L_eff jumped 356→1781 = the 64² k-bin quantization artifact
  (flagged earlier); trust full-ref L_eff only. RECOMMENDATION for the board row:
  step_300000 + churn as primary m2cond row (best W1_cond/tails/W1v), keep the 100k row
  until the 4-yr model replaces both — Shaurya to confirm.
- **3-way eye test @300k (Shaurya-requested; docs/figures/eye_test/*_300kvs.png)**: July
  passes for both checkpoints (amplitude + patchy texture match ERA5; maxes 13-14 vs 12).
  January: BOTH miss the 37 m/s jet streak (model maxes ~18) — the tail gap as a picture,
  unchanged by longer training; 300k visibly slightly smoother than 100k (the L_eff
  regression is visible to the eye). Interim temporal check (lag-k autocorr, ML58, all
  conditions): era5 0.971/0.908/0.836 (1/2/3h), 100k 0.943/0.822/0.688,
  300k 0.944/0.830/0.699 — coherent evolution but ~faster decorrelation than ERA5;
  300k marginally better. Full temporal suite rows stay N/A until temporal tiling
  (τ=4 < MIN_FRAMES=16); kinematic-advection toy remains the ready naive floor for then.
- **Level-matching bug found by Shaurya (2026-07-28): "mid level" comparisons paired
  INDICES, not altitudes.** ble_vae's level coordinate is PRESSURE (50-140 hPa) while
  everything else carries ERA5 model-level indices (49-66, ~54-124 hPa via l137 coeffs);
  the marginal figure was showing ble at 140 hPa against the rest at 90.9 hPa. Fixed:
  `_level_hpa()`/`_match_level()` in benchmark.py convert any artifact's levels to hPa
  (coeffs attr → hybrid formula; bare 49-66 ints → frozen-band l137 lookup; else already
  hPa) and the marginal panel + doc figures now match by pressure (all at ~90 hPa;
  title states the pressure). FLAGGED, not changed: the METRIC rows still pair ble's
  10 levels with the first 10 reference levels by index (ble 50 hPa vs ML 49=54.5 hPa …
  ble 140 hPa vs ML 58=90.9 hPa — wrong altitudes for most pairs; part of why its rows
  "partly measure climate mismatch"). Recommend nearest-pressure pairing for
  mismatched-grid artifacts — changes ble board numbers, needs Shaurya's sign-off.
- **ble_vae column recomputed with pressure-paired levels (Shaurya approved)**: pairing
  is clean (every ble level within 4.5 hPa of a ref ML; the band fully covers 50-140 hPa).
  Result — the index bug had FLATTERED ble: W1 u 13.65→17.92, W1 v 3.48→4.89, tail 1%
  10.74→15.36, tail 0.1% 13.22→18.94; spectra unchanged (level-averaged: SR_E 2.73,
  L_eff 842). Method: ref subset re-indexed to ble's nearest-pressure levels so the whole
  suite pairs correctly.
- **Eye-test date rule made explicit (was: arbitrary day from the held-out cache)**: now
  "hardest held-out condition" = the (day, hour) in the cached condition set with max
  ERA5 in-window speed at ~91 hPa → JAN = 2023-01-08 12h (39.5 m/s), JUL = 2023-07-12 00h
  (19.6 m/s). 4-way figures (ERA5 / diff 100k+churn / diff 300k+churn / ble_vae):
  docs/figures/eye_test/winds_{jan,jul}_hardest_4way.png. Read: diff rows wind-like in
  July, jet-less in Jan (max ~19-21 vs 39.5); **ble_vae rows are near-featureless banded
  slabs in BOTH seasons** (max 8-17, almost no spatial structure) — the eye test shows
  what its board numbers mean.
- **ble_vae time-axis caveat found + eye figures corrected (2026-07-29)**: BLE's 9 time
  slices span 48 h (vae.py time_horizon_hours=48) → native 6-h spacing, but the harness
  artifact stores time as raw steps and the hardest eye figures had shown ble slices
  0-3 under the "+0..+3h" headers (really +0/+6/+12/+18 h of BLE-time). Fixed by serving
  ble's +0..3 h frames the way BLE itself answers time queries — linear interpolation
  between slices; caption now states this. No board number affected (temporal rows never
  used ble; distribution/spectra metrics pool slices regardless of spacing); visual read
  unchanged (ble barely evolves: max 15.8→16.0 m/s over 3 h in the Jan figure).
  Decision made by Claude: interpolate-to-hourly (faithful to BLE query semantics)
  rather than relabeling frames "+0/+6/+12/+18h", so all rows share column timestamps.
- **ble_vae W1 shear filled in (2026-07-29, Shaurya asked)**: the suite guard had blanked
  shear for 10-level artifacts because the Δz climatology is per-adjacent-pair of the
  18-level stack. Now benchmark.py builds a matched Δz for ble — thickness between each
  pair of pressure-matched reference levels = sum of the per-pair Δz entries — and passes
  it via a new score(dz_s=...) override (suite guard passes naturally at 10 levels).
  Numbers (ble vs pressure-matched ref, layer thicknesses 0.32–1.03 km):
  **W1 shear u 3.62, v 0.61 (m/s)/km**; 10-level matched-stack floor computed for fairness:
  0.21 / 0.10 (≈ the 18-level floor 0.20/0.10, so cross-protocol comparison is fair).
  ble shear u is ~17× floor and ~3× the diff row. Report file still stale pending board
  regen (decision 5).

## Finding (2026-08-05): the 4-year run — more data does NOT fix the smoothing

Job 1041 finished clean (`done @ step 500000`, final loss 0.01495). Scored as a **curve**,
not a point, because on the 1-year run 100k→300k made `L_eff` WORSE (281 → 481 km) while the
tails never moved. Three fresh block sets (112 blocks each, tags `4yr_100000/250000/500000`);
checkpoint provenance verified per snapshot (4 zarrs in `cfg.data_path`, distinct weight
fingerprints). Reports archived at `docs/benchmark-reports/benchmark_v2_4yr_{100,250,500}k.md`.

| metric | floor | 100k | 250k | 500k | 1-yr @300k |
|---|---|---|---|---|---|
| SR_E | 0.25 | 1.83 | **1.53** | 1.53 | 2.04 |
| SR_div | 0.25 | 1.87 | 1.56 | **1.54** | 2.02 |
| SR_vort | 0.25 | 1.84 | **1.53** | 1.53 | 2.06 |
| L_eff (km) | 56† | **280.62** | 336.74 | 374.16 | 481.06 |
| W1 u | 1.07 | 3.91 | **3.82** | 3.94 | 3.83 |
| tail err 0.1% | 2.92 | 13.85 | **13.54** | 13.87 | 13.37 |
| W1 speed | 1.11 | **2.53** | 2.98 | 2.89 | 3.14 |
| jet speed err 99.9% | 5.63 | **12.75** | 12.88 | 13.16 | 13.01 |
| W1 cond | 2.38 | **3.14** | 3.52 | 3.44 | 3.13 |
| opp-wind frac (≈0.20) | 0.20 | 0.02 | **0.04** | 0.02 | 0.03 |

**Four readings, in order of how much they should change plans:**

1. **The smoothing pathology RECURS. It is not data-limited.** `L_eff` degrades monotonically
   281 → 337 → 374 km. 4× the data *slowed* it (the 1-yr run reached 481) but did not stop it,
   and the direction never reverses. **At matched step count (100k) the 4-yr model's `L_eff` is
   280.62 vs the 1-yr model's 281 — four times the data moved effective resolution by nothing.**
   Every apparent 4-yr "win" over the 1-yr final is the 1-yr run degrading further, not the
   4-yr model being better. This is the strongest evidence yet that the ceiling is structural,
   and the strongest argument yet for the coarse/cascade direction.

2. **Everything plateaus at 250k. The last 250k steps (~25 h of GPU) were wasted or harmful.**
   Spectral ratios stop improving exactly at 250k (1.53/1.56/1.53 → 1.53/1.54/1.53), while
   `L_eff` (337→374), both jet-speed rows, and `W1_cond` all get worse. **By the board, the
   best 4-yr checkpoint is `step_250000.pt`, not `step_500000.pt`.** Loss kept falling the
   whole time (0.0171 @ 231k → 0.0149 @ 500k) — confirming that this loss cannot see the
   defect, because it is dominated by large scales that were already solved.

3. **It still loses to fitted noise nearly everywhere.** simplex `SR_E` 0.75 vs 1.53;
   `W1 u` 1.12 vs 3.94 (floor 1.07); `tail 0.1%` 2.34 vs 13.87 — **6× worse on the tails.**
   `W1_cond` 3.44 is *above white noise's 3.13*, so conditioning still buys nothing over
   climatology; 4× data did not change the 2026-07-30 finding.

4. **Vertical decoupling did not respond at all** — `opp-wind frac` 0.02/0.04/0.02 against a
   0.20 reference, versus 0.03 for the 1-yr model. ~10× too few opposing columns, unchanged
   by 4× data. This is the property RL station-keeping depends on most.

**What DID improve with 4× data:** the spectral rows only. `SR_E` 2.04 → 1.53 (25%), `L_eff`
481 → 374 (22%), `W1 speed` 3.14 → 2.89. The distribution and tail rows are flat-to-slightly-
worse (`W1 u` 3.83 → 3.94, `tail 0.1%` 13.37 → 13.87). So more data sharpens the spectrum and
does nothing for the marginals — consistent with reading 1.

**Context — `diff+coarse` still dominates it** (SR_E 1.00, L_eff 198, tail 5.05, opp-wind 0.19
at g=1.0). The plain generator's best is roughly the coarse downscaler's worst.

**Decision made by Claude (flagged):** archived all three reports under
`docs/benchmark-reports/` instead of letting `benchmark_v2_report.md` be overwritten — the
overwrite already cost the guidance-1.5 report once.

### CORRECTION (2026-08-05, same day): the matched-step claim above was WRONG

The reading above said *"at matched step count the 4-yr model's L_eff is 280.62 vs the 1-yr
model's 281 — 4× the data moved effective resolution by nothing."* **That is false.** The
"281" came from `idiff_m2cond_blocks.npz`, an **untagged block cache dated 2026-07-14**,
generated before the sampler settings were finalized on 07-29/30. It was never comparable.

Re-scored the 1-yr run at 100k and 200k under **identical current settings** (fresh 112-block
sets, tags `1yr_100000`/`1yr_200000`; reports in `docs/benchmark-reports/`):

| metric | 1yr 100k | 1yr 200k | 1yr 300k | 4yr 100k | 4yr 250k | 4yr 500k |
|---|---|---|---|---|---|---|
| SR_E | 1.62 | 1.64 | 2.04 | 1.83 | **1.53** | 1.53 |
| L_eff (km) | 336.74 | 420.93 | 481.06 | **280.62** | 336.74 | 374.16 |
| W1 u | 3.88 | 4.01 | 3.83 | 3.91 | 3.82 | 3.94 |
| tail err 0.1% | 13.77 | 13.36 | 13.37 | 13.85 | 13.54 | 13.87 |
| opp-wind frac *(≈0.20)* | 0.03 | 0.05 | 0.03 | 0.02 | 0.04 | 0.02 |

**Corrected reading — 4× data helps the LEVEL and the RATE, but not the DIRECTION:**
- At matched 100k, `L_eff` **336.74 → 280.62 (17% better)**. More data genuinely helps.
- The whole 4-yr curve sits **below** the 1-yr curve, and the degradation **rate** falls from
  **72.2 km per 100k steps to 22.5 — 3.2× slower.**
- **But both runs still degrade monotonically with training.** 4× data did not change the
  sign, only the slope. Two independent runs on different datasets show the same pathology,
  which is *stronger* evidence it is real than the single-run version was.
- So **"not data-limited" was too strong.** The defensible statement is: *the pathology is
  data-sensitive but not cured; extrapolating the 4-yr slope, it would need ~10× more data
  again to flatten, which is not a plausible path.* The cascade argument survives on the
  persistence of the trend, not on data being irrelevant.
- `SR_E` is not monotone in data: at 100k the **1-yr model is better** (1.62 vs 1.83); by the
  end the 4-yr model wins (1.53 vs 2.04). Level and rate move together only for `L_eff`.
- **Best checkpoint is metric-dependent** — 4-yr 100k wins `L_eff` (281), 4-yr 250k wins
  `SR_E`/tails. The earlier flat claim "best is 250k" holds only on the spectral+tail rows.

Unchanged by this correction: tails never move on any run; `opp-wind frac` stays 0.02–0.05
against a 0.20 reference on every checkpoint of both runs; `W1_cond` never beats white noise.

**Process lesson (second time this class of bug has cost something):** the block cache is
keyed by sampler settings, NOT by checkpoint or by code version — so an old untagged cache
silently answers a question about a *different configuration*. I compared against a number
lifted from a prior handoff instead of re-measuring, and stated the result firmly. **Re-measure
the baseline whenever it is load-bearing; never quote a cached number across a settings change.**
