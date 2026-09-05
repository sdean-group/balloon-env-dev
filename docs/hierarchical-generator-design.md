# Hierarchical (coarse-to-fine) wind generator — design register

> **STATUS 2026-09-04: sections 0-7 superseded.** After reading NVIDIA's cBottle (arXiv 2505.06474)
> and meeting Sarah and Rohan, the plan is to adapt cBottle's whole-sphere coarse + patch
> super-resolution EDM cascade (Apache-2.0 code) to our 18-level stratospheric channels rather
> than build the cascade from scratch. Section 8 (mesh comparison, HEALPix recommendation) and
> section 7 (E0 measurements) remain valid evidence. Current plan: `docs/gameplan-2026-09.md`.


Started 2026-09-02. Status: **v0, for discussion**. Nothing below is an implementation
decision until its row says *Locked*. Supersedes `coarse-cascade-design.md` (kept for the
record; this file restarts the discussion from scratch).

Owners: Shaurya (generator design, final training, ablation hooks). Rohan owns the
simulation environment and globe visualisation and consumes the generator through §1.4.

Rule for this file: minimal design. A capability enters only with a rationale and an
evidence source. Every open row names the measurement or argument that would settle it.

---

## 0. Why a hierarchy at all — the go/no-go argument

### H1. A coarse stage is needed to get large-scale structure (jets) right

**Structural argument (the one that does not depend on any checkpoint).** The fine model
denoises 64-cell windows at 0.25°: a 16° footprint, about 1,800 × 1,400 km at 40°N. Under
InfiniteDiffusion every window draws its own synoptic state from
p(x | lat, lon, month, hour) and neighbours are reconciled only across the 50% overlap.
Blending guarantees *continuity*, not *coherence*: nothing makes a trough at 130°W and a
ridge at 100°W belong to the same day, and a jet meander spanning 60–90° of longitude has
no mechanism to exist. Goslin's paper states the same failure and the same fix: MultiDiffusion
"produces repetitive results when poorly conditioned, a problem that InfiniteDiffusion
inherits," which is why Terrain Diffusion adds a coarse planetary model (§5.3).

**Empirical support already on the board** (benchmark v2, held-out days 8–14 of
Jan/Apr/Jul/Oct 2023, NE-Pacific box):

| row | SR_E ↓ | L_eff km ↓ | W1 u ↓ | tail 0.1% ↓ | jet err 99.9% ↓ | opp-wind err ↓ |
|---|---|---|---|---|---|---|
| self-split floor | 0.25 | 56 | 1.07 | 2.92 | 5.63 | 0.00 |
| plain conditional diffusion (m2cond 300k) | 2.04 | 481 | 3.83 | 13.37 | 13.01 | 0.17 |
| same 4-yr data, 500k | 1.53 | 374 | 3.94 | 13.87 | – | – |
| **same architecture given the true 2° field (m2coarse2, g1.5)** | **0.88** | **146** | **1.76** | **5.01** | **0.59** | **0.01** |

Three readings: (a) the plain model over-smooths and gets *worse* with training on both
datasets (L_eff 281→374 km on 4 yr; 4× data slows the drift 3× but does not reverse it);
(b) handed the correct coarse context, the identical architecture fixes most of every row,
so the fine-scale defect is largely a symptom of missing large-scale context; (c) seed
spread: m2cond reproduces 58% of ERA5's day-to-day variability — it hedges toward
climatology, which is what a window that cannot see the synoptic state must do.

**What the hierarchy does not fix by itself.** Stage 1 is also a diffusion model trained
by the same recipe and can drift the same way. The hierarchy moves the problem to a
smaller, better-posed one (a few thousand coarse cells, global data, ~180 translation
offsets per timestep in longitude), it does not dissolve it. Stage 1 is therefore gated on
dispersion and jet statistics (§4), never assumed.

**Missing evidence, cheap to produce (E0-a in §4):** the along-strip spatial
autocorrelation of the m2cond + InfiniteDiffusion field versus ERA5 at the same latitude.
Prediction: the model decorrelates at about the window scale, ERA5 at the synoptic scale.
That is the direct number for H1 and the spatial twin of sprint milestone 3.

### H2. Fine detail improves too

Supported directly by the m2coarse2 row above, *conditional on the coarse field being
right*. The risk is distribution shift: Stage 2 was trained on exact block means of truth
and will be fed generated coarse fields (decision D6).

### H3. Forecast / ground-truth split for free

Coarse field C = the forecast; fine field x = realised truth; the exact block-mean
projection makes blockmean(x) = C. The hierarchy gives a *consistent* forecast/truth pair
with zero error at coarse scales. Two honest caveats: real forecasts also err at resolved
scales, so the RL phase will still layer a forecast-error model on top (BLE adds noise to
its forecast; we can do the same to C), and operational forecasts (GFS 0.25°) are finer
than 2°. So the hierarchy buys the *split and its consistency*; the error model is a small
separate downstream decision and must not drive the choice of coarse resolution.

### Alternatives considered

- **Bigger fine window.** 256² at 0.25° costs 16× per sample and is still bounded at 64°.
- **One model with a wide low-res context input.** The context still has to be generated
  somewhere; that generator *is* Stage 1. Same design, less modular.
- **Stage 1 = ERA5 replay** (real coarse days, generated detail). A legitimate simulator
  product: unbounded in space, real jets, ~35k coarse states from 4 years. Loses novel
  weather and the "generator" claim. **Kept as the benchmark upper-bound row and as the
  product fallback if Stage 1 fails its gate.** This is what makes the plan safe.
- **Three-level cascade, latent diffusion, consistency distillation.** All in the paper,
  all for interactive speed. Not needed for a research simulator. Excluded.

**Verdict (proposed): go**, with the gates in §4 and ERA5 replay as the fallback.

---

## 1. The mechanism (proposed lock)

Two stages, one machinery, two seeds.

```
(lat, lon, t) query, seeds (s1, s2)
        │
        ▼
 Stage 1 — COARSE GENERATOR (new)            p(C | lat, lon, t)
   C = u,v on 18 levels at R_c degrees, τ_c frames at Δt_c
   trained on coarsened ERA5 over a (near-)global domain
   unbounded via InfiniteDiffusion on the coarse lattice; seed s1
        │  C  (block means, normalised units)
        ▼
 Stage 2 — FINE RESIDUAL MODEL (exists: m2coarse2)   p(x | C, lat, lon, t)
   diffuses r = (x − U(C)) / scale; exact block-mean projection; CFG
   unbounded via the space-time InfiniteDiffusion wrapper on the fine lattice;
   each fine window reads its coarse crop from Stage 1's infinite tensor
   (infinite-tensor `args` / `args_windows` dependency); seed s2
        │
        ▼
   x  (u, v at 0.25°, 18 levels, hourly)
```

Invariants: the whole field is a deterministic function of (s1, s2); query order and
cache state never change a value; blockmean(x) = C exactly. Two sources of randomness
with distinct meanings: s1 picks the weather, s2 picks the sub-grid realisation.

What the simulator sees (§1.4 API, decision D8): `coarse(region, times, s1)` — cheap,
wide, the forecast; `fine(region, times, s1, s2)` — local, the truth.

---

## 2. Inherited and not reopened (locked by prior evidence)

- EDM objective and preconditioning; factorised space-time U-Net; per-(level, var)
  normalisation; lat/lon coordinate channels; annual/semiannual/diurnal harmonics.
- Coarsening = horizontal block mean (`avg_pool2d`), all 18 levels kept.
- Residual parameterisation with measured `coarse_scale`; exact nearest-lift projection;
  coarse dropout 0.1 + flag plane (CFG, guidance 1.5 as the headline setting).
- InfiniteDiffusion via `infinite-tensor`; Rohan's 4-D wrapper `spacetime_infinite.py`
  (on `origin/main`) for the fine stage.
- Data contract: days 8–14 of every month are never trained on; benchmark v2 is the
  scoreboard; every coarse row is read against `coarse upsampled`.

---

## 3. Open decisions, in the order we will take them

Order is by dependency and critical path, not importance. D1 first because CDS retrieval
is measured in days and gates everything downstream.

| ID | Decision | Status |
|---|---|---|
| D1 | Geographic domain and data plan for Stage 1 | open |
| D2 | Coarse resolution and coarsening-operator consistency | open |
| D3 | Stage 1 window geometry and spherical handling | open |
| D4 | Stage 1 temporal extent and cadence | open |
| D5 | Stage 1 model: architecture, capacity, conditioning | open (mostly inherit) |
| D6 | Stage 2 interface under *generated* conditioning | open |
| D7 | Stage 2 geographic generalisation | open |
| D8 | Seeds, determinism, and the simulator API | open |
| D9 | Evaluation protocol, pre-registered criteria, gates and budget | open |

### D1 — Geographic domain and data plan for Stage 1

*Why it matters.* Stage 1 must learn the planetary-scale state; the current 30°×30°
NE-Pacific box is 15×15 cells at 2° and cannot. Retrieval from CDS is serial (one active
request per account, tape-bound, 0.5–5 h per month-request observed) so this is the
critical path of the whole 1–2 week window.

*Options.* (a) Global ±60° at the coarse resolution, directly from MARS. (b) Full globe.
(c) Several regional boxes.

*Sizes at 2°, float16, u+v × 18 levels:* one ±60° step ≈ 0.8 MB → 4 years hourly
(days 8–14 excluded) ≈ 21 GB; 3-hourly ≈ 7 GB; 10 years hourly ≈ 53 GB. Disk is not the
constraint; MARS wall time is (≈ 48 monthly requests ≈ 4–8 days for 4 years).

*Evidence that settles it.* None needed beyond the sizes; the choice is about the
spherical handling (D3) and the ±60° cap versus the regions Rohan's simulator will use.

*Lean.* (a): global ±60°, 2°, hourly (a superset; cadence for D4 is chosen later by
subsampling), years 2020–2023, same day-8–14 exclusion. Submit a **one-month sample first**
(hours, not days) to run the E0 measurements and test the pipeline, then the full pull.
Check whether Rohan's Dean-cluster CDS account can run a second stream in parallel.

### D2 — Coarse resolution and coarsening-operator consistency

*Why it matters.* Stage 2 is validated at factor 8 (2°). Coarser leaves Stage 2 more to
invent (variance not explained by the bilinear lift: 2.6% at 2°, 7.6% at 4°); finer makes
Stage 1 larger and pushes Stage 2 toward pure texture. Separately, the operator must be the
*same* function at both stages: Stage 2 expects the 8×8 block mean of the 0.25° field,
whereas a MARS request at `grid=2.0/2.0` returns a spectral-transform regrid of spectral
u/v, not a block mean.

*Options.* Resolution: 2° (factor 8) vs 4° (factor 16). Operator: MARS 2° direct; MARS 1°
then a local 2×2 mean; or download finer and block-mean locally (0.5° global hourly is
≈ 340 GB — out).

*Evidence that settles it.* On the NE-Pacific box, for a few held-out days, compare
MARS-2° and MARS-1°→2×2-mean against the true 0.25°→8×8 block mean. If the RMS discrepancy
is small against the residual scale 0.139 (normalised units) it is absorbed by D6's
conditioning-noise augmentation; if not, the operator is redefined as "MARS 2°" and Stage 2
is retrained against that same field for the box (a tiny extra download).

*Lean.* Keep 2° (one variable at a time; Stage 2 is proven there). Treat 4° strictly as a
later ablation. Run the operator measurement before any Stage 1 training.

### D3 — Stage 1 window geometry and spherical handling

*Why it matters.* The window sets the largest structure Stage 1 can make coherent without
relying on blending. Planetary wavenumbers 4–7 at 40°N are 60–90° of longitude.

*Options at 2°.* (a) Square 32² = 64°×64°, tiled in both axes (needs a rule for the
poles and for latitude-dependent cell size). (b) **Full-latitude strip**: 61 rows (60°S–60°N)
× 32 columns (64°), tiled only in longitude (periodic, period 180 cells) and time. (c) Full
globe 61×180 (≈ 2.7× the pixels of the current fine model), tiled only in time.

*Spherical handling.* Plate carrée with the latitude channel carrying the cos-latitude
anisotropy (the regional model already spanned cos 0.57–0.91 this way); ±60° cap as in the
paper; periodic longitude by making per-tile seeds and conditioning periodic in the tile
index (then the blended field is periodic by construction). Equal-area regridding is the
paper's choice but couples to the 0.25° fine grid; excluded from v1.

*Evidence that settles it.* From the one-month 2° sample: zonal autocorrelation length of
u at three levels by latitude band (does 64° of longitude contain a decorrelation length?),
and the meridional structure (does the jet system need the full band?).

*Lean.* (b), the strip: complete meridional structure in one window, no polar tiling, the
tiling problem collapses to 1-D longitude + time, pixel count (≈1,950) is half the current
model's so batch and speed are unchanged. (c) is the fallback if the autocorrelation
measurement says 64° is too short; it costs ~2.7× per step and removes translation
augmentation in longitude.

### D4 — Stage 1 temporal extent and cadence

*Why it matters.* Balloon episodes are days long (BLE's wind field spans 48 h); the fine
model emits 4-hour blocks and its temporal rows are the weakest on the board (SR_time 5.46
vs floor 0.28; spread ratio 0.20 vs 1.04). Putting the slow synoptic evolution in Stage 1
gives the fine stage a time-consistent conditioner across blocks, so day-scale coherence
comes from Stage 1 and Stage 2 only has to be locally consistent.

*Options.* τ_c × Δt_c ∈ {4×1 h (mirror the fine model), 8×3 h (24 h), 8×6 h (48 h),
16×3 h (48 h)}. Also: how Stage 2 gets hourly coarse frames from a coarser cadence (linear
interpolation of C in time, trained-in so inference matches training).

*Evidence that settles it.* Temporal autocorrelation of the 2° field at our levels (from
the sample month): the block must span at least one decorrelation time to carry evolution
rather than persistence. VRAM: batch × τ_c × pixels must stay near the current 16×4×4096.

*Lean.* 8 frames × 6 h = 48 h (one BLE episode per block; batch 16 fits), tiled in time by
InfiniteDiffusion beyond that. Stage 2 conditioned on linearly time-interpolated C.

### D5 — Stage 1 model

*Why it matters.* One variable at a time: the architecture is the thing we understand.

*Options.* Reuse `SpaceTimeUNet` (128 ch, mult [1,2,2,2], attention at the two coarsest
levels → global receptive field inside the window) with the same conditioning
(lat/lon channels, time harmonics). Note the paper's coarse model deliberately *limits* its
receptive field because a user sketch supplies the planetary layout; we have no sketch, so
we want the opposite.

*Lean.* Reuse unchanged, including the same optimiser and 300k schedule; snapshot every
25k and score as a curve (the L_eff drift showed the last checkpoint is not the best).
Change architecture only if G1 fails on a receptive-field symptom.

### D6 — Stage 2 interface under generated conditioning

*Why it matters.* Stage 2 has only ever seen exact block means of truth. Generated C will
have a slightly different spectrum, bias and smoothness; cascaded diffusion (Ho et al.) and
the paper's inference-time corruption both exist because this shift breaks naïve cascades.

*Options.* (a) None: test first. (b) Gaussian conditioning-noise augmentation at training,
noise level as an embedding (Ho et al.), calibrated to measured Stage 1 error. (c) Train
Stage 2 on Stage 1 samples — impossible without paired truth.

*Evidence that settles it.* E2 in §4: feed Stage 1 samples over the box to the *existing*
m2coarse2 and score; measure the conditioner shift directly (spectrum and marginals of
generated C vs true block means). The size of the drop from the real-coarse row is the
number that says whether (b) is needed and how strong.

*Lean.* Measure with (a); plan (b) for the Stage 2 retrain regardless, because it is also
what lets the simulator feed a *perturbed* forecast later.

### D7 — Stage 2 geographic generalisation

*Why it matters.* The "global" claim is only as strong as Stage 2 outside its training box.
Stage 2 currently carries lat/lon channels and was trained on one box.

*Options.* (a) Keep the NE-Pacific model for v1 and test it on another box. (b) Retrain on
2–4 boxes at different latitudes with one held out. (c) Drop coordinate channels from
Stage 2 (paper-style location-agnostic refinement; the coarse field carries "where").

*Evidence that settles it.* Score the existing Stage 2 with real coarse on a second box
(the honest generalisation test). Rohan's Dean cluster may already hold 2018–2021 hourly
data for the box; check what other regions exist before downloading.

*Lean.* (a) for the first cascade result; (b) folded into the D6 retrain if the second-box
score is materially worse than in-box. (c) is a recorded ablation, not v1.

### D8 — Seeds, determinism, and the simulator API

*Why it matters.* This is the contract Rohan builds against. Both stages are infinite
tensors; Stage 2's windows depend on Stage 1 through `args_windows` with the coarse
geometry (fine window 64 ↔ coarse crop 8; the library supports cross-tensor windows).

*Lean.* Two seeds (s1 weather, s2 realisation); per-tile seeds hashed from (seed, tile
index) with the longitude index taken modulo the period; one `NormStats` for the cascade;
public calls `coarse(...)` and `fine(...)` returning m/s with lat/lon/time coordinates.
Determinism and order-independence are asserted by test, as the existing wrappers do.

### D9 — Evaluation protocol, pre-registered criteria, gates, budget

See §4. To be locked before Stage 1 training starts.

---

## 4. Experiment order, gates, budget (proposed)

Costs: the fine model runs at 2.7 it/s on a Kahan gpu48 slice → ~31 GPU-h per 300k steps.
Stage 1 (strip, τ_c = 8) has the same pixel-frame count → assume the same. Two gpu48 slices
exist. Disk quota ≈ 100 GB (≈ 74 used in August; prune old snapshots first).

| step | what | needs | gate |
|---|---|---|---|
| E0-a | along-strip autocorrelation of m2cond+tiling vs ERA5 (H1 evidence) | local ckpt, Mac | informative |
| E0-b | MARS-2° vs 0.25°→8×8 block-mean discrepancy on the box (D2) | 1-month 2° sample | ≪ 0.139 ⇒ MARS-2° is the operator |
| E0-c | zonal/meridional/temporal autocorrelation of 2° ERA5 at our levels (D3, D4) | same sample | sets window and τ_c |
| E1 | train Stage 1; score at 2° as a curve | full 2° pull | **G1**: beats fitted-noise baselines on the spectral rows at 2°; zonal-mean u and jet latitude/strength distributions within the self-split floor; seed-spread coverage ≥ 80% of ERA5 day-to-day |
| E2 | cascade with the *existing* Stage 2 over the box; measure conditioner shift | E1 ckpt | **G2**: beats m2cond on every spectral and tail row; retains ≥ 80% of the real-coarse improvement on those rows |
| E3 | retrain Stage 2 with conditioning-noise augmentation (± extra boxes, D7) | E2 numbers | **G3**: ≥ E2 on every row; held-out box within the floor of in-box |
| E4 | unbounded demo, eye test, API handoff to Rohan | E3 | — |

Rows on the final board: floor · fitted noise · plain diffusion (m2cond) · Stage 1 + bilinear
(no Stage 2) · **full cascade** · real coarse + Stage 2 (upper bound = ERA5 replay).
Coverage statistic (seed spread / ERA5 day-pair spread) becomes a headline row.

If G1 fails after one architecture change, ship ERA5 replay as Stage 1 (§0) and say so.

---

## 5. Repo state that must be fixed before any cascade code

- `wind-eval-harness` imports `spacetime_infinite` and `infinite_coordinates` (in
  `scripts/gen_temporal_sequences.py` and `scripts/zoom_montage_conditional.py`) but neither
  module exists on this branch; both live on `origin/main` (Rohan). The poster's temporal
  rows were produced with them.
- `origin/main` is 14k lines ahead (Rohan: 4-D wrapper, CFGD, tile-scaling benchmarks,
  multi-year data pipeline), touching `data.py`, `train.py`, `spacetime.py` — exactly the
  files the cascade edits. Merge or rebase **first**, as a mechanical step, before writing
  Stage 1 code, or every later merge fights the cascade diff.
- `docs/coarse-cascade-design.md` and this file are untracked; commit design docs with the
  code that implements them.

## 6. Out of scope for this cycle

Latent diffusion, consistency distillation, a third cascade level, equal-area regridding,
rotation augmentation, new architectures, forecast-error modelling inside the generator.

---

## 7. E0 measurements (2026-09-02, local Mac; scripts in the session scratchpad)

### 7.1 Feasibility probe — the existing 41.9M-param `SpaceTimeUNet`, fwd+bwd, batch 1, MPS

Relative cost is what transfers to Kahan; the anchor (64×64, τ=4, batch 16) is known to fit a
gpu48 slice at 2.7 it/s. Max batch on gpu48 ≈ 16 / (rel mem).

| block (H×W×τ) | pixel-frames | rel mem | rel time | ≈ max batch on gpu48 |
|---|---|---|---|---|
| fine anchor 0.25° 64×64×4 | 16,384 | 1.00 | 1.00 | 16 |
| coarse square 2° 32×32×8 | 8,192 | 0.59 | 0.45 | ~27 |
| **coarse strip 2° 64×32×8** | 16,384 | 0.99 | 0.78 | 16 |
| coarse strip 2° 64×32×16 | 32,768 | 1.27 | 1.48 | ~12 |
| coarse strip 2° 64×48×8 | 24,576 | 1.25 | 1.11 | ~12 |
| coarse globe 2° 64×176×8 | 90,112 | 2.64 | 3.81 | ~6 |
| coarse globe 4° 32×88×8 | 22,528 | 1.36 | 1.04 | ~11 |

Rule of thumb confirmed: cost ∝ B·τ·H·W with a mild attention surcharge for wide blocks.
Architectural constraint: H and W must be divisible by 8 (three stride-2 stages), so the
latitude cap is ±56° (56 rows) or ±64° (64 rows) at 2°, and a periodic longitude lattice
needs its period to be a multiple of the window stride.

### 7.2 Decorrelation scales of ERA5 at 2° (NE-Pacific box, 2023, levels 49/57/66, 3-hourly)

Synoptic anomaly (15-day running mean removed): temporal e-folding **30–33 h** for both u
and v at all three levels; correlation reaches ~0.1 by **45–54 h**; ~−0.2 at 96 h (the
wave passes). Only 20–38% of the 2° anomaly variance is synoptic (<15 d); the rest is
seasonal/intraseasonal and slower than any block we can afford.

Spatial (anomaly from annual mean): meridional e-fold 5–11 cells (10–22°); zonal
correlation of u is still 0.45–0.80 at 10 cells (20°) — the 30° box is too small to see
the zonal decorrelation length; needs the global sample.

Cadence from advection: mean speed 7–19 m/s by level → a 2° cell is crossed in **3–8 h**, a
0.25° cell in **0.4–1.0 h**. A 3-tap temporal kernel sees ≈1 cell of motion per frame at
Δt_c ≈ 6 h (coarse) and Δt = 1 h (fine) — the cadences match the resolutions.

Note: level 49 shows a correlation dip at 6 h that recovers at 24 h (tidal signal); the
diurnal harmonic conditioning already covers it.

### 7.3 ARCO-ERA5 (Google Cloud public archive) — the CDS queue is avoidable

`gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1`: global 0.25°, hourly,
all 137 model levels, float32, chunks (1 h, 18 levels, full globe); ~55 MB compressed per
chunk (1.4×). Our levels 49–66 straddle two level-chunks, so one hour of global u+v costs
4 chunks ≈ 220 MB. Measured 12.4 MB/s single stream from the Mac.

| pull | transfer | single stream | 4 streams |
|---|---|---|---|
| 4 yr hourly (days 8–14 excluded) | ≈ 6 TB | ≈ 5.6 d | ≈ 1.4 d |
| 4 yr 3-hourly | ≈ 2 TB | ≈ 1.9 d | ≈ 11 h |
| 4 yr 6-hourly | ≈ 1 TB | ≈ 22 h | ≈ 6 h |

Consequences: (a) the 2° field can be produced by the **exact 8×8 block mean** of the
0.25° field, so the D2 operator question disappears; (b) chunks are whole-globe, so a
regional fine-stage box costs the same transfer as the globe — one streaming pass can emit
the global 2° store (≈21 GB float16 hourly) *and* any regional 0.25° boxes (≈4.3 GB per
box-year float16) at once; (c) storage of the full 0.25° globe (≈2 TB) is out of scope,
stream and discard. Kahan compute-node bandwidth to GCS is unmeasured (1-minute probe).

---

## 8. Spatial discretisation — the mesh question (2026-09-02, revises D3)

Shaurya asked for uniform meshes to be considered seriously, with the code overhaul
accepted if the result is more elegant and defensible. Precedents checked: GraphCast /
GenCast (icosahedral multi-mesh, graph transformer, diffusion in GenCast), AIFS (octahedral
reduced Gaussian graph), Pangu / FuXi / Aurora / SEEDS (equiangular lat-lon), FourCastNet-
SFNO (lat-lon + spherical harmonics), DLWP / StretchCast (cubed sphere), DLWP-HPX, PEAR,
DLESyM and **cBottle** (HEALPix). InfiniteDiffusion's own equal-area tiling is a training-
data normalisation for a *flat* infinite world; it does not tile a sphere at generation time.

### 8.1 What our pipeline needs from a mesh

| | R1 uniform area & shape | R2 exact dyadic coarse↔fine | R3 rectangular local windows (conv U-Net, windowed tiling) | R4 closed globe, poles included | R5 cheap lat/lon query, spectra | R6 precedent for *generative* global use | R7 data cost |
|---|---|---|---|---|---|---|---|
| lat-lon (plate carrée) | ✗ 1/cos φ, 2× at 60° | ✓ | ✓ | ✗ needs a cap | ✓ | ✓ (non-tiled global models) | native |
| icosahedral mesh (GraphCast/GenCast) | ✓ | ~ triangles, 4:1 | ✗ graph, not conv; no windowed precedent | ✓ | needs learned encoder/decoder | ✓ GenCast | learned regrid |
| cubed sphere | ~ area varies ~1.3×; polar faces rotate east | ✓ | ✓ 6 faces, 8 three-face corners | ✓ | ✓ | DLWP 2020 | regrid |
| **HEALPix (nested)** | ✓ exactly equal area; east is "to the right" in every face | ✓ each pixel = 4ᵏ children, block mean exact | ✓ 12 square faces + halo padding; 8 three-face vertices at ±41.8° | ✓ | ✓ healpy `ang2pix`, bilinear; spherical-harmonic spectra native | ✓✓ cBottle = global coarse diffusion + patch super-res cascade; DLWP-HPX; PEAR | bilinear regrid 0.25°→nside 256 (0.23°) |

DLWP-HPX's reason for abandoning the cubed sphere applies to us directly: on HEALPix every
cell has the same east-west orientation, so one location-invariant kernel serves the whole
sphere; the cubed sphere needed separate kernels for polar and equatorial faces, and the
switch to HEALPix improved skill at fewer grid points. PEAR reports the same architecture on
HEALPix beating its lat-lon twin (MSLP ACC 0.790 vs 0.675 at 5 days) with 8× fewer params.

### 8.2 cBottle is the closest precedent to this design

NVIDIA's Climate in a Bottle (arXiv 2505.06474): a **whole-sphere coarse diffusion model
on HPX64 (~100 km)** conditioned on day-of-year and time-of-day (plus SST), followed by a
**patch-based super-resolution diffusion model** run as multidiffusion over the sphere —
default patch 128 px, overlap 32 px, patches cut from the `earth2grid.healpix.pad`-padded
12-face image with `unfold`, stitched by weighted averaging, the low-res conditioner
patchified identically so parent pixels align exactly. Their data path is ours: ERA5
regridded bilinearly to nside 256, then **average-pooled to nside 64 for the coarse model**.
That is a block-mean hierarchy on the nested grid. So a two-stage coarse-to-fine diffusion
cascade on HEALPix with padded-face patches is published, working practice at a much larger
scale than ours.

### 8.3 Recommendation (proposed lock for D3): HEALPix, nested, nside 256 fine / 32 coarse

- Fine grid nside 256: 0.229° (~25 km), 786k pixels, faces 256×256. Coarse nside 32: 1.83°
  (~203 km), 12,288 pixels, faces 32×32. Factor 8 = three nested levels; `coarsen` becomes
  the nested 8×8 average and stays exact. Layout `(12, nside, nside)` in earth2grid XY order.
- **Stage 1 = whole-sphere model**: all 12 faces in one forward pass, per-face convolutions
  with `healpix.pad` halos, attention at the coarsest U-Net level across all faces (global
  receptive field). No spatial tiling, no latitude cap, no periodic-longitude hack; jets are
  globally coherent by construction. Cost: 12,288 px per frame ≈ the 64×176 globe row of the
  probe (rel mem 2.64, rel time 3.8) → batch ≈ 6 at τ_c = 8 on a gpu48 slice. Tiled only in
  time.
- **Stage 2 = 64×64 windows on the 256-face grids** with halo padding, InfiniteDiffusion
  over the finite sphere and in time; each window's conditioner is its 8×8 nested parent
  block. This is cBottle's SR stage with our residual parameterisation and projection.
- u, v stay eastward/northward components: HEALPix's uniform orientation makes the relation
  to face axes the same everywhere, so no vector rotation is needed (unlike the cubed sphere).
- Coordinate channels (lat, lon per pixel) and cyclic time harmonics unchanged.
- Sim query: `ang2pix` + bilinear over the four nearest pixels; on HEALPix the balloon's
  altitude column is a single pixel index.
- Global spectra via spherical harmonics on HEALPix (a better global metric than box PSDs).

### 8.4 Costs and warts, stated

1. **Regridding.** ERA5 0.25° → nside 256 bilinear (cBottle's choice). Measure the
   round-trip loss on the box spectra (E0 item); expect a small loss at the finest scales.
2. **Eight three-face vertices at ±41.8°.** `healpix.pad` fills corner halos by a traversal
   rule; a fine window whose interior contains a vertex has one fabricated quadrant. Rule:
   vertex-centred windows get zero blend weight on the fabricated quadrant; DLWP-HPX and
   cBottle accept the same corner fill at every conv layer. The whole-sphere coarse stage has
   no windows and no such issue. Register a per-vertex seam check in the benchmark.
3. **Compute.** Stage 1 whole-sphere is ~5× per sample vs a strip and has no crop
   augmentation; mitigate with gradient accumulation and more years at 2° (storage is
   0.8 MB per step; transfer is the only cost).
4. **Retraining.** m2coarse2 was trained on lat-lon crops; treat it as a warm start at best.
5. **Engineering.** `earth2grid` (Apache-2.0, NVIDIA): source install needs a CUDA toolchain
   for the fast padding kernel but ships a `pure_python` padding backend; CPU/MPS use is
   untested here. `healpy` has wheels for macOS arm64 and Linux. The sphere tiler replaces
   the infinite lattice indexer: windows are `unfold`s of the padded 12-face image (as in
   cBottle) with deterministic per-window seeds; determinism and order-independence tests
   carry over. Benchmark v2 metrics run on HEALPix where they are pointwise (marginals,
   vertical, structure) and on a lat-lon regrid of both model and ERA5 for the box PSD rows.
6. **Storage for fine training data.** Global fine faces are 57 MB per hour float16; 4 years
   3-hourly ≈ 500 GB — fits Rohan's `/share/dean` (1.2 TB), not Kahan's 100 GB. Alternative:
   keep fine data for a subset of faces or hours; Stage 2 trains on random 64×64 patches from
   anywhere on the sphere (location-agnostic, the paper's factorisation).

### 8.5 Why not the others

- **Lat-lon**: workable with a cap and a latitude channel, but it is the option every
  uniform-mesh paper exists to escape, and the cap means the product is not global.
- **Icosahedral**: the most uniform, but it is a graph; it discards the conv U-Net, the
  residual block-mean contract on squares, and windowed lazy generation, with no precedent
  for the last.
- **Cubed sphere**: HEALPix dominates it on uniformity and orientation, per DLWP-HPX.
- **Equal-area longitude stretching (InfiniteDiffusion paper)**: a data-prep trick for a flat
  world; not a spherical tiling.

Decisions this revises: D1 (data pipeline emits HEALPix faces), D2 (operator = nested
average, no MARS ambiguity), D3 (this section), D5 (per-face conv + `pad`, whole-sphere
Stage 1), D8 (window indexing on faces), D9 (add vertex seam check, spherical spectra).
