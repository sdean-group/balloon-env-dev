# Coarse Cascade Design Register

Last updated: 2026-08-31

Purpose: settle the design before implementation. Nothing marked **open** or **proposed** is an implementation decision.

Status meanings:

- **Locked** — explicitly approved; changing it requires updating this file with a reason.
- **Proposed** — current recommendation, awaiting approval.
- **Open** — alternatives still need to be compared.

## Decision list

| ID | Decision | Status | Current recommendation |
|---|---|---|---|
| D0 | Product being built | Proposed | A two-stage generator: Stage 1 invents the coarse weather state; Stage 2 generates a fine realization. No ERA5 field is required at inference. |
| D1 | Stage 1 horizontal resolution | Proposed | Start at 2° (`factor=8`). Test 4° only as an ablation; do not infer the factor from the fixed-coarse 6.9% spread alone. |
| D2 | Stage 1 native output window | Proposed | **32×32 coarse cells at 2° = 64°×64°.** Generate larger/global regions lazily with InfiniteDiffusion rather than predicting one dense globe. |
| D3 | Geographic training domain | Open | Stage 1 needs substantially broader, ideally global, ERA5 training data. The current 30°×30° regional dataset is too small for D2. |
| D4 | Spherical geometry | Open | Periodic longitude plus a latitude-aware/equal-area tiling rule; decide how polar regions are handled before calling the model global. |
| D5 | Stage 1 temporal extent | Open | Prefer a synoptic horizon longer than four hours; candidate starting point is 24 hours at 3-hour cadence. |
| D6 | Vertical representation | Proposed | Generate u/v at all 18 levels in Stage 1. Vertical structure must be generated there, not copied from truth. |
| D7 | Stage 1 conditioning | Proposed | Location and cyclic time only; no real weather field. Independent coarse seeds represent different plausible weather states. |
| D8 | Stage 1 architecture | Open | Begin from the existing space-time EDM U-Net, resized for the coarse window; change architecture only if receptive-field or scaling tests fail. |
| D9 | Stage 2 target/interface | Proposed | Keep the measured residual target `(x-U(c))/coarse_scale`, all-level coarse input, bilinear lift, and exact block-mean projection. |
| D10 | Robustness to generated conditioning | Open | Train Stage 2 with structured conditioner perturbations calibrated to Stage 1 errors, plus flagged coarse dropout. Specify the perturbation mixture after Stage 1 is measured. |
| D11 | Cascade sampling and seeds | Open | Separate coarse and fine seeds; cache Stage 1; align Stage 1/Stage 2 windows so query order cannot change the result. |
| D12 | Normalization and scale transfer | Open | Decide whether Stage 1 uses the existing per-level normalization and how Stage 1 outputs are converted exactly into Stage 2 conditioning units. |
| D13 | Training data and splits | Open | Define global years, held-out dates/regions, leakage rules, and whether Stage 1 and Stage 2 share the same split. |
| D14 | Evaluation and success criteria | Proposed | Compare plain diffusion, generated coarse + bilinear, full cascade, and real-coarse upper bound. Add total/between/within variability, spectral-seam, residual-leakage, and longer-horizon metrics. |
| D15 | Experiment order and compute gate | Proposed | Validate Stage 1 first; test it through the existing Stage 2; retrain Stage 2 only after measuring conditioner shift; run factor 16 only if factor 8 fails the registered criteria. |

## D2 rationale: Stage 1 output size

- **Native model window is not maximum generation size.** InfiniteDiffusion can tile a fixed 32×32 model window into an arbitrarily large coarse field.
- At 2°, 32×32 covers 64°×64°: roughly 7,100 km north-south and 5,450 km east-west at 40°N, enough context for large jet segments and synoptic waves.
- It has one quarter the spatial pixels of the current 64×64 fine model, leaving room for temporal context or additional capacity.
- A single dense 90×180 global output is not the recommended first design: it couples the project immediately to spherical topology, polar distortion, and global-data requirements.
- D2 cannot be locked independently of D3–D5: global data coverage, spherical tiling, and temporal horizon must fit the same window choice.

## Approval order

Resolve in this order: **D2 output window → D3–D4 global domain/geometry → D5 temporal extent → D1 resolution → D6–D8 Stage 1 model → D9–D12 interface/training → D13–D15 protocol and gates.**
