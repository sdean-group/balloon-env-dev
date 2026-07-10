# Benchmark v2 — change log & design decisions

*Tracking doc for the metrics overhaul of 2026-07-10. The old reference-free 0–1 suite is
scrapped in favor of the reference-based suite specified in `Metrics & Baselines (2).pdf`.
Every change to the repo made during this overhaul is recorded here, along with who made
each design decision and why.*

## Decisions made by Shaurya (2026-07-10)

1. **Clean break with the old code.** `metrics/` is rewritten from scratch around the PDF
   spec; the five `benchmark_*.py` scripts are replaced by one unified runner. Old code
   lives only in git history (branch `wind-eval-harness`, commits before this date).
2. **Raw values only.** Every metric is reported as its raw value (spectral residuals,
   W1 distances in m/s, L_eff in km, …), each labeled lower/higher-better. No 0–1
   normalization, no composite — the clip constants and composite means of the old suite
   were exactly the opaque part.
3. **Held-out ERA5 reference.** The trained model saw days ~15–28 of Jan/Apr/Jul/Oct 2023
   (`era5_train.zarr`). The reference for all metrics is the **non-overlapping days 8–14**
   of each seasonal block, sliced from `era5_temporal.zarr` (hourly, NE Pacific,
   levels 49–66) → cached as `data/era5_heldout.zarr`.
4. **Procedural axis reduced to the tiling penalty.** Seam/revisit/budget/extent scores are
   dropped. A tiled generator is run once at single-tile size and once at multi-tile size;
   the suite is computed on both and the difference is the tiling penalty (per metric).
5. **Baselines distilled (2026-07-10, later the same day).** Default board = self-split
   floor (the scale for every raw metric), white noise (trivial spectral anchor; at-floor
   on marginal W1 by construction), phase shuffle (its complement: right stats / zero
   structure — the one anchor that fires the distribution metrics and brackets the
   diffusion failure mode), BLE-VAE (prior state of the art), and the model. Dropped:
   idiff toy (machinery validation is done; the tiling penalty now checks the real model)
   and time-shuffled (its calibration role lives permanently in `test_metrics_v2.py`).
   Kinematic toy is gated behind `--temporal` and returns when M2/M3 checkpoints land.
   This resolves the PDF's empty Baselines section.

## Decisions made by Claude (flagged for review — push back on any of these)

1. **Spectral binning across unequal grids.** Pred and ref can have different crop sizes
   (192² vs 121²). Spectra are computed on each grid's native isotropic shells in
   *physical* wavenumber (cycles/m, from the 0.25° spacing), then the pred log-PSD is
   linearly interpolated onto the ref's shell centers over the overlapping k-range before
   the residual/ratio is taken. Density-normalized periodogram so grid size cancels.
2. **Kaiser window (β=8) before every FFT** — the PDF's "Cool: multi-taper/Kaiser" note,
   applied as the default (identically to pred and ref, so it's fair). Power is
   renormalized by the window's mean-square so amplitudes stay comparable.
3. **Log-PSD averaging.** Per-(time,level) spectra are averaged in log space (geometric
   mean) before the residual — standard for red spectra, keeps one outlier frame from
   dominating.
4. **L_eff convention.** R(k)=E_pred/E_ref; L_eff = 1/k* at the smallest k where R<0.5 for
   5 consecutive shells (PDF rule). If R never dips below 0.5, L_eff is reported as the
   grid-Nyquist wavelength (2·Δx ≈ 42 km) — "resolved to the grid", not a failure.
5. **Vertical shear Δz.** Generated artifacts carry only u,v — no T/q/sp to integrate real
   altitudes. Layer thicknesses Δz per model-level pair are precomputed once from the ERA5
   stage-2 data (hypsometric, then averaged over time/space) and applied identically to
   both sides of the W1. Caveat: that thickness climatology comes from the SF box; layer
   thickness varies little horizontally, but this should move to the NE-Pacific box when
   T/q/sp are downloaded for it. Shear reported in (m/s)/km.
6. **Temporal PSD comparison.** The PDF asks for the per-gridpoint temporal power spectrum
   (graph). For a scalar we reuse the PDF's own spectral-residual idea in time:
   RMSE of log temporal-PSD over the common frequency range → `SR_time`.
7. **Trajectory dispersion protocol.** Per level: a regular lattice of passive tracers,
   advected by bilinearly-interpolated (u,v) with RK2 steps at the artifact's native dt,
   positions in meters (tracers may leave the crop; velocity sampling clamps to the edge —
   identical protocol both sides). Reported: RMSE of log MSD(t) vs ERA5 (`disp log-MSD
   RMSE`) and the final-spread ratio pred/ref averaged over levels.
8. **Extreme quantiles** at 0.1/1/99/99.9% per level & component; reported as the mean
   absolute error vs ERA5 across levels, separately for the 1%-pair and the 0.1%-pair
   (the PDF's "W1 misses the tails" complaint).
9. **Conditional-distribution metric: machinery only, degenerate for now.** The current
   model is *unconditional* (location/season conditioning is a future layer), so "fix the
   condition" degenerates to one condition = the whole training climate. Implemented as
   N-seed pooled per-level W1 vs the held-out reference (`W1_cond`); becomes the real
   per-condition average once conditioning exists.
10. **Sampling-noise floor row.** The report includes an `era5 self-split` row — days 8–10
    scored against days 11–14 of the held-out set — so every metric has an empirical
    "same-distribution" floor to read the other rows against.
11. **BLE-VAE row kept with a caveat.** Its artifacts are the SF box (37×41), a different
    region/climate than the NE-Pacific reference; spectra compare fine in physical k, but
    distribution metrics partly measure climate mismatch. Flagged in the report.
12. **gate.py (cluster training gate) rewritten** onto the new metrics: the gate is now
    SR_E + marginal-W1 of sampled crops vs reference statistics computed from the training
    zarr (which the cluster already has), still standalone-loadable (no jax). The old
    COMPOSITE≥toy criterion is gone. **Heads-up: rsyncing this to Kahan changes what the
    train→gate loop prints; in-flight sbatch jobs keep their local copy until you sync.**

## Findings from calibrating the new suite (2026-07-10)

1. **The Kaiser window kills the phase-shuffle trap.** Phase-shuffle preserves the
   *rectangular-window* periodogram exactly (verified: SR_E = 0.000 without windowing) —
   the classic way a PSD metric gets fooled. But much of a real field's raw high-k
   amplitude is boundary *leakage*; the Kaiser window strips it from the real field and
   cannot strip it from the shuffled field (there the same energy has become genuine
   interior signal). Result: under the windowed estimator even SR_E catches phase-shuffle
   (5.65 vs floor 0.32). The old suite needed vorticity/divergence cross-structure to
   catch it; the new estimator catches it in the energy spectrum itself, and SR_div /
   SR_vort / shear-W1 catch it independently (5.3 / 5.8 / 40× floor).
2. **Marginal W1's designed blind spot.** White noise moment-matched per (time, level)
   scores *at the floor* on marginal W1 (0.86 vs floor 1.06) — by construction, since W1
   measures distribution placement, not structure. SR_E catches that anchor at 8.9. This
   is why the suite has multiple axes; encoded as an explicit test expectation.
3. **Trajectory dispersion does not catch time-shuffling** — the jet is quasi-stationary,
   so temporally scrambled frames still transport tracers roughly correctly (the
   project's standing "winds aren't passive tracers" finding, reconfirmed). SR_time is
   what catches temporal incoherence (3.68 vs floor 0.21); dispersion instead catches
   wrong *transport* (amplitude-halved winds: log-MSD RMSE 1.36 vs floor 0.05, spread
   ratio 0.57). Both roles are asserted in the calibration test.

## Known limitations / to be aware of

- The PDF's **Baselines section is empty** — current set: era5 self-split (floor),
  phase-shuffle, white-noise, BLE-VAE (caveated), InfiniteDiffusion toy + trained;
  temporal: kinematic toy, time-shuffled anchor, M2/M3 when checkpoints land.
- Temporal metrics need a datetime64 time axis and ≥16 frames; static artifacts get N/A.
- `era5_real_stage2.zarr` (SF box) is still used for the Δz climatology only.

## File-level change log

All paths under `src/eval/windeval/` unless noted. Everything deleted is in git history
(branch `wind-eval-harness`).

| Change | Files |
|---|---|
| NEW change-tracking doc | `docs/benchmark-v2-changes.md` (this file) |
| NEW metrics package (PDF spec) | `metrics/spectra.py` (PSD/SR/L_eff), `metrics/distributions.py` (W1/tails/conditional), `metrics/shear.py` (shear W1 + Δz climatology), `metrics/temporal.py` (temporal PSD, trajectory dispersion; segment-aware), `metrics/suite.py` (orchestration, METRIC_INFO, tiling_penalty), `metrics/__init__.py` |
| NEW held-out reference | `reference.py` (`build_heldout`, `split`); cached data: `data/era5_heldout.zarr`, `data/climatological_dz_m.npy` |
| NEW unified runner | `benchmark.py` (replaces all five old scripts); output `benchmark_v2_report.md`, figures `docs/figures/benchmark_v2/` |
| NEW calibration test (14 checks) | `tests/test_windeval/test_metrics_v2.py` |
| REWRITTEN training gate | `generators/infinite_diffusion/gate.py` — now SR/W1 vs training-crop stats + self-split floor; no pass/fail constant. **Sync to Kahan changes gate output format.** |
| UPDATED tests | `test_idiff_artifact.py` (self-suite check replaces field_scores), `test_temporal_baseline.py` (metric assertions moved to test_metrics_v2) |
| DELETED old metrics | `metrics/realism.py`, `metrics/procedure.py`, `metrics/vertical.py` (old `metrics/temporal.py` replaced in place) |
| DELETED old benchmarks + reports | `benchmark.py` (old), `benchmark_m0.py`, `benchmark_m1.py`, `benchmark_stage2.py`, `benchmark_ble.py`, `benchmark_temporal.py` + their six `*_report.md` |
| DELETED old tests | `test_amplitude_churn.py`, `test_stage2_metrics.py`, `test_helmholtz.py`, `test_procedure.py`, `test_benchmark_m0/m1/temporal.py` |
| UNCHANGED (metrics-independent) | `artifact.py`, `anchors.py`, `derive.py`, `l137.py`, `ingest_era5.py`, `resample.py`, all generator internals (`sampler.py` untouched as always) |
