# wind-eval

Evaluation harness for conditioned wind-field generators (stratospheric-balloon RL).
Decouples generation from evaluation through a frozen artifact format:

```
generator → WindArtifact (zarr) → metric suite → calibration / leaderboard
```

Specs: `../artifact-format-spec.md`, `../reference-data-pipeline-spec.md`.

## Layout
- `windeval/artifact.py` — WindArtifact read/write (locked schema, capability flags).
- `windeval/ingest_era5.py` — ERA5 model-level GRIB → real-ERA5 anchor.
- `windeval/anchors.py` — phase-shuffle + white-noise anchors (pure functions of the ingest).
- `windeval/metrics/` — Axis-1 field-quality metrics + distances.
- `windeval/benchmark.py` — calibration runner (real / phase-shuffle / noise separation).

## Run the calibration benchmark
```bash
../.venv/bin/python -m windeval.benchmark [path/to.grib]   # defaults to ../sf_ml_test.grib
```
Writes artifacts to `data/*.zarr` and a report to `benchmark_report.md`.

## Axis-1 metrics — OBJECTIVE scoring (no distance-to-ERA5)
Each field is scored against physically-motivated ideals; ERA5 is a strong **peer row**,
not the answer (correct framing for conditional generation — we want plausible samples,
and a generator may legitimately match/exceed reanalysis-level self-consistency).

Minimal elegant set: **spectrum slope** (necessary gate, fool-able), **velocity-increment
kurtosis** (intermittency — the robust discriminator), plus **Helmholtz rot. fraction** and
**vort/div ratio** as caveated diagnostics. Composite = spectrum + intermittency.

## Calibration result (validated, 1 SF timestep)
Scoreboard ranks real (0.953) > phase-shuffle (0.564) > noise (0.004). Key findings:
- **Intermittency is THE reference-free discriminator**: real kurtosis 5.72 vs shuffle
  3.38 ≈ noise 3.03. Phase-randomization Gaussianizes increments toward 3.
- **Spectrum slope is necessary but fooled** (real ≈ shuffle = −3.95).
- **Helmholtz/vort-div are weak here** (real 0.63 vs shuffle 0.60): rotational character
  is mostly spectrum-encoded, and the FFT decomposition is biased on a bounded box.
  Demoted to diagnostics until a non-periodic solver lands.

## Stage 2 — vertical + temporal (validated)
24 hourly SF timesteps + `lnsp` → per-column pressure (`a+b·sp`, L137 coeffs) and
altitude (hypsometric). New modules: `l137.py`, `derive.py`, `metrics/vertical.py`,
`metrics/temporal.py`, `benchmark_stage2.py`. Run:
```bash
../.venv/bin/python scripts/extract_era5.py 2023-01-01 24   # one-time pull
../.venv/bin/python -m windeval.benchmark_stage2
```
Headline scores (real / phase-shuffle / noise):
- **vertical coherence** 0.81 / 0.01 / 0.00 — control axis; shear 8.6 vs 72/21 (m/s)/km
- **temporal persistence** 0.92 / 0.01 / 0.00 — tendency 1.8 vs 25/6.9 m/s/h
- **spatial composite** 0.93 / 0.57 / 0.00 — spectrum still fooled by phase-shuffle,
  but vertical+temporal catch it → **the full suite is robust where any single axis isn't.**
- real ERA5 drift slope ≈ 0 (machinery validated; drift bites on generator rollouts).

## Tests
`../.venv/bin/python -m tests.<name>` — `test_helmholtz` (analytic vortex/source),
`test_l137` (band pressures), `test_derive` (pressure/altitude/shear), `test_stage2_metrics`.

## BLE-VAE generator scored (first real generator)
`generators/ble_vae.py` reimplements the offlineskies22 decoder in numpy (flax only for
param load; the real flax Decoder breaks on py3.14 dataclass rules). Validated
divergence-free (`test_ble_vae`). Run: `../.venv/bin/python -m windeval.benchmark_ble`.
Result (mean of 6 samples vs ERA5 peer):
- **vertical coherence 0.79 vs 0.81, temporal persistence 0.94 vs 0.92** — the VAE learned
  coherent 4D structure, nearly matching ERA5 on the control-relevant axes.
- **Helmholtz 0.78 vs 0.45** — divergence-free-by-construction design detected.
- **Fair spectral comparison** (`resample.py`, common 16×16 @50km) revealed: spectrum
  **SLOPE is unreliable at SF-box scale** — ERA5's own slope flips −3.92→+2.31 under
  regridding (too few Fourier modes in a ~700 km box). On the fair grid BLE kurtosis 3.80 <
  ERA5 4.87 → BLE is mildly **over-smooth**; its native 14.6 was a resize artifact.

## Harness lessons learned
- Coherence/dimensionless metrics compare cleanly across grids; spectral metrics need a
  common-grid resample first (`resample.py`).
- Spectral **slope** needs a large domain — the SF ~1000 km box is too small (validated by
  ERA5's own slope being unstable under regridding). Kurtosis is grid-artifact-sensitive →
  compare only on a fair common grid.

## Status / next
- ✅ Stage 1 + Stage 2 + BLE-VAE generator + fair common-grid spectral comparison — validated.
- ⏳ Larger-domain pull if spectral slope is wanted; Axis-2 (seam/revisit/budget) once a
  lazy/tiled generator exists; generator over-smoothing + advection checks; full year/splits.
