# Gameplan, September 2026 — fidelity-to-transfer study on adapted cBottle

Written 2026-09-04 from Shaurya's plan after the Sarah/Rohan meeting, with two corrections
(marked **⚠**). Owners: Shaurya (generator, benchmark, transfer study), Rohan (balloon
environment, task, globe viz). Hub `LOG.md` has the decision record.

## The paper

**Question.** Which measurable realism properties of a wind simulator determine how well a
controller built in it performs on real winds?

**Design.** A *fidelity ladder* of simulators, each scored on the calibrated realism
benchmark (spectra, tails, vertical decoupling, temporal coherence, conditional W1), each
used to build a controller, every controller evaluated on the same held-out ERA5 winds.
The result is a map from realism metrics to transfer gap, per task.

| rung | simulator | what it has | what it lacks |
|---|---|---|---|
| 0 | white noise (moment-matched) | marginals | everything else |
| 1 | simplex / fBm (BLE's family) | spectra, smooth vertical | structure, conditioning |
| 2 | Helmholtz GP | div/vort split | tails, structure |
| 3 | BLE-VAE | coherent 4D structure | 10 levels, SF box, no conditioning |
| 4 | plain conditional diffusion (m2cond) | location/time conditioning | large-scale coherence |
| 5 | cascade: generated coarse + generated fine (adapted cBottle) | global coherence, vertical structure | ? (to be measured) |
| 6 | ERA5 coarse replay + generated fine | real synoptic state | novel weather |
| 7 | **ERA5 replay (sim\*)** | real | bounded to archive |

**⚠ Correction 1: sim\* is ERA5, not the adapted cBottle.** The transfer target must be
real winds (held-out ERA5 days replayed through the same environment). Evaluating transfer
into another generative model is circular. The adapted cBottle is rung 5, the top generative
rung, not the reference.

**⚠ Correction 2: pretrained cBottle cannot be scored on our benchmark as-is.** Its 3D
fields are on 1000/850/700/500/300/200/50/10 hPa; nothing between 200 and 50 hPa, where our
18 model levels (49-66, ~50-134 hPa) live. Vertical metrics need the column. What running the
pretrained model *does* buy: (a) the HEALPix/earth2grid/EDM tooling validated on Kahan,
(b) a horizontal-only sanity comparison of its 50 and 200 hPa u/v against ERA5 pressure-level
data on spectra/marginals/tails, (c) a reference implementation for the retrain. The
balloon-band generator is cBottle's architecture retrained on our channels.

## Controllers: two classes, both needed

- **Model-based (MPC).** The simulator is the planner's internal predictive model:
  sample futures from it (seed-consistent, so paired samples are possible), plan, act in
  sim\*. Fidelity of the internal model → closed-loop performance. No training loop.
- **Model-free (RL).** Policy trained in the simulator, executed in sim\*. BLE used
  QR-DQN, RL-HAB used PPO; start with those.

The two answer different questions and disagree in interesting ways (a model-based planner
may be hurt by fidelity gaps a policy is robust to). Paper 1 can use either; using both is
the stronger design if compute allows. Paper 2 (RL vs learned MPC in the locked sim) follows.

## Task

Pick the task whose success depends on the realism axes being varied. Station-keeping
(BLE's task: stay within radius R of a target by changing altitude) depends on vertical
decoupling and temporal coherence and is comparable to BLE/RL-HAB numbers. Long-range
navigation ("maximise distance" / reach a target) depends on horizontal coherence and jets.
Recommendation: station-keeping primary, navigation secondary; both are one arena each in
the existing `src/env`. Lock the observation, action, and reward definitions before any
training run and keep them fixed across rungs.

## Evaluation protocol (pre-register before results)

- Held-out ERA5: days 8-14 of every month, unchanged; add held-out regions for a
  generalisation split.
- Same environment code for every rung; only the wind provider changes.
- Paired evaluation with common random numbers where the wind provider supports it
  (seed-consistent generators) and independent seeds otherwise; report both.
- Realism metrics per rung come from the existing benchmark v2 plus a global spherical
  spectrum for HEALPix rungs.
- Primary outcome: transfer gap = performance(sim\*) − performance(train sim) and absolute
  performance on sim\*, per rung, per task, with confidence intervals over seeds and days.

## Status log

- **2026-09-05, step 1 done.** Unicorn env built and verified; pretrained cBottle-3d runs
  (2 global HPX64 samples in 17 s on the group RTX 3090). Step 1a at the one shared level
  (50 hPa, 16 timestamps): spectra decent (mean abs log ratio 0.10-0.19), but a systematic
  ~5-10 m/s easterly bias, a much-too-weak polar-night jet, and the QBO phase missing in the
  tropics. Details in the hub LOG. **New design item for Stage 1: a slow-state conditioner
  (QBO / previous-month zonal wind), since cyclic time harmonics cannot carry interannual
  stratospheric modes.**

## Steps in order

1. **Run cBottle** (pretrained) on Kahan: install `earth2grid` + cBottle, run the coarse
   model, regrid one sample to lat/lon, score 50/200 hPa horizontals. Gate: tooling works.
2. **Adapt cBottle to our channels**: data path ARCO 0.25° → HEALPix nside 256 → nested
   pool to nside 32, 18 model levels u/v; retrain the coarse model (whole sphere) and the
   patch SR model with our residual parameterisation, projection and time axis; wire the
   space-time InfiniteDiffusion wrapper for time. Gate: beats m2cond on the benchmark and
   passes the coarse-stage dispersion/jet checks. Fallback: rung 6 (ERA5 coarse replay).
3. **Environment + task** (Rohan): one wind-provider interface serving every rung including
   ERA5 replay; station-keeping arena; deterministic episodes keyed by (provider, seed, day).
4. **Fidelity-ladder runs**: MPC first (no training), then RL.
5. Write.

## First actions this week

- [ ] Kahan: confirm SSH socket, CUDA toolchain, install earth2grid (CUDA kernel) and
      cBottle; probe GCS bandwidth from the compute node.
- [ ] Download cBottle weights (needs Shaurya's OK; sizes below) and run one coarse sample.
- [ ] Start the ARCO streaming pull (2° global block means + fine faces) once the target
      cadence is fixed (6-hourly coarse, hourly fine windows).
- [ ] Rohan: wind-provider interface + ERA5 replay provider first (it is sim\* and the
      cheapest rung).
