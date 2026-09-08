# Unicorn cluster scripts

Cluster reference (partitions, nodes, storage, gotchas): the hub's
`research/docs/unicorn-reference.md`. Environment: `~/envs/cbottle` on Unicorn. Submit from
`~/balloon-env-dev/unicorn` so `logs/%x-%j.out` lands in `unicorn/logs/`. Slurm binaries are
in `/usr/local/slurm/current/bin` (not on the non-interactive PATH).

## Environment and cBottle

- `install_cbottle.sh` — builds `~/envs/cbottle` (torch cu128, earth2grid, cBottle, our
  stack) as a CPU job: `sbatch --partition=dean --cpus-per-task=4 --mem=16G --time=03:00:00
  --output=$HOME/envs/cbottle-install.log unicorn/install_cbottle.sh`
- `cbottle_coarse_smoke.sbatch`, `cbottle_coarse_seasons.sbatch`, `compare_cbottle_u50.py` —
  pretrained cBottle-3d samples on a group GPU and the 50 hPa comparison against ERA5. Weights
  in `~/cbottle-weights`.
- `test_hpx_stack.py`, `bench_stage1.py` — layout/net checks and the Stage 1 throughput table.

## Data (ARCO ERA5 -> HEALPix stores)

- `ingest_fine_2022_2023.sbatch` — fine (nside 256) + coarse (nside 32) 2022-2023 store on
  `/scratch/sps252/era5_hpx` (node-local to dean-compute-02).
- `ingest_coarse_2020_2021.sbatch` — coarse-only 2020-2021 store in `~/data`.
- `ingest_heldout_2023.sbatch` — days 8-14 of Jan/Apr/Jul/Oct 2023 (never trained on) to
  `/scratch/sps252/era5_hpx_heldout_2023`.
- `repair_coarse*.sbatch` — recompute coarse rows from fine after the chunk-write race.

## Stage 1 (whole-sphere coarse generator)

- `train_stage1.sbatch` — `CONFIG=<yaml> sbatch --job-name=<name> train_stage1.sbatch`.
  Forwards Slurm's USR1 to the trainer, which checkpoints and exits 75, then requeues; note
  the double `wait` (the first returns 128+signal when the trap fires).
- `configs/stage1_hpx32.yaml` — the 300k-step run (EDM log-normal sigma; its model ignores
  the season, see LOG 2026-09-07). `configs/stage1_hpx32_ft.yaml` — continuation with
  log-uniform sigma to 1000, the model to use.
- `eval_stage1_curve.sbatch` — the gate (`hpx/eval_coarse.py`) over every snapshot of a run:
  `RUN=<dir> CKPTS="step_60000.pt" CHURNS="0" STEPS=18 SUFFIX=_x sbatch --job-name=<name>
  eval_stage1_curve.sbatch [--sigma-max 200]`; writes `<run>/eval_<ckpt>[_churnC][_sN]`.
- `summarize_stage1_evals.py <run dir>` — one markdown table over the run's `eval_*` dirs.
- `jet_by_month.py --ckpt ... [--churn C --steps N --sigma-max S]` — per-month zonal-mean u
  profiles vs held-out ERA5 (the diagnostic that exposed the seasonal blindness).
