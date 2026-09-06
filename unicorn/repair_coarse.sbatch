#!/bin/bash
#SBATCH --job-name=repair-coarse
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# Fix for the 2026-09-05 chunk write race: rebuild every incomplete coarse chunk of the scratch
# store by pooling its (intact, one-chunk-per-hour) fine rows, and re-fetch incomplete chunks
# of the coarse-only 2020-2021 store. Both are idempotent; rerun to continue.
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
python src/eval/windeval/hpx/ingest_arco.py --out /scratch/sps252/era5_hpx --years 2022 2023 --workers 8 --repair-coarse-from-fine
rm -f /scratch/sps252/era5_hpx/valid.npy
