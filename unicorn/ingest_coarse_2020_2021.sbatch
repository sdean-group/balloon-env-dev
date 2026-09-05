#!/bin/bash
#SBATCH --job-name=ingest-coarse
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
# Coarse-only (nside 32, float32) store for 2020-2021 into home (~24 GB); with the coarse half
# of the scratch store this gives Stage 1 four years. Resumable.
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
python src/eval/windeval/hpx/ingest_arco.py --out ~/data/era5_hpx_coarse_2020_2021 --years 2020 2021 --workers 8 --no-fine
