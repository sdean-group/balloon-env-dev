#!/bin/bash
#SBATCH --job-name=ingest-fine
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
# ARCO-ERA5 -> HEALPix fine (nside 256, float16) + coarse (nside 32) stores, 2022-2023 hourly,
# days 8-14 excluded, onto the local NVMe of the training node. Resumable: rerun to continue.
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
python src/eval/windeval/hpx/ingest_arco.py --out /scratch/sps252/era5_hpx --years 2022 2023 --workers 8
