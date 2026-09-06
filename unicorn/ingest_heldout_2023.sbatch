#!/bin/bash
#SBATCH --job-name=ingest-heldout
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
# Benchmark reference on HEALPix: days 8-14 of Jan/Apr/Jul/Oct 2023 (672 hours), fine + coarse,
# never used for training. ~150 GB transfer, ~38 GB fine + 1.2 GB coarse on scratch.
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
python src/eval/windeval/hpx/ingest_arco.py --out /scratch/sps252/era5_hpx_heldout_2023 --years 2023 --heldout --months 1 4 7 10 --workers 8
