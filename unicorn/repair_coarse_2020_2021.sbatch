#!/bin/bash
#SBATCH --job-name=repair-coarse-2021
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
python src/eval/windeval/hpx/ingest_arco.py --out ~/data/era5_hpx_coarse_2020_2021 --years 2020 2021 --workers 8 --no-fine
rm -f ~/data/era5_hpx_coarse_2020_2021/valid.npy
