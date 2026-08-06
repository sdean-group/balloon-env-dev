#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/balloon-env-dev-code}"
CONFIG="$REPO/src/eval/windeval/generators/rlhab_synthwinds/configs"
NODES="${NODES:-dean-compute-[01-02]}"

DATA_JOB=$(sbatch --parsable --nodelist="$NODES" "$CONFIG/download_spatial_reference.sbatch")
JOB=$(sbatch --parsable --nodelist="$NODES" --dependency="afterok:$DATA_JOB" \
  "$CONFIG/score.sbatch")

echo "ERA5 spatial reference: $DATA_JOB"
echo "RL-HAB evaluation:       $JOB (afterok:$DATA_JOB)"
echo "This job reuses the existing RL-HAB artifact and scores no other method."
