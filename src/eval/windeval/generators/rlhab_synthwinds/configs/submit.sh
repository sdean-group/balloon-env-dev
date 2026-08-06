#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/balloon-env-dev-code}"
CONFIG="$REPO/src/eval/windeval/generators/rlhab_synthwinds/configs"
NODES="${NODES:-dean-compute-[01-02]}"

PREP_JOB=$(sbatch --parsable --nodelist="$NODES" "$CONFIG/prepare.sbatch")
SCORE_JOB=$(sbatch --parsable --nodelist="$NODES" --dependency="afterok:$PREP_JOB" \
  "$CONFIG/score.sbatch")

echo "RL-HAB download + construction: $PREP_JOB"
echo "RL-HAB scoring:                $SCORE_JOB (afterok:$PREP_JOB)"
