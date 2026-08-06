#!/bin/bash
set -euo pipefail

REPO="${REPO:-$HOME/balloon-env-dev-code}"
CONFIG="$REPO/src/eval/windeval/generators/rlhab_synthwinds/configs"
NODES="${NODES:-dean-compute-[01-02]}"

JOB=$(sbatch --parsable --nodelist="$NODES" "$CONFIG/score.sbatch")

echo "RL-HAB evaluation: $JOB"
echo "This job reuses the existing RL-HAB artifact and scores no other method."
