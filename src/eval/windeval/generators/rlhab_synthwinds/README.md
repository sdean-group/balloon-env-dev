# RL-HAB SynthWinds benchmark

This folder adapts the public RL-HAB `SynthWinds` baseline to the wind-evaluation
artifact contract. RL-HAB is an RL simulation environment; the comparable wind source
is its radiosonde-based SynthWinds field, not its DQN policy.

## Method reproduced

1. Download University of Wyoming radiosonde soundings for the held-out 2023 dates.
2. Interpolate each sounding vertically to the benchmark pressure levels.
3. Assign every horizontal grid cell the nearest station profile.
4. Apply the RL-HAB Gaussian spatial smoother. The notebook uses sigma 3 cells on a
   1-degree grid; this adapter uses the equivalent sigma 12 cells on a 0.25-degree grid.
5. Score the resulting fields with the existing ERA5 benchmark-v2 metrics.

The deliberate adaptations are grid resolution and vertical coordinates. The notebook
builds a 1-degree horizontal grid and a 250 m altitude grid; the benchmark uses a
0.25-degree grid and ERA5 model levels. Keeping the smoothing scale at 3 degrees and
interpolating directly in pressure preserves the intended construction while comparing
the same physical levels.

## Interpretation

SynthWinds uses radiosonde observations from the timestamps being evaluated. It is a
useful balloon-simulator baseline, but it is not a free-running or seeded generator. A
strong result therefore does not imply that it solves conditional wind generation.

## Unicorn

One-time dependency setup on a login node:

```bash
PY="$HOME/envs/idiff-eval-titan/bin/python"
"$PY" -m pip install requests beautifulsoup4 scipy
```

Then submit the resumable preparation and dependent scoring jobs:

```bash
cd ~/balloon-env-dev-code
bash src/eval/windeval/generators/rlhab_synthwinds/configs/submit.sh
```

The preparation job makes 1,288 small sounding requests (23 stations, 56 held-out
timestamps) with six workers. Expect roughly 45-90 minutes under normal archive load;
reruns reuse every completed response. Construction and scoring usually take only a few
minutes after the download.

Monitor both jobs with:

```bash
watch -n 10 'squeue --me'
```

Results are written to:

```text
/share/dean/$USER/balloon-research/outputs/rlhab_synthwinds/
```

Sources: RL-HAB and RadioWinds are U.S. Naval Research Laboratory projects. Their public
repositories and the RL-HAB paper describe the radiosonde aggregation and smoothing
procedure.
