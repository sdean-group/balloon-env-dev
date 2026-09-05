# Live InfiniteDiffusion Drifter

This runs one persistent `InfiniteSpaceTimeDiffusion` instance on a Unicorn GPU.
The server lazily requests new space-time chunks as the passive drifter approaches
uncached coordinates. The drifter pauses rather than substituting synthetic wind
when generation falls behind.

## Submit on Unicorn

```bash
cd ~/balloon-env-dev-code

PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt" \
OUTER_DEPTH=1 \
sbatch \
  src/eval/windeval/generators/infinite_diffusion/live_viewer/run_live.sbatch
```

`T=1` is the practical live default. To run the paper-style intermediate
overlap at `T=2`:

```bash
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt" \
OUTER_DEPTH=2 \
SPLIT_STEP=9 \
sbatch \
  src/eval/windeval/generators/infinite_diffusion/live_viewer/run_live.sbatch
```

## Connect from the Mac

After the job starts:

```bash
squeue --me
tail -f idiff-live-JOB_ID.out
```

The output prints a command like:

```bash
ssh -N -L 27654:dean-compute-02:27654 unicorn
```

Run that printed command in a second Mac terminal, leave it open, and visit the
printed `http://localhost:PORT` address.

## What is live

- The checkpoint is loaded once on the allocated GPU.
- One InfiniteDiffusion field and tile cache remain alive for the job.
- Missing 4-frame by 64 by 64 chunks are generated from the real sampler.
- The server integrates a non-actuated, fixed-level drifter through completed chunks.
- The browser polls the server-owned trajectory and generated wind volume.
- Model-window counts, generation bounds, cache size, and generation latency update
  while the sampler runs.

The particles are a display of the same streamed wind values. The yellow agent and
its trail come from the server-side physical integration.

## Local replay

The interface can be tested without a model or GPU:

```bash
python3 \
  src/eval/windeval/generators/infinite_diffusion/live_viewer/server.py \
  --demo-npz outputs/m2cond_infinite_full_dean_t2_split9/wind.npz \
  --host 127.0.0.1 \
  --port 8765
```

Then open `http://localhost:8765`.
