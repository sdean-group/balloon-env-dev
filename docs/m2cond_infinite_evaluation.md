# Conditional M2 InfiniteDiffusion Evaluation

## Checkpoint contract

`idiff_m2cond_latest.pt` is a step-100,000 conditional M2 EDM checkpoint:

- 4 consecutive hourly frames per block;
- 18 vertical levels, with interleaved `u,v` channels (`C=36`);
- 64x64 training crops at 0.25 degrees;
- latitude/longitude coordinate channels;
- annual, semiannual, and diurnal cyclic time features;
- a 41,928,676-parameter factorized space-time U-Net;
- deterministic 18-step Heun sampling by default.

## InfiniteDiffusion mapping

The wrapper uses the packed construction from the author's annotated implementation:

1. create coordinate-keyed Gaussian noise shared by overlapping windows;
2. run the complete EDM trajectory inside each space-time window;
3. return `concat(W * prediction, W)` from each window;
4. let `infinite-tensor` sum overlapping packed windows;
5. divide the value channels by the accumulated weight channel on query.

The first baseline is an outer `T=1` integration: each window performs the complete EDM
trajectory before final overlap blending. The `T=2` experiment splits that same trajectory
after step 9, blends the intermediate noisy states, and continues steps 9--17 from the
blended field. It never restarts the sampler.

For the 64x64x4 query with 50% overlap, `T=1` evaluates 27 complete window trajectories
(945 network forwards). `T=2` evaluates 125 initial-phase windows and 27 continuation
windows (2,709 network forwards). The extra dependency pyramid is the intended cost of
intermediate overlap propagation.

## Local results

Hardware: M1 Pro, 16 GB RAM. The available PyTorch runtime was CPU-only.

### Full base-model block

One 64x64x4 block with the trained 18-step Heun sampler:

- runtime: 61.3 seconds;
- finite output: yes;
- mean speed: 9.63 m/s;
- 95th percentile speed: 17.39 m/s;
- maximum speed: 22.99 m/s;
- adjacent-frame mean vector change: 2.72 m/s;
- adjacent-frame correlations: 0.82, 0.78, 0.95.

The generated normalized anomaly standard deviations were 0.25 for `u` and 0.42 for
`v`, versus a target near 1.0 from the ERA5 normalization. The deterministic sampler is
therefore strongly under-dispersed in this single-block check.

### Paper-style overlapping smoke run

The local compromise used 4 EDM steps, 32x32 windows, 50% spatial overlap, and 50%
temporal overlap:

- runtime: 106.1 seconds;
- unique model-window calls: 27;
- cached repeat: zero additional calls and exact equality;
- x seam jump / ordinary x jump: 1.03;
- y seam jump / ordinary y jump: 0.99;
- time seam jump / ordinary time jump: 0.98.

The seam ratios show no abnormal discontinuity at blend boundaries. The wind values from
this run are not a quality result because four steps and 32x32 windows differ from the
trained 18-step, 64x64 configuration.

## Local commands

Create an environment using the system PyTorch installation and add `infinite-tensor`:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'infinite-tensor>=0.3'
```

Run the short mechanics check:

```bash
.venv/bin/python \
  src/eval/windeval/generators/infinite_diffusion/evaluate_spacetime_infinite.py \
  --checkpoint ../idiff_m2cond_latest.pt \
  --output-dir outputs/m2cond_infinite_smoke \
  --device cpu --num-steps 1 --window 16 --stride 8 \
  --time-stride 2 --query-size 8 --query-frames 2
```

Run the wrapper tests:

```bash
.venv/bin/python -m pytest tests/test_windeval/test_spacetime_infinite.py -q
```

## Unicorn full run

Use the GPU partition for the real 64x64, 18-step evaluation. Prepare a conda environment
in an interactive allocation, not on the login node:

```bash
salloc --partition=gpu --mem=16g --gres=gpu:1 --cpus-per-task=4
/share/apps/software/anaconda3/bin/conda create -p ~/envs/idiff-eval python=3.12 -y
conda activate ~/envs/idiff-eval
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow 'infinite-tensor>=0.3'
exit
```

From the repository directory, submit:

```bash
mkdir -p outputs
PYTHON="$HOME/envs/idiff-eval/bin/python" \
CHECKPOINT="$PWD/idiff_m2cond_latest.pt" \
sbatch --requeue \
  src/eval/windeval/generators/infinite_diffusion/configs/evaluate_m2cond.sbatch
```

Run the controlled `T=2` comparison with the same checkpoint, seed, and query:

```bash
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
CHECKPOINT="$PWD/idiff_m2cond_latest.pt" \
OUTPUT_DIR="$PWD/outputs/m2cond_infinite_full_dean_t2_split9" \
OUTER_DEPTH=2 SPLIT_STEP=9 \
sbatch --nodes=1 --nodelist='dean-compute-[01-02]' \
  src/eval/windeval/generators/infinite_diffusion/configs/evaluate_m2cond.sbatch
```

Monitor with `squeue --me`. The job writes `metrics.json`, `wind.npz`, and `wind.png`
under `outputs/m2cond_infinite_full`.

## Interpretation gate

Do not judge InfiniteDiffusion field quality from the local four-step output. The next
decision should use the full Unicorn output and separate:

1. base-model calibration, especially normalized anomaly variance and wind-speed tails;
2. overlap quality, measured by spatial and temporal seam ratios;
3. generation mechanics, measured by exact repeat/query consistency and model calls;
4. runtime, measured cold and warm with the 64x64/18-step configuration.

## Full Dean-node T=1 vs T=2 result

Both runs used checkpoint step 100,000, seed 7, a 64x64x4 query, 50% overlap,
and the same 18-step deterministic EDM trajectory. The T=2 run split after step 9
at sigma 1.9233.

| Metric | T=1 | T=2 split 9 |
|---|---:|---:|
| Generation time | 19.13 s | 45.28 s |
| Network forwards | 945 | 2,709 |
| X seam ratio | 1.116 | 1.133 |
| Y seam ratio | 1.198 | 1.248 |
| Time seam ratio | 1.095 | 1.088 |
| Mean temporal vector change | 2.84 m/s | 2.48 m/s |
| Mean adjacent-frame correlation | 0.883 | 0.906 |

The two fields remain highly similar (paired correlation 0.9985; component RMSE
0.334 m/s). T=2 improved adjacent-frame correlation in all 54 frame-transition/level
comparisons and improved the temporal seam ratio in 12 of 18 levels. It improved the
X seam ratio in 27 of 72 frame/level comparisons and the Y seam ratio in only 17 of 72.
High-frequency spectral power decreased by 2.7%, while grid-unit divergence and vorticity
standard deviations both decreased by about 9%. At this split, T=2 therefore acts mainly
as a temporal and spatial smoother; it does not improve spatial overlap quality.

This is one seed, location, time interval, and seam per axis. Split location and multiple
seeds/queries must be evaluated before drawing a general conclusion.

## Exact ERA5 comparison for the existing January 15 run

The public ARCO-ERA5 model-level archive supplied the exact four reference hours,
location, 0.25-degree grid, and levels 49--66 for the existing T=1/T=2 outputs. This is
a supplementary sanity check, not the held-out benchmark: January 15 is at the start of
the model's training-date range, and one stochastic sample is not expected to reproduce
one reanalysis realization pointwise.

| Metric | ERA5 | T=1 | T=2 split 9 |
|---|---:|---:|---:|
| Speed mean | 12.68 m/s | 9.91 m/s | 9.87 m/s |
| Speed p95 | 24.83 m/s | 18.63 m/s | 18.29 m/s |
| Temporal vector change | 1.30 m/s | 2.84 m/s | 2.48 m/s |
| Adjacent-frame correlation | 0.990 | 0.883 | 0.906 |
| High-frequency power fraction | 0.0235 | 0.0111 | 0.0108 |
| Grid divergence std | 0.713 | 0.379 | 0.344 |
| Grid vorticity std | 0.558 | 0.401 | 0.364 |
| Component RMSE vs ERA5 | - | 5.263 m/s | 5.204 m/s |
| Component MAE vs ERA5 | - | 3.961 m/s | 3.902 m/s |
| Vector correlation vs ERA5 | - | 0.815 | 0.822 |
| Per-level marginal W1 | - | 2.648 m/s | 2.659 m/s |

T=2 is slightly closer on pointwise RMSE, MAE, correlation, and temporal coherence, but
slightly worse on marginal W1. Both runs are too calm and contain substantially less
high-frequency, divergent, and vortical variation than this ERA5 event. T=2 suppresses
those components further, consistent with the earlier conclusion that it acts as a
smoother. The condition-matched held-out benchmark across four seasons and paired seeds
is required before deciding whether the small pointwise improvement generalizes.

## Arbitrary Outer Depth

`InfiniteSpaceTimeDiffusion` accepts any positive outer depth no larger than the number
of EDM steps. A depth `T` has `T-1` strictly increasing split steps. Every segment is one
lazy, cached `InfiniteTensor` phase whose input is the overlap-normalized previous phase.
This preserves the existing deterministic random-access construction while allowing the
dependency pyramid to grow to the requested depth.

For an 18-step sampler:

```text
T=1: no splits
T=2: --split-steps 9
T=3: --split-steps 6 12
T=4: --split-steps 4 9 14
```

If split steps are omitted, they are distributed as evenly as integer step indices allow.
The T=2 `--split-step 9` option remains supported for compatibility.

For the standard `4x64x64` query, expected model-forward counts are:

```text
T=1:    945
T=2:  2,709
T=3:  5,913
T=4: 10,701
```

Because the held-out T=2 benchmark worsened every reported frame-realism metric, T=3 and
T=4 should begin with an eight-condition pilot: day 8, hours 00/12, four seasons, one seed.
Only a positive pilot result justifies generating the complete 112-block condition set.
