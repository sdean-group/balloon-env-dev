# Unicorn cluster scripts

Cluster reference (partitions, nodes, storage, gotchas): the hub's
`research/docs/unicorn-reference.md`. Environment: `~/envs/cbottle` on Unicorn.

- `install_cbottle.sh` — builds `~/envs/cbottle` (torch cu128, earth2grid, cBottle, our
  stack) as a CPU job: `sbatch --partition=dean --cpus-per-task=4 --mem=16G --time=03:00:00
  --output=$HOME/envs/cbottle-install.log unicorn/install_cbottle.sh`
- `cbottle_coarse_smoke.sbatch` — one pretrained cBottle-3d sample on a group GPU to
  validate the HEALPix/EDM tooling. Needs the weights in `~/cbottle-weights`
  (`cBottle-3d/training-state-009856000.checkpoint`, `amip_midmonth_sst.nc`).
