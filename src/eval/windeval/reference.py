"""Held-out ERA5 reference for benchmark v2.

The trained model saw days ~15–28 of Jan/Apr/Jul/Oct 2023 (`era5_train.zarr`). The
benchmark reference must not contain training dates, so it is the days-8–14 slice of
`era5_temporal.zarr` (hourly, NE Pacific, model levels 49–66) — same climate, zero date
overlap. `split` gives two disjoint halves (days 8–10 vs 11–14) so the report can show a
same-distribution sampling-noise floor for every metric.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import artifact

DATA = Path(__file__).resolve().parent / "data"
HELDOUT_DAYS = (8, 14)          # inclusive; training used ~15–28
SPLIT_DAY = 11                  # floor halves: days 8–10 vs 11–14


def build_heldout(temporal_zarr: Path = DATA / "era5_temporal.zarr",
                  out: Path = DATA / "era5_heldout.zarr") -> Path:
    """Slice the held-out days out of the contiguous download and cache as an artifact."""
    if out.exists():
        return out
    ds = xr.open_zarr(temporal_zarr, consolidated=False, zarr_format=2)
    day = ds["time"].dt.day
    sel = ds.sel(time=(day >= HELDOUT_DAYS[0]) & (day <= HELDOUT_DAYS[1]))
    sel = sel.compute()
    attrs = dict(ds.attrs)
    attrs.update({
        "generator": {"name": "era5_heldout", "version": "v2"},
        "capabilities": {"extent": "bounded", "temporally_evolving": True},
        "conditioning": {"source": "era5_temporal.zarr",
                         "days": list(HELDOUT_DAYS),
                         "note": "non-overlapping with era5_train.zarr training dates"},
    })
    return artifact.write(sel, attrs, out)


def split(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Two disjoint halves of the held-out set, for the sampling-noise floor row."""
    day = ds["time"].dt.day
    return (ds.sel(time=day < SPLIT_DAY), ds.sel(time=day >= SPLIT_DAY))
