"""Stage-2 ERA5 model-level extraction: u,v,T,q on the band + lnsp for surface pressure.

Two MARS retrieves (different levelists):
  1. u,v,T,q  on model levels 49-66        (param 131/132/130/133)
  2. lnsp     on model level 1 (2D field)  (param 152)  -> sp = exp(lnsp)

Pulls a time range (default 1 day, hourly) over the SF box, for the temporal +
vertical metrics. NOT the full year — deliberately small to keep iteration fast.

Run:  ../.venv/bin/python scripts/extract_era5.py [YYYY-MM-DD] [n_hours]
"""
import sys
from pathlib import Path

import cdsapi

LEVELS = "/".join(str(l) for l in range(49, 67))   # 49..66
AREA = "42/-128/33/-118"                            # N/W/S/E box around SF
GRID = "0.25/0.25"
DATA = Path(__file__).resolve().parent.parent / "data"


def extract(date="2023-01-01", n_hours=24):
    DATA.mkdir(exist_ok=True)
    times = "/".join(f"{h:02d}:00:00" for h in range(n_hours))
    c = cdsapi.Client()

    base = dict(cls="ea", stream="oper", expver="1", type="an", levtype="ml",
                date=date, time=times, grid=GRID, area=AREA, format="grib")
    # cdsapi uses "class" (reserved word) -> pass via dict key
    base["class"] = base.pop("cls")

    uvtq = DATA / "era5_sf_uvtq.grib"
    lnsp = DATA / "era5_sf_lnsp.grib"

    print(f"[1/2] u,v,T,q  levels 49-66  {date}  {n_hours}h ...")
    c.retrieve("reanalysis-era5-complete",
               {**base, "levelist": LEVELS, "param": "130/131/132/133"}, str(uvtq))

    print(f"[2/2] lnsp     level 1       {date}  {n_hours}h ...")
    c.retrieve("reanalysis-era5-complete",
               {**base, "levelist": "1", "param": "152"}, str(lnsp))

    print(f"done: {uvtq.name}, {lnsp.name}")
    return uvtq, lnsp


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    n_hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    extract(date, n_hours)
