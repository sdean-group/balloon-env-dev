"""Download the radiosonde observations used to construct RL-HAB SynthWinds.

RL-HAB's public notebook aggregates University of Wyoming soundings, fills a regular
horizontal grid from the nearest station, and applies a broad Gaussian smoother.  This
module only performs the first, network-bound step. Each sounding response is cached
verbatim, making retries and later regeneration cheap and auditable.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import time
from datetime import datetime
from pathlib import Path

import requests

MONTHS = (1, 4, 7, 10)
TARGET_DAYS = tuple(range(8, 15))
TARGET_HOURS = (0, 12)


def read_stations(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def download_soundings(
    output: Path,
    *,
    stations_path: Path,
    year: int = 2023,
    months: tuple[int, ...] = MONTHS,
    workers: int = 6,
    retries: int = 8,
    timeout_s: int = 120,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stations = read_stations(stations_path)
    session = requests.Session()
    session.headers["User-Agent"] = "balloon-env-dev RL-HAB benchmark"
    endpoint = "https://weather.uwyo.edu/wsgi/sounding"
    requests_to_make = [
        (datetime(year, month, day, hour), station["wmo"])
        for month in months
        for day in TARGET_DAYS
        for hour in TARGET_HOURS
        for station in stations
    ]

    def fetch(item) -> tuple[str, str]:
        timestamp, wmo = item
        target = output / f"{timestamp:%Y%m%dT%H}-{wmo}.html"
        if target.exists() and target.stat().st_size > 1000:
            return "cached", target.name
        params = {
            "datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "id": wmo,
            "type": "TEXT:LIST",
            # Compact mandatory/significant-level archive. The denser BUFR source is
            # unnecessary because the benchmark interpolates to 18 pressure levels.
            "src": "FM35",
        }
        local_session = requests.Session()
        local_session.headers["User-Agent"] = session.headers["User-Agent"]
        for attempt in range(1, retries + 1):
            try:
                response = local_session.get(endpoint, params=params, timeout=timeout_s)
                response.raise_for_status()
                text = response.text
                if "PRES" not in text or "SPED" not in text:
                    raise RuntimeError("archive returned no valid data")
                target.write_text(text)
                return "downloaded", target.name
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == retries:
                    return "unavailable", f"{target.name}: {exc}"
                delay = min(60, 2 ** attempt)
                time.sleep(delay)

    total = len(requests_to_make)
    unavailable = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for done, (status, detail) in enumerate(executor.map(fetch, requests_to_make), 1):
            if status == "unavailable":
                unavailable.append(detail)
            print(f"[download {done}/{total}] {status} {detail}", flush=True)
    print(
        f"[download] complete: {total - len(unavailable)}/{total} soundings available",
        flush=True,
    )


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download RL-HAB radiosonde inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stations", type=Path, default=here / "stations.csv")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    download_soundings(
        args.output,
        stations_path=args.stations,
        year=args.year,
        months=tuple(args.months),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
