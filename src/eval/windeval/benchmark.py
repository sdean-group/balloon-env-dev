"""Calibration benchmark — OBJECTIVE quality scoreboard.

No distance-to-ERA5. Each field is scored against physically-motivated ideals;
ERA5 is just a strong peer row. The anchors calibrate the *scale* of the score:
real should top the board, phase-shuffle should pass the spectrum but FAIL the
physics (Helmholtz + intermittency), white noise should fail everything.

Run:  python -m windeval.benchmark [grib_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import artifact, anchors
from .ingest_era5 import ingest
from .metrics import field_scores, _wasserstein1d, RAW_NAMES, SCORE_NAMES

DATA = Path(__file__).resolve().parent / "data"


def run(grib_path: str) -> str:
    DATA.mkdir(exist_ok=True)
    real_z = ingest(grib_path, DATA / "era5_real.zarr")
    ps_z = anchors.phase_shuffle(real_z, DATA / "anchor_phase_shuffle.zarr", seed=0)
    no_z = anchors.white_noise(real_z, DATA / "anchor_white_noise.zarr", seed=0)

    fields = {
        "era5_real (peer)": field_scores(artifact.read(real_z)),
        "phase_shuffle": field_scores(artifact.read(ps_z)),
        "white_noise": field_scores(artifact.read(no_z)),
    }
    cols = list(fields.keys())

    lines = [
        "# Calibration Benchmark — Axis-1 OBJECTIVE quality scoreboard", "",
        "Objective scores vs physical ideals (no distance-to-ERA5). ERA5 is a **peer "
        "row**, not the answer. Scores ∈ [0,1], higher = more wind-like.", "",
        "**Empirical finding (this resolution):** the robust reference-free discriminator "
        "of wind-like structure is **intermittency** (non-Gaussian velocity increments) — "
        "real wind is leptokurtic, phase-randomization Gaussianizes toward 3. The spectrum "
        "slope is *necessary but fooled* by phase-shuffle. **Helmholtz/vort-div are weak "
        "here** (rotational character is mostly spectrum-encoded, so phase-shuffle keeps it) "
        "and the FFT decomposition is biased on a bounded box — so they are caveated "
        "diagnostics, excluded from the composite.", "",
        "**Raw physical values**", "",
        "| Metric (ideal) | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    ideals = {
        "spectrum slope": "≈ −3 (steep)",
        "Helmholtz rot. frac": "→ 1 (rotational)",
        "vort/div ratio": "> 1",
        "increment kurtosis": "> 3 (intermittent)",
    }
    for m in RAW_NAMES:
        cells = [f"{fields[c][m]:.3g}" for c in cols]
        lines.append(f"| {m} ({ideals[m]}) | " + " | ".join(cells) + " |")

    lines += ["", "**Objective scores ∈ [0,1]**", "",
              "| Score | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 1)]
    for s in SCORE_NAMES:
        cells = [(f"**{fields[c][s]:.3f}**" if s == "COMPOSITE" else f"{fields[c][s]:.3f}")
                 for c in cols]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")

    # relative diagnostic (climatology-anchored, ERA5 as peer-distribution)
    ref_speed = fields["era5_real (peer)"]["speed"]
    lines += ["", "**Relative diagnostic** (speed dist. vs ERA5 peer-distribution — "
              "weak separator, kept as a sanity check)", "",
              "| Metric | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 1),
              "| speed Wasserstein (m/s) | " +
              " | ".join(f"{_wasserstein1d(ref_speed, fields[c]['speed']):.3g}" for c in cols)
              + " |"]

    lines += ["", "## Verdict (automated checks)"]
    for ok, msg in _verdict(fields):
        lines.append(f"- {'✅' if ok else '❌'} {msg}")
    report = "\n".join(lines)

    out = DATA.parent / "benchmark_report.md"
    out.write_text(report)
    print(report)
    print(f"\nartifacts: {real_z.name}, {ps_z.name}, {no_z.name}\nreport: {out}")
    return report


def _verdict(f):
    real, ps, no = f["era5_real (peer)"], f["phase_shuffle"], f["white_noise"]
    c = []
    c.append((real["COMPOSITE"] > ps["COMPOSITE"] > no["COMPOSITE"],
              f"ranking: real ({real['COMPOSITE']:.3f}) > phase-shuffle "
              f"({ps['COMPOSITE']:.3f}) > noise ({no['COMPOSITE']:.3f})"))
    c.append((ps["score: spectrum"] > 0.5 and abs(real["score: spectrum"] - ps["score: spectrum"]) < 0.15,
              f"spectrum FOOLED by phase-shuffle (real {real['score: spectrum']:.2f} ≈ "
              f"shuffle {ps['score: spectrum']:.2f}) — necessary but insufficient"))
    c.append((real["increment kurtosis"] > 4.5 and ps["increment kurtosis"] < 4
              and no["increment kurtosis"] < 3.5,
              f"intermittency is THE discriminator: real kurtosis {real['increment kurtosis']:.2f} "
              f">> shuffle {ps['increment kurtosis']:.2f} ≈ noise {no['increment kurtosis']:.2f} (~3)"))
    c.append((abs(real["Helmholtz rot. frac"] - ps["Helmholtz rot. frac"]) < 0.1,
              f"FINDING: Helmholtz NOT usable here (real {real['Helmholtz rot. frac']:.2f} ≈ shuffle "
              f"{ps['Helmholtz rot. frac']:.2f}, Δ={abs(real['Helmholtz rot. frac']-ps['Helmholtz rot. frac']):.2f}). "
              "Decomposition math is validated on analytic vortex/source (tests/test_helmholtz) but "
              "real structure fills this small/coarse box — needs a larger domain or a bounded-domain "
              "Poisson solver. Excluded from composite."))
    c.append((no["COMPOSITE"] < 0.05,
              f"white noise fails the composite (score {no['COMPOSITE']:.3f}) — lower bound confirmed"))
    return c


def main():
    grib = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parents[2] / "sf_ml_test.grib")
    if len(sys.argv) <= 1:
        print(f"(no path given, using {grib})")
    run(grib)


if __name__ == "__main__":
    main()
