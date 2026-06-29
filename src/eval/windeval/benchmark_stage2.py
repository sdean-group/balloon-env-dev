"""Stage-2 calibration benchmark — spatial + VERTICAL + TEMPORAL objective scores.

Adds the two axes Stage 1 couldn't reach (one timestep, no altitude):
  - vertical coherence  (the control-relevant axis; needs the band + sp)
  - temporal persistence + tendency + drift (needs the time series)

Key point this surfaces: the per-timestep anchors (phase-shuffle, white-noise) are
INCOHERENT across level and time, so vertical + temporal metrics catch them cleanly —
even though phase-shuffle *fooled* the spatial spectrum in Stage 1. The full suite is
robust where any single axis is not.

Run:  python -m windeval.benchmark_stage2
"""
from __future__ import annotations

from pathlib import Path

from . import artifact, anchors
from .ingest_era5 import ingest
from .metrics import field_scores, vertical_scores, temporal_scores, drift

DATA = Path(__file__).resolve().parent / "data"


def run() -> str:
    real_z = ingest(DATA / "era5_sf_uvtq.grib", DATA / "era5_real_stage2.zarr",
                    lnsp_path=DATA / "era5_sf_lnsp.grib")
    ps_z = anchors.phase_shuffle(real_z, DATA / "anchor_ps_stage2.zarr", seed=0)
    no_z = anchors.white_noise(real_z, DATA / "anchor_noise_stage2.zarr", seed=0)

    fields = {}
    for name, z in [("era5_real (peer)", real_z), ("phase_shuffle", ps_z),
                    ("white_noise", no_z)]:
        ds = artifact.read(z)
        s = {**field_scores(ds), **vertical_scores(ds), **temporal_scores(ds)}
        s.update(drift(ds, field_scores))
        fields[name] = s
    cols = list(fields.keys())

    def table(title, rows, fmt="{:.3g}"):
        out = ["", f"**{title}**", "",
               "| Metric | " + " | ".join(cols) + " |",
               "|" + "---|" * (len(cols) + 1)]
        for label, key in rows:
            out.append(f"| {label} | " +
                       " | ".join(fmt.format(fields[c][key]) for c in cols) + " |")
        return out

    lines = [
        "# Stage-2 Calibration Benchmark — spatial + vertical + temporal", "",
        "Objective scores ∈ [0,1] (higher = more wind-like); ERA5 is a peer row. "
        "24 hourly SF timesteps, model-level band 49-66.", "",
        "**Headline scores**",
        "", "| Score | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1),
    ]
    for label, key in [("spatial (spectrum+intermittency)", "COMPOSITE"),
                       ("vertical coherence", "score: vertical"),
                       ("temporal persistence", "score: temporal")]:
        lines.append(f"| {label} | " +
                     " | ".join(f"**{fields[c][key]:.3f}**" for c in cols) + " |")

    lines += table("Vertical (control axis)",
                   [("vertical coherence", "vertical coherence"),
                    ("shear mean (m/s/km)", "shear mean (m/s/km)")])
    lines += table("Temporal / dynamics",
                   [("temporal persistence", "temporal persistence"),
                    ("tendency (m/s/h)", "tendency (m/s/h)"),
                    ("drift slope/step (≈0 ok)", "drift slope/step")])
    lines += table("Spatial detail",
                   [("spectrum slope", "spectrum slope"),
                    ("increment kurtosis", "increment kurtosis"),
                    ("Helmholtz (diagnostic)", "Helmholtz rot. frac")])

    lines += ["", "## Verdict (automated checks)"]
    for ok, msg in _verdict(fields):
        lines.append(f"- {'✅' if ok else '❌'} {msg}")
    report = "\n".join(lines)

    out = DATA.parent / "benchmark_stage2_report.md"
    out.write_text(report)
    print(report)
    print(f"\nreport: {out}")
    return report


def _verdict(f):
    real, ps, no = f["era5_real (peer)"], f["phase_shuffle"], f["white_noise"]
    c = []
    c.append((real["score: vertical"] > 0.8 and ps["score: vertical"] < 0.3
              and no["score: vertical"] < 0.3,
              f"VERTICAL coherence CATCHES both anchors: real {real['score: vertical']:.2f} "
              f"vs shuffle {ps['score: vertical']:.2f} / noise {no['score: vertical']:.2f} "
              "(per-level randomization decorrelates adjacent levels)"))
    c.append((real["score: temporal"] > 0.8 and ps["score: temporal"] < 0.3
              and no["score: temporal"] < 0.3,
              f"TEMPORAL persistence CATCHES both anchors: real {real['score: temporal']:.2f} "
              f"vs shuffle {ps['score: temporal']:.2f} / noise {no['score: temporal']:.2f}"))
    c.append((real["tendency (m/s/h)"] < ps["tendency (m/s/h)"]
              and real["tendency (m/s/h)"] < no["tendency (m/s/h)"],
              f"tendency: real evolves slowly ({real['tendency (m/s/h)']:.2f} m/s/h) vs "
              f"incoherent anchors ({ps['tendency (m/s/h)']:.1f} / {no['tendency (m/s/h)']:.1f})"))
    c.append((abs(real["drift slope/step"]) < 0.01,
              f"real ERA5 does NOT drift (slope {real['drift slope/step']:+.4f}/step ≈ 0) — "
              "machinery validated; drift bites on generator rollouts"))
    c.append((ps["COMPOSITE"] > 0.4,
              f"spatial spectrum STILL fooled by phase-shuffle (composite {ps['COMPOSITE']:.2f}) "
              "— but vertical+temporal catch it: the full suite is robust"))
    return c


if __name__ == "__main__":
    run()
