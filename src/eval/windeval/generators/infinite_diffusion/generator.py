"""InfiniteDiffusion -> WindArtifact adapter (Phase 1c).

Materializes a finite *crop* of the infinite wind field into the frozen artifact format
so the benchmark can score it. The artifact is a photograph of the unbounded field:

- ``field/``    : a chosen rectangular crop  -> Axis-1 (field quality) + the seam metric
- ``querylog/`` : scattered point queries with cost + revisits -> Axis-2 (revisit, budget)
- ``attrs.capabilities`` : declares extent=unbounded, tiled, random_access -> metric selection
- ``attrs.seam_boundaries`` : window stitch lines as LOCAL array indices (rows/cols within
  the crop), the targets the seam-discontinuity metric probes
- ``attrs.hardware`` : makes the budget numbers comparable

The infinite-ness lives in the *generator* (capabilities + querylog evidence), never in the
file: the artifact stays finite and frozen, exactly like every other generator's.
"""
from __future__ import annotations

import os
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

from ... import artifact
from .denoiser import ToyDivFreeDenoiser
from .sampler import InfiniteDiffusion

# Geographic anchor (shared with ble_vae for cross-comparability).
SF_LAT, SF_LON = 37.77, 237.58
DEG_M = 111_320.0


def _hardware(device) -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        gpu = "mps"
    ram_gb = None
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return {"device": str(device), "gpu": gpu, "ram_gb": ram_gb,
            "platform": platform.platform(), "torch": torch.__version__}


class InfiniteDiffusionGenerator:
    """Wraps an :class:`InfiniteDiffusion` sampler and emits WindArtifacts.

    Args:
        denoiser: a WindowDenoiser; defaults to the analytic toy with ``n_levels`` levels.
        n_levels: vertical levels (channels = 2*n_levels). Ignored if ``denoiser`` given.
        levels: level coordinate values (hPa); defaults to ``linspace(50, 140, n_levels)``
            to coincide with BLE-VAE for Axis-1 comparison.
        pixel_km: physical size of one lattice pixel (≈ ERA5 0.25° ≈ 28 km).
        anchor: (lat, lon) of pixel (0, 0).
        name: generator name written into provenance.
        window, stride, T, seed, **sampler_kw: passed to InfiniteDiffusion.
    """

    def __init__(
        self,
        denoiser=None,
        *,
        n_levels: int = 16,
        levels=None,
        pixel_km: float = 28.0,
        anchor: tuple[float, float] = (SF_LAT, SF_LON),
        name: str = "infinite_diffusion_toy",
        window: int = 64,
        stride: int | None = None,
        T: int = 2,
        seed: int = 0,
        **sampler_kw,
    ) -> None:
        if denoiser is None:
            denoiser = ToyDivFreeDenoiser(n_levels)
        self.n_levels = denoiser.n_channels // 2
        self.levels = (np.asarray(levels, dtype=float) if levels is not None
                       else np.linspace(50.0, 140.0, self.n_levels))
        if len(self.levels) != self.n_levels:
            raise ValueError("len(levels) must equal n_levels")
        self.pixel_km = float(pixel_km)
        self.anchor = anchor
        self.name = name
        self.seed = int(seed)
        self.sampler = InfiniteDiffusion(denoiser, window=window, stride=stride, T=T,
                                         seed=seed, **sampler_kw)

    # ----- geographic mapping (tangent plane about the anchor) -----
    def _coords(self, y0: int, y1: int, x0: int, x1: int):
        lat0, lon0 = self.anchor
        dlat = self.pixel_km * 1000.0 / DEG_M
        dlon = self.pixel_km * 1000.0 / (DEG_M * np.cos(np.deg2rad(lat0)))
        lat = lat0 + np.arange(y0, y1) * dlat
        lon = lon0 + np.arange(x0, x1) * dlon
        return lat, lon

    # ----- the main entry point -----
    def to_artifact(
        self,
        out_path: str | Path,
        *,
        height: int = 256,
        width: int = 256,
        offset: tuple[int, int] = (0, 0),
        querylog: bool = True,
        n_queries: int = 48,
        coord_to_meters: str = "tangent_plane",
        n_times: int = 1,
        dt_seconds: float = 3600.0,
        advect_vel=None,
    ) -> Path:
        """Materialize a (height x width) crop at ``offset`` (oy, ox) into a WindArtifact.

        ``n_times > 1`` produces a *temporally evolving* artifact by wrapping the sampler in an
        :class:`AdvectedField` (the kinematic baseline) — requires ``advect_vel`` (per-level
        (u,v) m/s, e.g. ``velocity_from_stats(phi.stats)``). Time is written as datetime64 at
        ``dt_seconds`` spacing so the temporal metrics can recover the cadence.
        """
        oy, ox = offset
        y0, y1, x0, x1 = oy, oy + height, ox, ox + width
        lat, lon = self._coords(y0, y1, x0, x1)
        temporal = n_times > 1

        if temporal:
            from .advected import AdvectedField
            if advect_vel is None:
                raise ValueError("temporal artifact (n_times>1) needs advect_vel (per-level m/s)")
            adv = AdvectedField(self.sampler, advect_vel, pixel_km=self.pixel_km,
                                dt_seconds=dt_seconds)
            us, vs = [], []
            for ti in range(n_times):
                u, v = adv.field_uv(y0, y1, x0, x1, t=ti)
                us.append(u); vs.append(v)
            u, v = np.stack(us), np.stack(vs)                 # (n_times, L, H, W)
            tcoord = np.datetime64("2023-01-01T00") + np.arange(n_times) * np.timedelta64(
                int(dt_seconds), "s")
        else:
            u, v = self.sampler.field_uv(y0, y1, x0, x1)      # (n_levels, H, W) each
            tcoord = np.array([0])

        ds = artifact.make_field(u, v, level=self.levels, lat=lat, lon=lon, time=tcoord)

        seams = self.sampler.seam_lines(y0, y1, x0, x1)
        seam_local = {"y": [int(s - y0) for s in seams["y"]],
                      "x": [int(s - x0) for s in seams["x"]]}

        cfg = {"window": self.sampler.window, "stride": self.sampler.stride,
               "T": self.sampler.T, "pixel_km": self.pixel_km,
               "anchor": list(self.anchor), "region": [y0, y1, x0, x1]}
        if temporal:
            cfg.update({"n_times": int(n_times), "dt_seconds": float(dt_seconds),
                        "temporal_kind": "kinematic_advection"})
        attrs = artifact.default_attrs(
            generator={"name": self.name, "config": cfg},
            capabilities={"extent": "unbounded", "tiled": True,
                          "random_access": True, "temporally_evolving": bool(temporal)},
            conditioning={"lat": self.anchor[0], "lon": self.anchor[1],
                          "season": "n/a", "time": "n/a"},
            model_levels=self.levels, seed=self.seed,
            units="u,v:m/s; level:hPa", coord_to_meters=coord_to_meters,
        )
        attrs["seam_boundaries"] = seam_local
        attrs["hardware"] = _hardware(self.sampler.device)

        artifact.write(ds, attrs, out_path)
        if querylog:
            qds = self._build_querylog(y0, y1, x0, x1, n_queries=n_queries)
            artifact.write_querylog(qds, out_path)
        return Path(out_path)

    # ----- querylog: scattered queries with cost + revisits -----
    def _query_cost(self, y: int, x: int, level: int):
        """Cold single-point query: clears cache, times it, measures python-heap peak.

        latency is the real signal (TTFT-like); peak_mem is a python-heap proxy (torch
        tensor storage is C-level and only partially tracked) — the budget metric leans
        on latency + the bounded cache cap, not this number alone.
        """
        self.sampler.clear_cache()
        tracemalloc.start()
        t0 = time.perf_counter()
        f = self.sampler.materialize(y, y + 1, x, x + 1)      # (C,1,1) -> O(1) window eval
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        uv = f.reshape(self.n_levels, 2, 1, 1)[level, :, 0, 0].cpu().numpy()
        return float(uv[0]), float(uv[1]), dt, int(peak)

    def _build_querylog(self, y0, y1, x0, x1, *, n_queries: int):
        rng = np.random.default_rng(self.seed)
        H, W = y1 - y0, x1 - x0
        n_far = max(2, n_queries // 6)
        n_revisit = max(2, n_queries // 6)
        n_inside = n_queries - n_far - n_revisit

        # each point carries its own level so a revisit reuses (x,y,level,seed) exactly.
        def pt(y, x):
            return (int(y), int(x), int(rng.integers(0, self.n_levels)))

        # inside the crop, plus far-away points (probe unbounded extent + O(1) at distance)
        inside = [pt(rng.integers(y0, y1), rng.integers(x0, x1)) for _ in range(n_inside)]
        far = [pt(rng.integers(y0 + 10**6, y0 + 10**6 + H),
                  rng.integers(x0 - 10**6, x0 - 10**6 + W)) for _ in range(n_far)]
        unique = inside + far
        # revisits: repeat earlier points (after intervening queries) -> determinism probe
        revisit_idx = rng.choice(len(unique), size=n_revisit, replace=False)
        points = unique + [unique[i] for i in revisit_idx]

        rows = {k: [] for k in ("x", "y", "level", "t", "seed", "u", "v", "latency_s", "peak_mem")}
        for (qy, qx, lvl) in points:
            uu, vv, dt, peak = self._query_cost(qy, qx, lvl)
            rows["x"].append(qx); rows["y"].append(qy); rows["level"].append(self.levels[lvl])
            rows["t"].append(0); rows["seed"].append(self.seed)
            rows["u"].append(uu); rows["v"].append(vv)
            rows["latency_s"].append(dt); rows["peak_mem"].append(peak)
        return artifact.make_querylog(**rows, trajectory_source="random+far+revisit")
