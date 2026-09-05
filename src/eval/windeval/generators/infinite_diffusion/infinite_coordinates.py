"""Shared global-coordinate mapping and deterministic noise for wind generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def coordinate_noise(
    channels: int,
    tau: int,
    height: int,
    width: int,
    *,
    t0: int,
    y0: int,
    x0: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coordinate-keyed Gaussian noise, generated on CPU for MPS compatibility."""
    cpu = torch.device("cpu")
    c = torch.arange(channels, dtype=torch.int64, device=cpu)[:, None, None, None]
    t = torch.arange(t0, t0 + tau, dtype=torch.int64, device=cpu)[None, :, None, None]
    y = torch.arange(y0, y0 + height, dtype=torch.int64, device=cpu)[None, None, :, None]
    x = torch.arange(x0, x0 + width, dtype=torch.int64, device=cpu)[None, None, None, :]
    h = torch.full((1, 1, 1, 1), int(seed), dtype=torch.int64, device=cpu)
    h = h ^ (x * 6364136223846793005)
    h = h ^ (y * 1442695040888963407)
    h = h ^ (t * 22695477)
    h = h ^ (c * 1103515245)
    h = h ^ (h >> 33)
    h = h * 2862933555777941757 + 3037000493
    h = h ^ (h >> 29)
    uniform = (h & ((1 << 53) - 1)).to(torch.float64) / float(1 << 53)
    uniform = uniform.clamp(1e-12, 1.0 - 1e-12)
    noise = 2.0**0.5 * torch.erfinv(2.0 * uniform - 1.0)
    return noise.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class SpaceTimeGrid:
    """Map integer tensor coordinates to physical wind-model conditioning."""

    lat_origin: float = 25.0
    lon_origin: float = 225.0
    dlat: float = 0.25
    dlon: float = 0.25
    time_origin: str = "2023-01-15T00"
    dt_hours: int = 1

    def coordinates(
        self,
        *,
        t0: int,
        y0: int,
        x0: int,
        tau: int,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat = self.lat_origin + self.dlat * np.arange(y0, y0 + height)
        lon = self.lon_origin + self.dlon * np.arange(x0, x0 + width)
        offsets = np.arange(t0, t0 + tau) * np.timedelta64(self.dt_hours, "h")
        times = np.datetime64(self.time_origin) + offsets
        return lat, lon, times
