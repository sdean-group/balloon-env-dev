"""BLE-VAE generator (offlineskies22) -> WindArtifact, to score against the harness.

A faithful pure-numpy reimplementation of the BLE decoder forward pass. We avoid the
real flax Decoder because its mutable-dataclass defaults break on Python 3.14; flax is
used only to restore the msgpack params. The decoder is simple and fully understood:

    latents(64) -> 3x[Dense(1000)+relu] -> Dense(4410) -> reshape(7,7,90)
                -> bilinear resize to (23,23,90) -> u=dΨ/dy, v=-dΨ/dx (curl of Ψ)
                -> (21,21,10,9,2)  = (lat, lng, pressure, time, uv)

Because the wind is the curl of a streamfunction Ψ, it is divergence-free BY
CONSTRUCTION and deliberately smooth (low-res Ψ + linear resize). That is exactly what
we expect the metrics to reveal: high vertical/rotational structure, but likely
over-smooth / under-intermittent vs real ERA5.

Grid: 21x21 @ ~50 km (±500 km box), 10 pressure levels 50-140 hPa, 9 time slices.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import flax

from .. import artifact

SF_LAT, SF_LON = 37.77, 237.58
# Pretrained BLE decoder weights (external dependency, not vendored). Override with
# $WINDEVAL_BLE_DECODER; default resolves to the BLE checkout beside the vault root.
# parents: generators[0]/windeval[1]/eval[2]/src[3]/balloon-env-dev[4]/<vault>[5]
_BLE_DEFAULT = (Path(__file__).resolve().parents[5] / "balloon-learning-environment"
                / "balloon_learning_environment" / "models" / "offlineskies22_decoder.msgpack")
WEIGHTS = Path(os.environ["WINDEVAL_BLE_DECODER"]) if os.environ.get("WINDEVAL_BLE_DECODER") else _BLE_DEFAULT

# FieldShape constants (from vae.FieldShape): 21x21 latlng, 10 pressure, 9 time, Ψ=7x7
N_LATLNG, N_PRESS, N_TIME = 21, 10, 9
FLOW_W = 7
DISP_KM = 500.0
P_MIN_HPA, P_MAX_HPA = 50.0, 140.0


def load_params(path=WEIGHTS) -> dict:
    return flax.serialization.msgpack_restore(Path(path).read_bytes())["params"]


def _resize_axis(a, n_out, axis):
    """Half-pixel-center bilinear resize along one axis (matches jax linear upsample)."""
    n_in = a.shape[axis]
    coord = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
    lo = np.floor(coord).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (coord - lo).reshape([n_out if i == axis else 1 for i in range(a.ndim)])
    return np.take(a, lo, axis) * (1 - w) + np.take(a, hi, axis) * w


def decode(params: dict, latents: np.ndarray) -> np.ndarray:
    """Forward pass -> wind field (21, 21, 10, 9, 2)."""
    z = latents
    for i in range(3):                                   # 3 hidden Dense(1000)+relu
        z = np.maximum(z @ params[f"Dense_{i}"]["kernel"] + params[f"Dense_{i}"]["bias"], 0)
    z = z @ params["Dense_3"]["kernel"] + params["Dense_3"]["bias"]   # (4410,)

    flow = z.reshape(FLOW_W, FLOW_W, N_PRESS * N_TIME)   # Ψ flow fields
    flow = _resize_axis(_resize_axis(flow, N_LATLNG + 2, 0), N_LATLNG + 2, 1)

    dflow_dy = ((np.roll(flow, -1, 0) - np.roll(flow, 1, 0)) / 2.0)[1:-1, 1:-1, :]
    dflow_dx = ((np.roll(flow, -1, 1) - np.roll(flow, 1, 1)) / 2.0)[1:-1, 1:-1, :]
    u = dflow_dy.reshape(N_LATLNG, N_LATLNG, N_PRESS, N_TIME)
    v = -dflow_dx.reshape(N_LATLNG, N_LATLNG, N_PRESS, N_TIME)
    return np.stack([u, v], axis=-1)                     # (lat,lng,press,time,uv)


def sample(params: dict, seed: int) -> np.ndarray:
    """One field. Latents are iid standard normal (the VAE prior) -> faithful sample."""
    latents = np.random.default_rng(seed).standard_normal(64).astype("float32")
    return decode(params, latents)


def _coords():
    pts = np.linspace(-DISP_KM, DISP_KM, N_LATLNG)               # km
    lat = SF_LAT + pts / 111.32
    lon = SF_LON + pts / (111.32 * np.cos(np.deg2rad(SF_LAT)))
    press = np.linspace(P_MIN_HPA, P_MAX_HPA, N_PRESS)
    return lat, lon, press


def to_artifact(field: np.ndarray, out_path, *, seed: int) -> Path:
    """(21,21,10,9,2) -> WindArtifact (time, level, y, x)."""
    lat, lon, press = _coords()
    u = np.transpose(field[..., 0], (3, 2, 0, 1))   # (time, level, y, x)
    v = np.transpose(field[..., 1], (3, 2, 0, 1))

    ds = artifact.make_field(u, v, level=press, lat=lat, lon=lon,
                             time=np.arange(field.shape[3]))
    attrs = artifact.default_attrs(
        generator={"name": "ble_vae", "config": {"weights": "offlineskies22", "seed": seed}},
        capabilities={"extent": "bounded", "tiled": False,
                      "random_access": True, "temporally_evolving": True},
        conditioning={"lat": SF_LAT, "lon": SF_LON, "season": "n/a", "time": "n/a"},
        model_levels=press, seed=seed, units="u,v:m/s; level:hPa",
        coord_to_meters="tangent_plane",
    )
    return artifact.write(ds, attrs, out_path)
