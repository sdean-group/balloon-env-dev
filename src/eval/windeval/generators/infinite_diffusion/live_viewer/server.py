"""Live passive-drifter server backed by a persistent InfiniteDiffusion field."""
from __future__ import annotations

import argparse
import json
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from infinite_coordinates import SpaceTimeGrid  # noqa: E402


class WindSource(Protocol):
    levels: np.ndarray
    model_window_calls: int
    model_forward_evaluations: int
    phase_window_calls: dict[str, int]

    def field_uv(
        self, t0: int, t1: int, y0: int, y1: int, x0: int, x1: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


class InfiniteWindSource:
    def __init__(self, field: Any) -> None:
        self.field = field
        self.levels = np.asarray(field.sampler.stats.levels)

    @property
    def model_window_calls(self) -> int:
        return self.field.model_window_calls

    @property
    def model_forward_evaluations(self) -> int:
        return self.field.model_forward_evaluations

    @property
    def phase_window_calls(self) -> dict[str, int]:
        return dict(self.field.phase_window_calls)

    def field_uv(
        self, t0: int, t1: int, y0: int, y1: int, x0: int, x1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.field.field_uv(t0, t1, y0, y1, x0, x1)


class RepeatingArrayWindSource:
    """Small replay backend used to exercise the UI without loading a model."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.u = np.asarray(data["u"], dtype=np.float32)
        self.v = np.asarray(data["v"], dtype=np.float32)
        self.levels = np.asarray(data["levels"])
        self.model_window_calls = 0
        self.model_forward_evaluations = 0
        self.phase_window_calls: dict[str, int] = {}

    def field_uv(
        self, t0: int, t1: int, y0: int, y1: int, x0: int, x1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model_window_calls += 1
        ts = np.arange(t0, t1) % self.u.shape[0]
        ys = np.arange(y0, y1) % self.u.shape[2]
        xs = np.arange(x0, x1) % self.u.shape[3]
        return (
            self.u[np.ix_(ts, np.arange(self.u.shape[1]), ys, xs)].copy(),
            self.v[np.ix_(ts, np.arange(self.v.shape[1]), ys, xs)].copy(),
        )


@dataclass(frozen=True)
class Bounds:
    t0: int
    t1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def key(self) -> str:
        return f"t{self.t0}_y{self.y0}_x{self.x0}"

    def contains(self, t: float, y: float, x: float, *, interpolation: bool = True) -> bool:
        time_end = self.t1 - 1 if interpolation else self.t1
        return (
            self.t0 <= t < time_end
            and self.y0 <= y < self.y1 - 1
            and self.x0 <= x < self.x1 - 1
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "y0": self.y0,
            "y1": self.y1,
            "x0": self.x0,
            "x1": self.x1,
        }


@dataclass
class WindChunk:
    bounds: Bounds
    u: np.ndarray
    v: np.ndarray
    generation_seconds: float


def _bilinear(field: np.ndarray, x: float, y: float) -> float:
    ny, nx = field.shape
    x = float(np.clip(x, 0.0, nx - 1.001))
    y = float(np.clip(y, 0.0, ny - 1.001))
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return float(
        (1 - fy) * ((1 - fx) * field[y0, x0] + fx * field[y0, x0 + 1])
        + fy * ((1 - fx) * field[y0 + 1, x0] + fx * field[y0 + 1, x0 + 1])
    )


class LiveSimulation:
    def __init__(
        self,
        source: WindSource,
        *,
        grid: SpaceTimeGrid,
        chunk_size: int = 64,
        chunk_frames: int = 4,
        spatial_step: int = 32,
        temporal_step: int = 2,
        start_level: int = 8,
        simulated_hours_per_second: float = 0.08,
        max_chunks: int = 8,
    ) -> None:
        self.source = source
        self.grid = grid
        self.chunk_size = int(chunk_size)
        self.chunk_frames = int(chunk_frames)
        self.spatial_step = int(spatial_step)
        self.temporal_step = int(temporal_step)
        self.max_chunks = int(max_chunks)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.requests: queue.Queue[Bounds] = queue.Queue()
        self.requested: set[Bounds] = set()
        self.chunks: dict[Bounds, WindChunk] = {}
        self.chunk_order: list[Bounds] = []
        self.payload_cache: dict[str, bytes] = {}
        self.field_version = 0
        self.generating: Bounds | None = None
        self.last_generation_seconds: float | None = None
        self.error: str | None = None
        self.running = True
        self.playback = float(simulated_hours_per_second)
        self.sim_time = 0.0
        self.elapsed_hours = 0.0
        self.agent_x = self.chunk_size / 2
        self.agent_y = self.chunk_size / 2
        self.level_index = int(np.clip(start_level, 0, len(source.levels) - 1))
        self.agent_wind = (0.0, 0.0)
        self.waiting_for_field = True
        self.trail: list[tuple[float, float, float]] = [
            (self.agent_x, self.agent_y, self.sim_time)
        ]
        self._queue_needed_chunk()
        self.worker = threading.Thread(target=self._generation_loop, daemon=True)
        self.clock = threading.Thread(target=self._simulation_loop, daemon=True)

    def start(self) -> None:
        self.worker.start()
        self.clock.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=3)
        self.clock.join(timeout=3)

    def _target_bounds(self) -> Bounds:
        t0 = math.floor(self.sim_time / self.temporal_step) * self.temporal_step
        half_overlap = (self.chunk_size - self.spatial_step) / 2
        x0 = math.floor((self.agent_x - half_overlap) / self.spatial_step) * self.spatial_step
        y0 = math.floor((self.agent_y - half_overlap) / self.spatial_step) * self.spatial_step
        return Bounds(
            int(t0),
            int(t0 + self.chunk_frames),
            int(y0),
            int(y0 + self.chunk_size),
            int(x0),
            int(x0 + self.chunk_size),
        )

    def _queue_needed_chunk(self) -> None:
        bounds = self._target_bounds()
        if bounds not in self.chunks and bounds not in self.requested:
            self.requested.add(bounds)
            self.requests.put(bounds)

    def _generation_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                bounds = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                self.generating = bounds
                self.error = None
            started = time.perf_counter()
            try:
                u, v = self.source.field_uv(
                    bounds.t0,
                    bounds.t1,
                    bounds.y0,
                    bounds.y1,
                    bounds.x0,
                    bounds.x1,
                )
                seconds = time.perf_counter() - started
                chunk = WindChunk(bounds=bounds, u=u, v=v, generation_seconds=seconds)
                with self.lock:
                    self.chunks[bounds] = chunk
                    self.chunk_order.append(bounds)
                    self.last_generation_seconds = seconds
                    self.field_version += 1
                    while len(self.chunk_order) > self.max_chunks:
                        oldest = self.chunk_order.pop(0)
                        self.chunks.pop(oldest, None)
                        self.payload_cache.pop(oldest.key, None)
            except Exception as exc:  # pragma: no cover - surfaced in live UI
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
            finally:
                with self.lock:
                    self.generating = None
                    self.requested.discard(bounds)
                self.requests.task_done()

    def _active_chunk(self) -> WindChunk | None:
        candidates = [
            chunk
            for chunk in self.chunks.values()
            if chunk.bounds.contains(self.sim_time, self.agent_y, self.agent_x)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.bounds.t0, item.bounds.x0, item.bounds.y0))

    def _sample(self, chunk: WindChunk, t: float, x: float, y: float) -> tuple[float, float]:
        b = chunk.bounds
        local_t = float(np.clip(t - b.t0, 0.0, chunk.u.shape[0] - 1.001))
        t0 = int(local_t)
        t1 = min(t0 + 1, chunk.u.shape[0] - 1)
        alpha = local_t - t0
        lx, ly = x - b.x0, y - b.y0
        level = self.level_index
        u0 = _bilinear(chunk.u[t0, level], lx, ly)
        v0 = _bilinear(chunk.v[t0, level], lx, ly)
        u1 = _bilinear(chunk.u[t1, level], lx, ly)
        v1 = _bilinear(chunk.v[t1, level], lx, ly)
        return u0 * (1 - alpha) + u1 * alpha, v0 * (1 - alpha) + v1 * alpha

    def _advance(self, real_seconds: float) -> None:
        chunk = self._active_chunk()
        if chunk is None:
            self.waiting_for_field = True
            self._queue_needed_chunk()
            return
        self.waiting_for_field = False
        simulated_hours = real_seconds * self.playback
        if simulated_hours <= 0:
            return
        u0, v0 = self._sample(chunk, self.sim_time, self.agent_x, self.agent_y)
        dt_seconds = simulated_hours * 3600.0
        latitude = self.grid.lat_origin + self.grid.dlat * self.agent_y
        dx_m = 111_320.0 * max(0.1, math.cos(math.radians(latitude))) * self.grid.dlon
        dy_m = 111_320.0 * self.grid.dlat
        mid_x = self.agent_x + 0.5 * u0 * dt_seconds / dx_m
        mid_y = self.agent_y + 0.5 * v0 * dt_seconds / dy_m
        mid_t = self.sim_time + 0.5 * simulated_hours
        if chunk.bounds.contains(mid_t, mid_y, mid_x):
            u, v = self._sample(chunk, mid_t, mid_x, mid_y)
        else:
            u, v = u0, v0
        self.agent_x += u * dt_seconds / dx_m
        self.agent_y += v * dt_seconds / dy_m
        self.sim_time += simulated_hours
        self.elapsed_hours += simulated_hours
        self.agent_wind = (u, v)
        self.trail.append((self.agent_x, self.agent_y, self.sim_time))
        if len(self.trail) > 400:
            self.trail = self.trail[-400:]
        self._queue_needed_chunk()

    def _simulation_loop(self) -> None:
        previous = time.perf_counter()
        while not self.stop_event.wait(0.1):
            now = time.perf_counter()
            dt = min(0.25, now - previous)
            previous = now
            with self.lock:
                if self.running and not self.error:
                    self._advance(dt)

    def control(self, command: dict) -> None:
        with self.lock:
            if "running" in command:
                self.running = bool(command["running"])
            if "playback" in command:
                self.playback = float(np.clip(float(command["playback"]), 0.005, 1.0))
            if "level_index" in command:
                self.level_index = int(
                    np.clip(int(command["level_index"]), 0, len(self.source.levels) - 1)
                )
            if "x" in command and "y" in command:
                self.agent_x = float(command["x"])
                self.agent_y = float(command["y"])
                self.trail = [(self.agent_x, self.agent_y, self.sim_time)]
            if command.get("reset"):
                self.sim_time = 0.0
                self.elapsed_hours = 0.0
                self.agent_x = self.chunk_size / 2
                self.agent_y = self.chunk_size / 2
                self.agent_wind = (0.0, 0.0)
                self.trail = [(self.agent_x, self.agent_y, self.sim_time)]
            self._queue_needed_chunk()

    def state(self) -> dict:
        with self.lock:
            chunk = self._active_chunk()
            active_bounds = chunk.bounds if chunk else None
            lat = self.grid.lat_origin + self.grid.dlat * self.agent_y
            lon = self.grid.lon_origin + self.grid.dlon * self.agent_x
            timestamp = np.datetime64(self.grid.time_origin) + np.timedelta64(
                int(round(self.sim_time * 3600)), "s"
            )
            return {
                "running": self.running,
                "waiting_for_field": self.waiting_for_field,
                "error": self.error,
                "sim_time_hours": self.sim_time,
                "elapsed_hours": self.elapsed_hours,
                "timestamp": str(timestamp),
                "playback_hours_per_second": self.playback,
                "agent": {
                    "x": self.agent_x,
                    "y": self.agent_y,
                    "lat": lat,
                    "lon": lon,
                    "level_index": self.level_index,
                    "model_level": int(self.source.levels[self.level_index]),
                    "u_mps": self.agent_wind[0],
                    "v_mps": self.agent_wind[1],
                    "speed_mps": math.hypot(*self.agent_wind),
                },
                "trail": [
                    {"x": x, "y": y, "t": t}
                    for x, y, t in self.trail[-240:]
                ],
                "field_version": self.field_version,
                "active_chunk_id": active_bounds.key if active_bounds else None,
                "active_bounds": active_bounds.as_dict() if active_bounds else None,
                "generating_bounds": (
                    self.generating.as_dict() if self.generating is not None else None
                ),
                "queued_chunks": self.requests.qsize(),
                "cached_chunks": len(self.chunks),
                "last_generation_seconds": self.last_generation_seconds,
                "model_window_calls": self.source.model_window_calls,
                "model_forward_evaluations": self.source.model_forward_evaluations,
                "phase_window_calls": self.source.phase_window_calls,
                "levels": [int(level) for level in self.source.levels],
                "grid": {
                    "lat_origin": self.grid.lat_origin,
                    "lon_origin": self.grid.lon_origin,
                    "dlat": self.grid.dlat,
                    "dlon": self.grid.dlon,
                    "dt_hours": self.grid.dt_hours,
                },
            }

    def field_payload(self) -> bytes:
        with self.lock:
            chunk = self._active_chunk()
            if chunk is None:
                return json.dumps({"ready": False}).encode()
            cached = self.payload_cache.get(chunk.bounds.key)
            if cached is not None:
                return cached
            indices = np.linspace(0, self.chunk_size - 1, 12).round().astype(int)
            u = chunk.u[:, :, indices][:, :, :, indices]
            v = chunk.v[:, :, indices][:, :, :, indices]
            payload = {
                "ready": True,
                "chunk_id": chunk.bounds.key,
                "bounds": chunk.bounds.as_dict(),
                "shape": list(u.shape),
                "levels": [int(level) for level in self.source.levels],
                "u": np.round(u, 3).ravel().tolist(),
                "v": np.round(v, 3).ravel().tolist(),
            }
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            self.payload_cache[chunk.bounds.key] = encoded
            return encoded


class LiveRequestHandler(BaseHTTPRequestHandler):
    simulation: LiveSimulation
    page: bytes

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(self.page, "text/html; charset=utf-8")
        elif path == "/api/state":
            body = json.dumps(self.simulation.state(), separators=(",", ":")).encode()
            self._send(body, "application/json")
        elif path == "/api/field":
            self._send(self.simulation.field_payload(), "application/json")
        elif path == "/api/health":
            self._send(b'{"ok":true}', "application/json")
        else:
            self._send(b'{"error":"not found"}', "application/json", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/control":
            self._send(b'{"error":"not found"}', "application/json", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            command = json.loads(self.rfile.read(length) or b"{}")
            self.simulation.control(command)
            body = json.dumps(self.simulation.state(), separators=(",", ":")).encode()
            self._send(body, "application/json")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self._send(body, "application/json", HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args) -> None:
        if self.path != "/api/state":
            super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--demo-npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--outer-depth", type=int, default=1)
    parser.add_argument("--split-step", type=int)
    parser.add_argument("--split-steps", type=int, nargs="+")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lat-origin", type=float, default=25.0)
    parser.add_argument("--lon-origin", type=float, default=225.0)
    parser.add_argument("--time-origin", default="2023-01-15T00")
    parser.add_argument("--start-level-index", type=int, default=8)
    parser.add_argument("--playback-hours-per-second", type=float, default=0.08)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    return parser.parse_args()


def build_source(args: argparse.Namespace, grid: SpaceTimeGrid) -> WindSource:
    if args.demo_npz:
        return RepeatingArrayWindSource(Path(args.demo_npz))
    from spacetime import SpaceTimeSampler
    from spacetime_infinite import InfiniteSpaceTimeDiffusion

    sampler = SpaceTimeSampler(
        args.checkpoint,
        num_steps=args.num_steps,
        device=args.device,
        use_ema=True,
    )
    field = InfiniteSpaceTimeDiffusion(
        sampler,
        grid=grid,
        window=args.window,
        stride=args.stride,
        time_stride=args.time_stride,
        seed=args.seed,
        outer_depth=args.outer_depth,
        split_step=args.split_step,
        split_steps=args.split_steps,
        cache_bytes=int(args.cache_gb * 1024**3),
    )
    return InfiniteWindSource(field)


def main() -> None:
    args = parse_args()
    grid = SpaceTimeGrid(
        lat_origin=args.lat_origin,
        lon_origin=args.lon_origin,
        time_origin=args.time_origin,
    )
    print("[live] loading wind source", flush=True)
    source = build_source(args, grid)
    simulation = LiveSimulation(
        source,
        grid=grid,
        chunk_size=args.window,
        chunk_frames=4,
        spatial_step=args.stride,
        temporal_step=args.time_stride,
        start_level=args.start_level_index,
        simulated_hours_per_second=args.playback_hours_per_second,
    )
    page = (Path(__file__).with_name("viewer.html")).read_bytes()
    handler = type(
        "ConfiguredLiveRequestHandler",
        (LiveRequestHandler,),
        {"simulation": simulation, "page": page},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    node = __import__("socket").gethostname()
    print(f"[live] node={node} port={args.port}", flush=True)
    print(
        f"[live] tunnel: ssh -N -L {args.port}:{node}:{args.port} unicorn",
        flush=True,
    )
    print(f"[live] open: http://localhost:{args.port}", flush=True)
    simulation.start()

    def shutdown(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        simulation.stop()
        server.server_close()


if __name__ == "__main__":
    main()
