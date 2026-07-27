"""Tests for the stateful live InfiniteDiffusion viewer backend."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

MODULE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/eval/windeval/generators/infinite_diffusion/live_viewer"
)
sys.path.insert(0, str(MODULE_DIR))

from server import LiveSimulation  # noqa: E402
from infinite_coordinates import SpaceTimeGrid  # noqa: E402


class _ConstantSource:
    levels = np.array([57, 58])
    model_window_calls = 0
    model_forward_evaluations = 0
    phase_window_calls: dict[str, int] = {}

    def field_uv(self, t0, t1, y0, y1, x0, x1):  # noqa: ARG002
        self.model_window_calls += 1
        shape = (t1 - t0, 2, y1 - y0, x1 - x0)
        return np.full(shape, 10.0, np.float32), np.zeros(shape, np.float32)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def test_live_backend_generates_advects_and_serializes() -> None:
    simulation = LiveSimulation(
        _ConstantSource(),
        grid=SpaceTimeGrid(),
        chunk_size=16,
        chunk_frames=4,
        spatial_step=8,
        temporal_step=2,
        start_level=0,
        simulated_hours_per_second=0.1,
    )
    simulation.start()
    try:
        _wait_until(lambda: simulation.state()["active_chunk_id"] is not None)
        initial = simulation.state()
        _wait_until(lambda: simulation.state()["agent"]["x"] > initial["agent"]["x"])
        state = simulation.state()
        payload = json.loads(simulation.field_payload())

        assert state["agent"]["speed_mps"] == 10.0
        assert state["cached_chunks"] >= 1
        assert payload["ready"]
        assert payload["shape"] == [4, 2, 12, 12]
        assert len(payload["u"]) == 4 * 2 * 12 * 12
    finally:
        simulation.stop()


def test_live_controls_change_level_position_and_clock() -> None:
    simulation = LiveSimulation(
        _ConstantSource(),
        grid=SpaceTimeGrid(),
        chunk_size=16,
        chunk_frames=4,
        spatial_step=8,
        temporal_step=2,
    )
    simulation.control(
        {"running": False, "level_index": 1, "playback": 0.2, "x": 3.5, "y": 4.5}
    )
    state = simulation.state()
    assert not state["running"]
    assert state["agent"]["model_level"] == 58
    assert state["playback_hours_per_second"] == 0.2
    assert state["agent"]["x"] == 3.5
    assert state["agent"]["y"] == 4.5

    simulation.control({"reset": True})
    reset = simulation.state()
    assert reset["sim_time_hours"] == 0.0
    assert reset["agent"]["x"] == 8.0
    assert reset["agent"]["y"] == 8.0
