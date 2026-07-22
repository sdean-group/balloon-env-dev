import sys
from pathlib import Path

import numpy as np

WINDEVAL_PARENT = Path(__file__).resolve().parents[2] / "src" / "eval"
sys.path.insert(0, str(WINDEVAL_PARENT))

from windeval import artifact  # noqa: E402
from windeval.metrics.diagnostics import field_diagnostics  # noqa: E402


def test_field_diagnostics_are_finite():
    rng = np.random.default_rng(4)
    u = rng.normal(size=(2, 2, 16, 16)).astype("float32")
    v = rng.normal(size=(2, 2, 16, 16)).astype("float32")
    ds = artifact.make_field(
        u,
        v,
        level=np.array([49, 50]),
        lat=np.arange(16) * 0.25 + 25,
        lon=np.arange(16) * 0.25 + 225,
    )
    result = field_diagnostics(ds)
    assert result
    assert all(np.isfinite(value) for value in result.values())
