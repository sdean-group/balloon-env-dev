from pathlib import Path

import numpy as np
import pytest

from windeval import artifact
from windeval.benchmark import _observation_condition_groups
from windeval.generators.rlhab_synthwinds.generate import (
    _nearest_smoothed_field,
    _profile_at_levels,
    parse_archive,
    radiosonde_layer_thickness,
)
from windeval.generators.rlhab_synthwinds.score import _spatial_reference


def _profile(lat, lon, u, v):
    return {
        "lat": lat,
        "lon": lon,
        "pressure": np.array([50.0, 100.0, 150.0]),
        "u": np.asarray(u, dtype=float),
        "v": np.asarray(v, dtype=float),
    }


def test_parse_current_wyoming_fm35_response(tmp_path: Path):
    path = tmp_path / "20230108T00-72493.html"
    path.write_text(
        """<html><pre>
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SPED   THTA   THTE   THTV
    hPa      m      C      C      %   g/kg    deg    m/s      K      K      K
-----------------------------------------------------------------------------
  130.0  15000  -50.0  -60.0     20   0.10    270   20.0  300.0  301.0  300.1
  100.0  17000  -55.0  -65.0     20   0.08    180   30.0  310.0  311.0  310.1
   70.0  19000  -60.0  -70.0     20   0.05     90   40.0  320.0  321.0  320.1
   50.0  21000  -65.0  -75.0     20   0.03               330.0  331.0  330.1
   40.0  22000  -67.0  -77.0     20   0.02      0   10.0  340.0  341.0  340.1
</pre></html>"""
    )
    result = parse_archive(path, "72493", (37.73, 237.78))
    assert len(result) == 1
    profile = result[0]
    assert str(profile["time"]).startswith("2023-01-08T00:00:00")
    assert profile["pressure"].tolist() == [130.0, 100.0, 70.0, 40.0]
    np.testing.assert_allclose(profile["u"], [20.0, 0.0, -40.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(profile["v"], [0.0, 30.0, 0.0, -10.0], atol=1e-6)
    assert profile["height"].tolist() == [15000.0, 17000.0, 19000.0, 22000.0]


def test_profile_interpolates_in_pressure():
    profile = _profile(0, 0, [1, 2, 3], [4, 6, 8])
    u, v = _profile_at_levels(profile, np.array([75.0, 125.0]))
    np.testing.assert_allclose(u, [1.5, 2.5])
    np.testing.assert_allclose(v, [5.0, 7.0])


def test_nearest_station_field_is_finite_and_smoothed():
    profiles = [
        _profile(0, 0, [1, 2, 3], [2, 3, 4]),
        _profile(0, 1, [3, 4, 5], [4, 5, 6]),
        _profile(1, 0, [5, 6, 7], [6, 7, 8]),
    ]
    u, v = _nearest_smoothed_field(
        profiles, np.array([75.0, 125.0]), np.linspace(0, 1, 8), np.linspace(0, 1, 8)
    )
    assert u.shape == v.shape == (2, 8, 8)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert np.ptp(u) < 2.0


def test_observation_conditions_use_seven_matching_days():
    times = np.array(
        [
            np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}")
            for month in (1, 4, 7, 10)
            for day in range(8, 15)
            for hour in (0, 12)
        ]
    )
    shape = (len(times), 2, 64, 64)
    ds = artifact.make_field(
        np.zeros(shape),
        np.ones(shape),
        level=np.array([100.0, 80.0]),
        lat=np.arange(64),
        lon=np.arange(64),
        time=times,
    )
    groups = _observation_condition_groups(ds, ds)
    assert len(groups) == 8
    assert all(len(samples) == 7 and reference.sizes["time"] == 7
               for samples, reference in groups)


def test_radiosonde_layer_thickness_uses_pressure_height_profiles(tmp_path: Path):
    template = """<html><pre>
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SPED   THTA   THTE   THTV
    hPa      m      C      C      %   g/kg    deg    m/s      K      K      K
-----------------------------------------------------------------------------
  130.0  15000  -50.0  -60.0     20   0.10    270   20.0  300.0  301.0  300.1
  100.0  17000  -55.0  -65.0     20   0.08    180   30.0  310.0  311.0  310.1
   70.0  19000  -60.0  -70.0     20   0.05     90   40.0  320.0  321.0  320.1
   40.0  22000  -67.0  -77.0     20   0.02      0   10.0  340.0  341.0  340.1
</pre></html>"""
    (tmp_path / "20230108T00-72493.html").write_text(template)
    dz = radiosonde_layer_thickness(tmp_path, np.array([70.0, 100.0, 130.0]))
    np.testing.assert_allclose(dz, [2000.0, 2000.0])


def test_spatial_reference_selects_complete_four_hour_protocol():
    times = np.array([
        np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}")
        for month in (1, 4, 7, 10)
        for day in range(8, 15)
        for hour in range(0, 24, 4)
    ])
    shape = (len(times), 2, 4, 4)
    ds = artifact.make_field(
        np.zeros(shape), np.ones(shape), level=np.array([100.0, 80.0]),
        lat=np.arange(4), lon=np.arange(4), time=times,
    )
    selected = _spatial_reference(ds)
    assert selected.sizes["time"] == 168


def test_spatial_reference_rejects_missing_floor_days():
    times = np.array([
        np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}")
        for month in (1, 4, 7, 10)
        for day in (8, 9)
        for hour in range(0, 24, 4)
    ])
    shape = (len(times), 2, 4, 4)
    ds = artifact.make_field(
        np.zeros(shape), np.ones(shape), level=np.array([100.0, 80.0]),
        lat=np.arange(4), lon=np.arange(4), time=times,
    )
    with pytest.raises(ValueError, match="incomplete"):
        _spatial_reference(ds)
