from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rfdeembed193.api import app
from rfdeembed193.deembed import deembed
from rfdeembed193.errors import ParameterError
from rfdeembed193.models import DeembeddingResult
from rfdeembed193.reflect import check_frequency_grid, locate_reflections, validate_config

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"
LOCATE_FORM = {
    "left_swapped": "false",
    "right_swapped": "false",
    "port": "1",
    "velocity_m_per_s": "200000000",
    "distance_min_m": "0.0",
    "distance_max_m": "4.5",
    "amplitude_threshold": "0.05",
    "min_peak_distance_m": "0.3",
    "zero_padding": "4",
}


def _device_result(frequencies, s11):
    s = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    s[:, 0, 0] = s11
    s[:, 1, 0] = 0.7
    s[:, 0, 1] = 0.7
    return DeembeddingResult(
        frequencies=frequencies,
        s_parameters=s,
        reference_impedance=50.0,
        left_swapped=False,
        right_swapped=False,
        input_hashes={},
        max_reembed_abs_error=0.0,
    )


def _config(**overrides):
    values = dict(
        port=1,
        velocity_m_per_s=2.0e8,
        distance_min_m=0.0,
        distance_max_m=4.5,
        amplitude_threshold=0.05,
        min_peak_distance_m=0.3,
        zero_padding=4,
    )
    values.update(overrides)
    return validate_config(**values)


def test_two_known_reflections_are_located():
    frequencies = np.linspace(100.0e6, 8.1e9, 401)
    velocity = 2.0e8
    s11 = np.zeros(frequencies.size, dtype=np.complex128)
    for distance, amplitude in ((1.2, 0.30), (2.8, 0.15)):
        s11 += amplitude * np.exp(-2j * np.pi * frequencies * 2.0 * distance / velocity)
    result = locate_reflections(_device_result(frequencies, s11), _config())
    assert result.transform_length == 2048
    assert [round(p.distance_m, 1) for p in result.peaks] == [1.2, 2.8]
    assert abs(result.peaks[0].magnitude - 0.30) < 0.02
    assert abs(result.peaks[1].magnitude - 0.15) < 0.02


def test_unequal_spacing_rejected_without_resampling():
    frequencies = np.array([1.0e9, 2.0e9, 3.0e9, 4.001e9])
    with pytest.raises(ParameterError, match="equally spaced"):
        check_frequency_grid(frequencies)


def test_parameter_validation():
    with pytest.raises(ParameterError):
        _config(velocity_m_per_s=3.0e8)
    with pytest.raises(ParameterError):
        _config(distance_min_m=2.0, distance_max_m=2.0)
    with pytest.raises(ParameterError):
        _config(distance_min_m=-1.0)
    with pytest.raises(ParameterError):
        _config(amplitude_threshold=0.0)
    with pytest.raises(ParameterError):
        _config(zero_padding=3)
    frequencies = np.linspace(100.0e6, 8.1e9, 401)
    device = _device_result(frequencies, np.zeros(frequencies.size, dtype=np.complex128))
    with pytest.raises(ParameterError, match="unambiguous"):
        locate_reflections(device, _config(distance_max_m=5.0))  # unambiguous range is exactly 5.0 m


def test_min_peak_distance_suppression_and_empty_result():
    frequencies = np.linspace(100.0e6, 8.1e9, 401)
    velocity = 2.0e8
    s11 = np.zeros(frequencies.size, dtype=np.complex128)
    for distance, amplitude in ((1.2, 0.30), (1.25, 0.20)):
        s11 += amplitude * np.exp(-2j * np.pi * frequencies * 2.0 * distance / velocity)
    result = locate_reflections(_device_result(frequencies, s11), _config(min_peak_distance_m=0.3))
    assert len(result.peaks) == 1

    quiet = locate_reflections(_device_result(frequencies, s11), _config(amplitude_threshold=0.9))
    assert quiet.peaks == []


def test_http_locate_zip_contract():
    files = {
        "measurement": (EXAMPLE_DIR / "reflection_measurement.s2p").read_bytes(),
        "left_fixture": (EXAMPLE_DIR / "reflection_left_fixture.s2p").read_bytes(),
        "right_fixture": (EXAMPLE_DIR / "reflection_right_fixture.s2p").read_bytes(),
    }
    with TestClient(app) as client:
        response = client.post(
            "/locate",
            data=LOCATE_FORM,
            files=[(name, (f"{name}.s2p", data, "application/octet-stream")) for name, data in files.items()],
        )
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(archive.namelist()) == {
        "device.s2p",
        "reflection_trajectory.csv",
        "peaks.json",
        "config.json",
        "diagnostics.json",
    }
    peaks = json.loads(archive.read("peaks.json"))["peaks"]
    assert [round(p["distance_m"], 1) for p in peaks] == [1.2, 2.8]
    diagnostics = json.loads(archive.read("diagnostics.json"))
    assert diagnostics["bandwidth_hz"] == pytest.approx(8.0e9)
    assert diagnostics["unambiguous_distance_m"] == pytest.approx(5.0)
    header = archive.read("reflection_trajectory.csv").decode().splitlines()[0]
    assert header == "index,time_s,distance_m,real,imag,magnitude"


def test_http_locate_rejects_bad_velocity():
    files = {
        "measurement": (EXAMPLE_DIR / "reflection_measurement.s2p").read_bytes(),
        "left_fixture": (EXAMPLE_DIR / "reflection_left_fixture.s2p").read_bytes(),
        "right_fixture": (EXAMPLE_DIR / "reflection_right_fixture.s2p").read_bytes(),
    }
    form = dict(LOCATE_FORM, velocity_m_per_s="300000000")
    with TestClient(app) as client:
        response = client.post(
            "/locate",
            data=form,
            files=[(name, (f"{name}.s2p", data, "application/octet-stream")) for name, data in files.items()],
        )
    assert response.status_code == 400
