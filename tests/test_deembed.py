from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rfdeembed193.api import app
from rfdeembed193.deembed import deembed, s_to_transmission, swap_ports, transmission_to_s
from rfdeembed193.errors import ComputationError, TouchstoneError
from rfdeembed193.touchstone import parse_touchstone, write_touchstone


def _network(frequencies, matrix):
    matrices = np.broadcast_to(np.asarray(matrix, dtype=np.complex128), (frequencies.size, 2, 2)).copy()
    return parse_touchstone(write_touchstone(frequencies, matrices, 50.0), "synthetic.s2p")


def test_touchstone_formats_and_line_error():
    text = "# GHz S MA R 50\n! comment\n1 .9 10 .8 20 .7 30 .6 40\n2 .9 10 .8 20 .7 30 .6 40\n"
    parsed = parse_touchstone(text.encode(), "ma.s2p")
    assert parsed.frequencies[0] == 1.0e9
    assert abs(parsed.s_parameters[0, 1, 0] - 0.8 * np.exp(1j * np.deg2rad(20.0))) < 1e-14

    invalid = "# MHz S RI R 50\n1 0 0 1 0 0 0 0 0\n2 0 0 1 0 0 0 0 0\n! gap\n2 0 0 1 0 0 0 0 0\n"
    try:
        parse_touchstone(invalid.encode(), "bad.s2p")
    except TouchstoneError as exc:
        assert exc.filename == "bad.s2p"
        assert exc.line == 5
    else:
        raise AssertionError("expected strict frequency rejection")


def test_matrix_deembedding_recovers_asymmetric_device():
    frequencies = np.array([1.0e9, 2.0e9])
    device = np.array([[0.11 + 0.02j, 0.04 - 0.01j], [0.72 + 0.05j, -0.18 + 0.03j]], dtype=np.complex128)
    left = np.array([[0.03 - 0.02j, 0.02 + 0.01j], [0.91 + 0.03j, 0.05 - 0.01j]], dtype=np.complex128)
    right = np.array([[-0.07 + 0.02j, 0.03 - 0.02j], [0.85 - 0.06j, 0.08 + 0.01j]], dtype=np.complex128)
    measured_t = s_to_transmission(left[None], frequencies, "l") @ s_to_transmission(device[None], frequencies, "d") @ s_to_transmission(right[None], frequencies, "r")
    measured = transmission_to_s(measured_t, frequencies, "m")
    result = deembed(_network(frequencies, measured[0]), _network(frequencies, left), _network(frequencies, right), left_swapped=False, right_swapped=False)
    assert np.max(np.abs(result.s_parameters - device[None])) < 1e-12
    assert result.max_reembed_abs_error < 1e-12


def test_no_extrapolation():
    measured = _network(np.array([1.0, 2.0]), [[0, 1], [1, 0]])
    left = _network(np.array([1.5, 2.5]), [[0, 1], [1, 0]])
    right = _network(np.array([1.0, 2.0]), [[0, 1], [1, 0]])
    with pytest.raises(ComputationError):
        deembed(measured, left, right, left_swapped=False, right_swapped=False)


def test_port_swapping_recovers_when_fixture_is_reversed():
    frequencies = np.array([1.0e9, 2.0e9])
    device = np.array([[0.12 - 0.01j, 0.06 + 0.02j], [0.68 - 0.04j, 0.10 - 0.02j]], dtype=np.complex128)
    left = np.array([[0.04 + 0.01j, 0.03 - 0.02j], [0.90 - 0.02j, -0.04 + 0.02j]], dtype=np.complex128)
    right = np.array([[-0.06 - 0.03j, 0.04 + 0.01j], [0.86 + 0.05j, 0.07 - 0.02j]], dtype=np.complex128)
    measured_t = s_to_transmission(left[None], frequencies, "l") @ s_to_transmission(device[None], frequencies, "d") @ s_to_transmission(right[None], frequencies, "r")
    measured = transmission_to_s(measured_t, frequencies, "m")
    reversed_left = swap_ports(left[None])
    reversed_right = swap_ports(right[None])
    result = deembed(
        _network(frequencies, measured[0]),
        _network(frequencies, reversed_left[0]),
        _network(frequencies, reversed_right[0]),
        left_swapped=True,
        right_swapped=True,
    )
    assert np.max(np.abs(result.s_parameters - device[None])) < 1e-12


def test_http_zip_contract():
    from pathlib import Path

    example_dir = Path(__file__).resolve().parents[1] / "examples"
    files = {
        "measurement": (example_dir / "measurement.s2p").read_bytes(),
        "left_fixture": (example_dir / "left_fixture.s2p").read_bytes(),
        "right_fixture": (example_dir / "right_fixture.s2p").read_bytes(),
    }
    with TestClient(app) as client:
        response = client.post(
            "/deembed",
            data={"left_swapped": "false", "right_swapped": "false"},
            files=[(name, (f"{name}.s2p", data, "application/octet-stream")) for name, data in files.items()],
        )
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(archive.namelist()) == {"device.s2p", "diagnostics.json"}
    assert b"# Hz S RI R 50" in archive.read("device.s2p")
