from __future__ import annotations

import numpy as np

from .errors import ComputationError
from .models import DeembeddingResult, TouchstoneNetwork

TRANSMISSION_EPSILON = 1.0e-12
CONDITION_LIMIT = 1.0e10


def _require_finite(name: str, values: np.ndarray, frequencies: np.ndarray) -> None:
    bad = np.flatnonzero(~(np.isfinite(values.real) & np.isfinite(values.imag)))
    if bad.size:
        index = int(bad[0])
        raise ComputationError(f"non-finite value while processing {name}", frequency_hz=float(frequencies[index]))


def swap_ports(s_parameters: np.ndarray) -> np.ndarray:
    swapped = np.empty_like(s_parameters)
    swapped[:, 0, 0] = s_parameters[:, 1, 1]
    swapped[:, 0, 1] = s_parameters[:, 1, 0]
    swapped[:, 1, 0] = s_parameters[:, 0, 1]
    swapped[:, 1, 1] = s_parameters[:, 0, 0]
    return swapped


def interpolate_complex(
    network: TouchstoneNetwork,
    target_frequencies: np.ndarray,
    *,
    swapped: bool,
) -> np.ndarray:
    source = network.s_parameters
    if swapped:
        source = swap_ports(source)

    if target_frequencies[0] < network.frequencies[0] or target_frequencies[-1] > network.frequencies[-1]:
        raise ComputationError(
            f"fixture '{network.filename}' does not cover the complete measured frequency range; extrapolation is not allowed"
        )

    indices = np.searchsorted(network.frequencies, target_frequencies, side="left")
    candidate = np.minimum(indices, network.frequencies.size - 1)
    exact = (indices < network.frequencies.size) & np.equal(network.frequencies[candidate], target_frequencies)
    upper = np.where(exact, np.minimum(indices, network.frequencies.size - 1), indices)
    lower = np.where(exact, upper, indices - 1)
    spans = network.frequencies[upper] - network.frequencies[lower]
    weights = np.zeros_like(target_frequencies)
    np.divide(
        target_frequencies - network.frequencies[lower],
        spans,
        where=~exact,
        out=weights,
    )
    aligned = source[lower] + (source[upper] - source[lower]) * weights[:, None, None]
    _require_finite(f"fixture '{network.filename}'", aligned, target_frequencies)
    return aligned


def s_to_transmission(s_parameters: np.ndarray, frequencies: np.ndarray, label: str) -> np.ndarray:
    s21 = s_parameters[:, 1, 0]
    scale = s21[:, None, None]
    t = np.empty_like(s_parameters)
    determinant = s_parameters[:, 0, 0] * s_parameters[:, 1, 1] - s_parameters[:, 0, 1] * s_parameters[:, 1, 0]
    t[:, 0, 0] = 1.0
    t[:, 0, 1] = -s_parameters[:, 1, 1]
    t[:, 1, 0] = s_parameters[:, 0, 0]
    t[:, 1, 1] = -determinant
    t = t / scale
    _require_finite(f"transmission matrix of {label}", t, frequencies)
    return t


def transmission_to_s(t_parameters: np.ndarray, frequencies: np.ndarray, label: str) -> np.ndarray:
    s = np.empty_like(t_parameters)
    s[:, 1, 0] = 1.0 / t_parameters[:, 0, 0]
    s[:, 1, 1] = -t_parameters[:, 0, 1] / t_parameters[:, 0, 0]
    s[:, 0, 0] = t_parameters[:, 1, 0] / t_parameters[:, 0, 0]
    s[:, 0, 1] = (
        t_parameters[:, 1, 1]
        - t_parameters[:, 1, 0] * t_parameters[:, 0, 1] / t_parameters[:, 0, 0]
    )
    _require_finite(f"S-parameters of {label}", s, frequencies)
    return s


def _check_transmission(
    name: str,
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    *,
    check_condition: bool,
) -> None:
    tiny = np.flatnonzero(np.abs(s_parameters[:, 1, 0]) <= TRANSMISSION_EPSILON)
    if tiny.size:
        index = int(tiny[0])
        raise ComputationError(f"|S21| of {name} is not greater than {TRANSMISSION_EPSILON:g}", frequency_hz=float(frequencies[index]))
    t = s_to_transmission(s_parameters, frequencies, name)
    singular_values = np.linalg.svd(t, compute_uv=False)
    conditions = singular_values[:, 0] / singular_values[:, 1]
    if check_condition:
        bad = np.flatnonzero((~np.isfinite(conditions)) | (conditions > CONDITION_LIMIT))
        if bad.size:
            index = int(bad[0])
            raise ComputationError(
                f"transmission matrix condition number of {name} exceeds {CONDITION_LIMIT:g}",
                frequency_hz=float(frequencies[index]),
            )


def deembed(
    measurement: TouchstoneNetwork,
    left: TouchstoneNetwork,
    right: TouchstoneNetwork,
    *,
    left_swapped: bool,
    right_swapped: bool,
) -> DeembeddingResult:
    if not (measurement.reference_impedance == left.reference_impedance == right.reference_impedance):
        raise ComputationError("measurement and both fixtures must use the same reference impedance")

    frequencies = measurement.frequencies
    measured_s = measurement.s_parameters
    left_s = interpolate_complex(left, frequencies, swapped=left_swapped)
    right_s = interpolate_complex(right, frequencies, swapped=right_swapped)

    _check_transmission("aligned measurement", measured_s, frequencies, check_condition=False)
    _check_transmission("aligned left fixture", left_s, frequencies, check_condition=True)
    _check_transmission("aligned right fixture", right_s, frequencies, check_condition=True)

    measured_t = s_to_transmission(measured_s, frequencies, "aligned measurement")
    left_t = s_to_transmission(left_s, frequencies, "aligned left fixture")
    right_t = s_to_transmission(right_s, frequencies, "aligned right fixture")
    # Cascade order is T_measurement = T_left @ T_device @ T_right.
    left_removed = np.linalg.solve(left_t, measured_t)
    device_t = np.linalg.solve(
        right_t.swapaxes(-2, -1),
        left_removed.swapaxes(-2, -1),
    ).swapaxes(-2, -1)
    _require_finite("de-embedded device transmission matrix", device_t, frequencies)
    device_s = transmission_to_s(device_t, frequencies, "de-embedded device")

    reembedded_t = left_t @ device_t @ right_t
    reembedded_s = transmission_to_s(reembedded_t, frequencies, "re-embedded verification")
    error = float(np.max(np.abs(reembedded_s - measured_s)))
    if not np.isfinite(error):
        raise ComputationError("re-embedding verification produced a non-finite error")

    return DeembeddingResult(
        frequencies=frequencies,
        s_parameters=device_s,
        reference_impedance=measurement.reference_impedance,
        left_swapped=left_swapped,
        right_swapped=right_swapped,
        input_hashes={
            "measurement": measurement.sha256,
            "left_fixture": left.sha256,
            "right_fixture": right.sha256,
        },
        max_reembed_abs_error=error,
    )
