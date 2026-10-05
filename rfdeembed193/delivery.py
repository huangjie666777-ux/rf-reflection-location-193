from __future__ import annotations

import io
import json
import zipfile

from .models import DeembeddingResult
from .reflect import ReflectionResult
from .touchstone import write_touchstone


def build_result_zip(result: DeembeddingResult) -> bytes:
    diagnostics = {
        "reference_impedance_ohm": result.reference_impedance,
        "frequency_count": int(result.frequencies.size),
        "frequency_start_hz": float(result.frequencies[0]),
        "frequency_stop_hz": float(result.frequencies[-1]),
        "port_orientation": {
            "left_fixture_swapped": result.left_swapped,
            "right_fixture_swapped": result.right_swapped,
        },
        "input_sha256": result.input_hashes,
        "max_reembed_complex_absolute_error": result.max_reembed_abs_error,
    }
    touchstone = write_touchstone(result.frequencies, result.s_parameters, result.reference_impedance)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("device.s2p", touchstone)
        archive.writestr("diagnostics.json", json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    return buffer.getvalue()


def _trajectory_csv(result: ReflectionResult) -> bytes:
    lines = ["index,time_s,distance_m,real,imag,magnitude"]
    for index in range(result.transform_length):
        value = result.trajectory[index]
        lines.append(
            "{},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}".format(
                index,
                result.times_s[index],
                result.distances_m[index],
                value.real,
                value.imag,
                abs(value),
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_locate_zip(device: DeembeddingResult, result: ReflectionResult) -> bytes:
    config = result.config
    bandwidth_hz = (result.point_count - 1) * result.frequency_spacing_hz
    unambiguous_distance_m = config.velocity_m_per_s / (2.0 * result.frequency_spacing_hz)
    configuration = {
        "port": config.port,
        "reflection_parameter": "S11" if config.port == 1 else "S22",
        "velocity_m_per_s": config.velocity_m_per_s,
        "search_interval_m": [config.distance_min_m, config.distance_max_m],
        "amplitude_threshold": config.amplitude_threshold,
        "min_peak_distance_m": config.min_peak_distance_m,
        "zero_padding": config.zero_padding,
        "window": "symmetric Hann",
        "point_count": result.point_count,
        "transform_length": result.transform_length,
        "frequency_start_hz": result.frequency_start_hz,
        "frequency_spacing_hz": result.frequency_spacing_hz,
    }
    peaks = [
        {
            "index": peak.index,
            "distance_m": peak.distance_m,
            "time_s": peak.time_s,
            "magnitude": peak.magnitude,
            "real": peak.value.real,
            "imag": peak.value.imag,
        }
        for peak in result.peaks
    ]
    diagnostics = {
        "reference_impedance_ohm": device.reference_impedance,
        "bandwidth_hz": bandwidth_hz,
        "unambiguous_distance_m": unambiguous_distance_m,
        "nominal_physical_resolution_m": config.velocity_m_per_s / (2.0 * bandwidth_hz),
        "port_orientation": {
            "left_fixture_swapped": device.left_swapped,
            "right_fixture_swapped": device.right_swapped,
        },
        "input_sha256": device.input_hashes,
        "max_reembed_complex_absolute_error": device.max_reembed_abs_error,
        "notes": [
            "Zero-padding only interpolates the transform; it does not improve the physical resolution set by the bandwidth.",
            "Peak magnitude is a reflection-strength indicator, not an impedance measurement.",
            "No DC extrapolation and no real-valued step construction are performed; the trajectory is the complex IFFT of the windowed band-limited reflection.",
        ],
    }
    touchstone = write_touchstone(device.frequencies, device.s_parameters, device.reference_impedance)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("device.s2p", touchstone)
        archive.writestr("reflection_trajectory.csv", _trajectory_csv(result))
        archive.writestr("peaks.json", json.dumps({"peaks": peaks}, indent=2) + "\n")
        archive.writestr("config.json", json.dumps(configuration, indent=2, sort_keys=True) + "\n")
        archive.writestr("diagnostics.json", json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    return buffer.getvalue()
