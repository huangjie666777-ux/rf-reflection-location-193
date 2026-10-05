from __future__ import annotations

import io
import json
import zipfile

from .models import DeembeddingResult
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
