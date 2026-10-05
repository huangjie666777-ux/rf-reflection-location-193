from __future__ import annotations

import math
import re

import numpy as np

from .errors import TouchstoneError
from .models import TouchstoneNetwork, sha256_bytes

_FREQUENCY_SCALES = {
    "HZ": 1.0,
    "KHZ": 1.0e3,
    "MHZ": 1.0e6,
    "GHZ": 1.0e9,
}
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _finite_float(token: str, *, filename: str, line: int) -> float:
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise TouchstoneError(f"invalid numeric value '{token}'", filename=filename, line=line) from exc
    if not math.isfinite(value):
        raise TouchstoneError(f"non-finite numeric value '{token}'", filename=filename, line=line)
    return value


def parse_touchstone(data: bytes, filename: str) -> TouchstoneNetwork:
    raw = data.decode("utf-8", errors="replace")
    digest = sha256_bytes(data)
    scale = None
    value_format = None
    reference_impedance = None
    option_line = None
    rows: list[tuple[float, float, float, float, float, float, float, float, float]] = []
    data_lines: list[int] = []

    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        comment_at = raw_line.find("!")
        line = raw_line[:comment_at] if comment_at >= 0 else raw_line
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            if option_line is not None:
                raise TouchstoneError("duplicate option line", filename=filename, line=line_number)
            tokens = stripped.split()
            if (
                len(tokens) != 6
                or tokens[0] != "#"
                or tokens[1].upper() not in _FREQUENCY_SCALES
                or tokens[2].upper() != "S"
                or tokens[3].upper() not in {"RI", "MA", "DB"}
                or tokens[4].upper() != "R"
            ):
                raise TouchstoneError(
                    "option line must be '# <Hz|kHz|MHz|GHz> S <RI|MA|DB> R <positive impedance>'",
                    filename=filename,
                    line=line_number,
                )
            impedance = _finite_float(tokens[5], filename=filename, line=line_number)
            if impedance <= 0.0:
                raise TouchstoneError("reference impedance must be positive", filename=filename, line=line_number)
            scale = _FREQUENCY_SCALES[tokens[1].upper()]
            value_format = tokens[3].upper()
            reference_impedance = impedance
            option_line = line_number
            continue

        if option_line is None:
            raise TouchstoneError("data appears before complete option line", filename=filename, line=line_number)

        tokens = stripped.split()
        if len(tokens) != 9:
            raise TouchstoneError(
                f"expected 9 values (frequency, S11, S21, S12, S22), found {len(tokens)}",
                filename=filename,
                line=line_number,
            )
        values = tuple(_finite_float(token, filename=filename, line=line_number) for token in tokens)
        if any(not _NUMBER_RE.fullmatch(token) for token in tokens):
            raise TouchstoneError("malformed numeric token", filename=filename, line=line_number)
        rows.append(values)  # type: ignore[arg-type]
        data_lines.append(line_number)

    if option_line is None:
        raise TouchstoneError("missing complete option line", filename=filename)
    if not 2 <= len(rows) <= 20_000:
        raise TouchstoneError(
            f"expected 2 to 20000 frequency points, found {len(rows)}",
            filename=filename,
            line=option_line,
        )

    array = np.asarray(rows, dtype=np.float64)
    frequencies = array[:, 0] * scale
    if not np.all(np.isfinite(frequencies)):
        bad = int(np.flatnonzero(~np.isfinite(frequencies))[0])
        raise TouchstoneError("scaled frequency is not finite", filename=filename, line=data_lines[bad])
    if np.any(frequencies <= 0.0):
        bad = int(np.flatnonzero(frequencies <= 0.0)[0])
        raise TouchstoneError("frequency must be positive", filename=filename, line=data_lines[bad])
    if np.any(np.diff(frequencies) <= 0.0):
        bad = int(np.flatnonzero(np.diff(frequencies) <= 0.0)[0]) + 1
        raise TouchstoneError("frequencies must be strictly increasing", filename=filename, line=data_lines[bad])

    s_parameters = np.empty((frequencies.size, 2, 2), dtype=np.complex128)
    for point_index in range(frequencies.size):
        line = data_lines[point_index]
        values = array[point_index, 1:]
        complex_values = []
        for pair_index in range(4):
            first = values[2 * pair_index]
            second = values[2 * pair_index + 1]
            if value_format == "RI":
                complex_values.append(complex(first, second))
            elif value_format == "MA":
                if first < 0.0:
                    raise TouchstoneError("magnitude must be non-negative", filename=filename, line=line)
                complex_values.append(first * complex(math.cos(math.radians(second)), math.sin(math.radians(second))))
            else:
                magnitude = 10.0 ** (first / 20.0)
                complex_values.append(magnitude * complex(math.cos(math.radians(second)), math.sin(math.radians(second))))
        s_parameters[point_index, 0, 0] = complex_values[0]
        s_parameters[point_index, 0, 1] = complex_values[2]
        s_parameters[point_index, 1, 0] = complex_values[1]
        s_parameters[point_index, 1, 1] = complex_values[3]

    if not np.all(np.isfinite(s_parameters.real)) or not np.all(np.isfinite(s_parameters.imag)):
        raise TouchstoneError("converted S-parameters must be finite", filename=filename, line=option_line)

    return TouchstoneNetwork(filename, frequencies, s_parameters, float(reference_impedance), digest)


def write_touchstone(frequencies: np.ndarray, s_parameters: np.ndarray, reference_impedance: float) -> bytes:
    lines = ["# Hz S RI R {:.12g}".format(reference_impedance), "! Hz S11_real S11_imag S21_real S21_imag S12_real S12_imag S22_real S22_imag"]
    for frequency, matrix in zip(frequencies, s_parameters, strict=True):
        order = (matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1])
        values = [f"{frequency:.12g}"]
        for value in order:
            values.extend((f"{value.real:.17g}", f"{value.imag:.17g}"))
        lines.append(" ".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")
