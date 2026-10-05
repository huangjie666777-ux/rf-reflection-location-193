from __future__ import annotations

from pathlib import Path

import numpy as np

from rfdeembed193.deembed import s_to_transmission, transmission_to_s
from rfdeembed193.touchstone import write_touchstone

from .generate_example import LEFT, LEFT_SCALE, RIGHT, RIGHT_SCALE, frequency_dependent

VELOCITY_M_PER_S = 2.0e8
REFLECTIONS_PORT1 = ((1.2, 0.30), (2.8, 0.15))  # (distance_m, amplitude)
REFLECTIONS_PORT2 = ((3.5, 0.20),)


def reflection_series(frequencies: np.ndarray, reflections: tuple[tuple[float, float], ...]) -> np.ndarray:
    series = np.zeros(frequencies.size, dtype=np.complex128)
    for distance, amplitude in reflections:
        delay = 2.0 * distance / VELOCITY_M_PER_S
        series += amplitude * np.exp(-2j * np.pi * frequencies * delay)
    return series


def main() -> None:
    output = Path(__file__).resolve().parent
    measurement_frequencies = np.linspace(100.0e6, 8.1e9, 401)  # uniform 20 MHz spacing
    fixture_frequencies = np.linspace(50.0e6, 8.2e9, 164)

    device_s = np.zeros((measurement_frequencies.size, 2, 2), dtype=np.complex128)
    device_s[:, 0, 0] = reflection_series(measurement_frequencies, REFLECTIONS_PORT1)
    device_s[:, 1, 1] = reflection_series(measurement_frequencies, REFLECTIONS_PORT2)
    device_s[:, 1, 0] = 0.70 + 0.02j
    device_s[:, 0, 1] = 0.70 - 0.02j

    left_at_measurement = frequency_dependent(LEFT, measurement_frequencies, LEFT_SCALE)
    right_at_measurement = frequency_dependent(RIGHT, measurement_frequencies, RIGHT_SCALE)
    measured_t = (
        s_to_transmission(left_at_measurement, measurement_frequencies, "left")
        @ s_to_transmission(device_s, measurement_frequencies, "device")
        @ s_to_transmission(right_at_measurement, measurement_frequencies, "right")
    )
    measured_s = transmission_to_s(measured_t, measurement_frequencies, "measurement")

    left_s = frequency_dependent(LEFT, fixture_frequencies, LEFT_SCALE)
    right_s = frequency_dependent(RIGHT, fixture_frequencies, RIGHT_SCALE)

    (output / "reflection_measurement.s2p").write_bytes(write_touchstone(measurement_frequencies, measured_s, 50.0))
    (output / "reflection_left_fixture.s2p").write_bytes(write_touchstone(fixture_frequencies, left_s, 50.0))
    (output / "reflection_right_fixture.s2p").write_bytes(write_touchstone(fixture_frequencies, right_s, 50.0))


if __name__ == "__main__":
    main()
