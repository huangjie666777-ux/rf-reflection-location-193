from __future__ import annotations

from pathlib import Path

import numpy as np

from rfdeembed193.deembed import s_to_transmission, transmission_to_s
from rfdeembed193.touchstone import write_touchstone


def frequency_dependent(base: np.ndarray, frequencies: np.ndarray, scale: np.ndarray) -> np.ndarray:
    factor = (frequencies[:, None, None] - 100.0e6) / 100.0e6
    values = np.broadcast_to(base, (frequencies.size, 2, 2)).copy().astype(np.complex128)
    values += factor * np.broadcast_to(scale, (frequencies.size, 2, 2)).astype(np.complex128)
    return values


DEVICE = np.array([
    [0.10 + 0.02j, 0.05 - 0.03j],
    [0.70 + 0.05j, -0.15 + 0.01j],
], dtype=np.complex128)
DEVICE_SCALE = np.array([
    [0.010 + 0.004j, -0.004 + 0.002j],
    [-0.020 + 0.006j, 0.012 - 0.003j],
], dtype=np.complex128)
LEFT = np.array([
    [0.03 - 0.01j, 0.02 + 0.01j],
    [0.91 + 0.04j, 0.04 - 0.02j],
], dtype=np.complex128)
LEFT_SCALE = np.array([
    [0.004j, 0.003 - 0.002j],
    [-0.012 + 0.003j, -0.003j],
], dtype=np.complex128)
RIGHT = np.array([
    [-0.08 + 0.03j, 0.04 - 0.02j],
    [0.84 - 0.07j, 0.07 + 0.02j],
], dtype=np.complex128)
RIGHT_SCALE = np.array([
    [-0.004 - 0.002j, 0.002j],
    [0.010 - 0.004j, 0.005 + 0.001j],
], dtype=np.complex128)


def main() -> None:
    output = Path(__file__).resolve().parent
    measurement_frequencies = np.array([100.0e6, 150.0e6, 200.0e6])
    left_frequencies = np.array([50.0e6, 125.0e6, 250.0e6])
    right_frequencies = np.array([75.0e6, 175.0e6, 300.0e6])

    device_s = frequency_dependent(DEVICE, measurement_frequencies, DEVICE_SCALE)
    left_s = frequency_dependent(LEFT, left_frequencies, LEFT_SCALE)
    right_s = frequency_dependent(RIGHT, right_frequencies, RIGHT_SCALE)

    left_at_measurement = frequency_dependent(LEFT, measurement_frequencies, LEFT_SCALE)
    right_at_measurement = frequency_dependent(RIGHT, measurement_frequencies, RIGHT_SCALE)
    measured_t = (
        s_to_transmission(left_at_measurement, measurement_frequencies, "left")
        @ s_to_transmission(device_s, measurement_frequencies, "device")
        @ s_to_transmission(right_at_measurement, measurement_frequencies, "right")
    )
    measured_s = transmission_to_s(measured_t, measurement_frequencies, "measurement")

    (output / "measurement.s2p").write_bytes(write_touchstone(measurement_frequencies, measured_s, 50.0))
    (output / "left_fixture.s2p").write_bytes(write_touchstone(left_frequencies, left_s, 50.0))
    (output / "right_fixture.s2p").write_bytes(write_touchstone(right_frequencies, right_s, 50.0))


if __name__ == "__main__":
    main()
