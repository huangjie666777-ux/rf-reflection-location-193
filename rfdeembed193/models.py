from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class TouchstoneNetwork:
    filename: str
    frequencies: np.ndarray
    s_parameters: np.ndarray
    reference_impedance: float
    sha256: str

    @property
    def point_count(self) -> int:
        return int(self.frequencies.size)


@dataclass(frozen=True)
class DeembeddingResult:
    frequencies: np.ndarray
    s_parameters: np.ndarray
    reference_impedance: float
    left_swapped: bool
    right_swapped: bool
    input_hashes: dict[str, str]
    max_reembed_abs_error: float


@dataclass(frozen=True)
class ReflectionConfig:
    velocity: float
    range_start: float
    range_stop: float
    amplitude_threshold: float
    min_peak_distance: float
    zero_padding: int
    device_port: int


@dataclass(frozen=True)
class ReflectPeak:
    distance_m: float
    time_s: float
    amplitude: float
    phase_rad: float
    complex_value: complex
    bin_index: int


@dataclass(frozen=True)
class ReflectionResult:
    frequencies: np.ndarray
    frequency_step_hz: float
    transform_size: int
    window_weight_sum: float
    times_s: np.ndarray
    distances_m: np.ndarray
    trajectory: np.ndarray
    peaks: list[ReflectPeak]
    config: ReflectionConfig
    deembedding: DeembeddingResult


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
