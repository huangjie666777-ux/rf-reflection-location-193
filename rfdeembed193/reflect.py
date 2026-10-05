from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ParameterError
from .models import DeembeddingResult

LIGHT_SPEED_M_PER_S = 299_792_458.0
MIN_POINT_COUNT = 4
MAX_POINT_COUNT = 4096
SPACING_RELATIVE_TOLERANCE = 1.0e-8
ALLOWED_ZERO_PADDING = (1, 2, 4, 8)


@dataclass(frozen=True)
class ReflectionConfig:
    port: int
    velocity_m_per_s: float
    distance_min_m: float
    distance_max_m: float
    amplitude_threshold: float
    min_peak_distance_m: float
    zero_padding: int


@dataclass(frozen=True)
class ReflectionPeak:
    index: int
    time_s: float
    distance_m: float
    magnitude: float
    value: complex


@dataclass(frozen=True)
class ReflectionResult:
    config: ReflectionConfig
    frequency_start_hz: float
    frequency_spacing_hz: float
    point_count: int
    transform_length: int
    window_sum: float
    times_s: np.ndarray
    distances_m: np.ndarray
    trajectory: np.ndarray
    peaks: list[ReflectionPeak]


def validate_config(
    *,
    port: int,
    velocity_m_per_s: float,
    distance_min_m: float,
    distance_max_m: float,
    amplitude_threshold: float,
    min_peak_distance_m: float,
    zero_padding: int,
) -> ReflectionConfig:
    if port not in (1, 2):
        raise ParameterError("port must be 1 (S11) or 2 (S22)")
    numeric = {
        "velocity_m_per_s": velocity_m_per_s,
        "distance_min_m": distance_min_m,
        "distance_max_m": distance_max_m,
        "amplitude_threshold": amplitude_threshold,
        "min_peak_distance_m": min_peak_distance_m,
    }
    for name, value in numeric.items():
        if not np.isfinite(value):
            raise ParameterError(f"{name} must be finite")
    if not 0.0 < velocity_m_per_s <= LIGHT_SPEED_M_PER_S:
        raise ParameterError(f"velocity_m_per_s must be in (0, {LIGHT_SPEED_M_PER_S:g}]")
    if distance_min_m < 0.0:
        raise ParameterError("distance_min_m must be non-negative")
    if not distance_min_m < distance_max_m:
        raise ParameterError("search interval must be strictly increasing (distance_min_m < distance_max_m)")
    if amplitude_threshold <= 0.0:
        raise ParameterError("amplitude_threshold must be positive")
    if min_peak_distance_m <= 0.0:
        raise ParameterError("min_peak_distance_m must be positive")
    if zero_padding not in ALLOWED_ZERO_PADDING:
        raise ParameterError(f"zero_padding must be one of {ALLOWED_ZERO_PADDING}")
    return ReflectionConfig(
        port=port,
        velocity_m_per_s=velocity_m_per_s,
        distance_min_m=distance_min_m,
        distance_max_m=distance_max_m,
        amplitude_threshold=amplitude_threshold,
        min_peak_distance_m=min_peak_distance_m,
        zero_padding=zero_padding,
    )


def check_frequency_grid(frequencies: np.ndarray) -> float:
    count = int(frequencies.size)
    if not MIN_POINT_COUNT <= count <= MAX_POINT_COUNT:
        raise ParameterError(
            f"reflection localization requires {MIN_POINT_COUNT} to {MAX_POINT_COUNT} frequency points, found {count}"
        )
    spacings = np.diff(frequencies)
    spacing = float(np.mean(spacings))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ParameterError("mean frequency spacing must be positive and finite")
    relative_deviation = float(np.max(np.abs(spacings - spacing)) / spacing)
    if relative_deviation > SPACING_RELATIVE_TOLERANCE:
        raise ParameterError(
            f"frequency points are not equally spaced (max relative deviation {relative_deviation:.3g} "
            f"exceeds {SPACING_RELATIVE_TOLERANCE:g}); resampling is not performed"
        )
    return spacing


def locate_reflections(device: DeembeddingResult, config: ReflectionConfig) -> ReflectionResult:
    spacing = check_frequency_grid(device.frequencies)
    unambiguous_distance = config.velocity_m_per_s / (2.0 * spacing)
    if not config.distance_max_m < unambiguous_distance:
        raise ParameterError(
            f"distance_max_m ({config.distance_max_m:g} m) must be smaller than the unambiguous range "
            f"v/(2*df) = {unambiguous_distance:g} m"
        )

    series = device.s_parameters[:, 0, 0] if config.port == 1 else device.s_parameters[:, 1, 1]
    count = int(series.size)
    window = np.hanning(count)  # symmetric Hann window
    window_sum = float(np.sum(window))
    target = count * config.zero_padding
    transform_length = 1 << (target - 1).bit_length()  # smallest power of two >= target

    spectrum = np.fft.ifft(series * window, n=transform_length) * (transform_length / window_sum)
    times = np.arange(transform_length, dtype=np.float64) / (transform_length * spacing)
    # Keep the phase contribution of the actual start frequency f0: the IFFT alone
    # assumes the first sample sits at DC, so re-apply exp(-j*2*pi*f0*t).
    trajectory = spectrum * np.exp(-2j * np.pi * device.frequencies[0] * times)
    distances = config.velocity_m_per_s * times / 2.0

    peaks = _pick_peaks(times, distances, trajectory, config)
    return ReflectionResult(
        config=config,
        frequency_start_hz=float(device.frequencies[0]),
        frequency_spacing_hz=spacing,
        point_count=count,
        transform_length=transform_length,
        window_sum=window_sum,
        times_s=times,
        distances_m=distances,
        trajectory=trajectory,
        peaks=peaks,
    )


def _pick_peaks(
    times: np.ndarray,
    distances: np.ndarray,
    trajectory: np.ndarray,
    config: ReflectionConfig,
) -> list[ReflectionPeak]:
    magnitude = np.abs(trajectory)
    candidates: list[int] = []
    for index in range(1, magnitude.size - 1):
        if not (config.distance_min_m <= distances[index] <= config.distance_max_m):
            continue
        if magnitude[index] < config.amplitude_threshold:
            continue
        # Internal local maximum; on a plateau the left edge (smaller distance) wins.
        if magnitude[index] > magnitude[index - 1] and magnitude[index] >= magnitude[index + 1]:
            candidates.append(index)

    candidates.sort(key=lambda k: (-magnitude[k], distances[k]))
    selected: list[int] = []
    for index in candidates:
        if all(abs(distances[index] - distances[other]) >= config.min_peak_distance_m for other in selected):
            selected.append(index)
    selected.sort(key=lambda k: distances[k])
    return [
        ReflectionPeak(
            index=index,
            time_s=float(times[index]),
            distance_m=float(distances[index]),
            magnitude=float(magnitude[index]),
            value=complex(trajectory[index]),
        )
        for index in selected
    ]
