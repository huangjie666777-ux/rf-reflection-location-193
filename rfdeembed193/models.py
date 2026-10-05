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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
