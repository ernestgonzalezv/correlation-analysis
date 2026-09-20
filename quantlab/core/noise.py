from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_SCORES: dict[float, float] = {
    0.80: 1.281552,
    0.90: 1.644854,
    0.95: 1.959964,
    0.98: 2.326348,
    0.99: 2.575829,
}

MIN_OBSERVATION_RATIO = 10.0


@dataclass(frozen=True)
class NoiseBounds:
    lower: float
    upper: float
    ratio: float

    def is_signal(self, eigenvalue: float) -> bool:
        return eigenvalue > self.upper


def marchenko_pastur_bounds(n_obs: int, n_series: int) -> NoiseBounds:
    if n_obs <= n_series:
        raise ValueError(
            f"Marchenko-Pastur requires n_obs > n_series; "
            f"got n_obs={n_obs}, n_series={n_series}"
        )

    ratio = n_series / n_obs
    root = np.sqrt(ratio)

    return NoiseBounds(
        lower=float((1.0 - root) ** 2),
        upper=float((1.0 + root) ** 2),
        ratio=ratio,
    )


def signal_mask(eigenvalues: np.ndarray, n_obs: int, n_series: int) -> np.ndarray:
    bounds = marchenko_pastur_bounds(n_obs, n_series)
    return np.asarray(eigenvalues, dtype=float) > bounds.upper


def count_signal_factors(eigenvalues: np.ndarray, n_obs: int, n_series: int) -> int:
    return int(signal_mask(eigenvalues, n_obs, n_series).sum())


def correlation_stderr(correlation: np.ndarray, n_obs: int) -> np.ndarray:
    if n_obs <= 3:
        raise ValueError(f"n_obs must be greater than 3; got {n_obs}")

    correlation = np.asarray(correlation, dtype=float)
    return np.full_like(correlation, 1.0 / np.sqrt(n_obs - 3))


def correlation_interval(
    correlation: np.ndarray, n_obs: int, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    if level not in Z_SCORES:
        raise ValueError(
            f"unsupported confidence level {level}; available: {sorted(Z_SCORES)}"
        )

    if n_obs <= 3:
        raise ValueError(f"n_obs must be greater than 3; got {n_obs}")

    correlation = np.clip(np.asarray(correlation, dtype=float), -0.999999, 0.999999)

    z = np.arctanh(correlation)
    margin = Z_SCORES[level] / np.sqrt(n_obs - 3)

    return np.tanh(z - margin), np.tanh(z + margin)


def is_estimate_reliable(
    n_obs: int, n_series: int, min_ratio: float = MIN_OBSERVATION_RATIO
) -> bool:
    return (n_obs / n_series) >= min_ratio
