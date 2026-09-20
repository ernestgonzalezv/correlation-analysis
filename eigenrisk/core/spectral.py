from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEGATIVE_EIGENVALUE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class SpectralResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance: np.ndarray
    effective_bets: float

    @property
    def n_factors(self) -> int:
        return len(self.eigenvalues)

    def loadings(self, factor: int) -> np.ndarray:
        if not 0 <= factor < self.n_factors:
            raise IndexError(
                f"factor {factor} out of range [0, {self.n_factors - 1}]"
            )
        return self.eigenvectors[:, factor]

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance)

    def factors_for_variance(self, threshold: float = 0.90) -> int:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold}")
        return int(np.searchsorted(self.cumulative_variance(), threshold) + 1)


def _validate_symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square (N, N); got shape {matrix.shape}")

    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains NaN or inf")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("matrix must be symmetric")

    return matrix


def effective_bets(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("eigenvalues must sum to a positive number")

    proportions = eigenvalues / total
    return float(1.0 / np.sum(proportions**2))


def decompose(matrix: np.ndarray) -> SpectralResult:
    matrix = _validate_symmetric(matrix)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    smallest = float(eigenvalues.min())
    if smallest < -NEGATIVE_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"matrix is not positive semi-definite; smallest eigenvalue is "
            f"{smallest:.3e}, beyond the {-NEGATIVE_EIGENVALUE_TOLERANCE:.0e} "
            "tolerance for floating point error"
        )

    eigenvalues = np.clip(eigenvalues, 0.0, None)

    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("matrix has no positive variance to decompose")

    explained = eigenvalues / total

    return SpectralResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance=explained,
        effective_bets=effective_bets(eigenvalues),
    )
