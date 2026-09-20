from __future__ import annotations

import numpy as np

from correlation_analysis.core.panel import ReturnsPanel


def equicorrelation_matrix(n_series: int, rho: float) -> np.ndarray:
    if n_series < 1:
        raise ValueError(f"n_series must be positive; got {n_series}")

    lower_bound = -1.0 / (n_series - 1) if n_series > 1 else -1.0
    if not lower_bound < rho < 1.0:
        raise ValueError(
            f"rho must lie in ({lower_bound:.4f}, 1) to stay positive definite; "
            f"got {rho}"
        )

    matrix = np.full((n_series, n_series), float(rho))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def block_correlation_matrix(
    block_sizes: list[int], within: float, between: float
) -> np.ndarray:
    total = sum(block_sizes)
    matrix = np.full((total, total), float(between))

    start = 0
    for size in block_sizes:
        end = start + size
        matrix[start:end, start:end] = within
        start = end

    np.fill_diagonal(matrix, 1.0)

    smallest = float(np.linalg.eigvalsh(matrix).min())
    if smallest <= 0.0:
        raise ValueError(
            f"the requested block structure is not positive definite "
            f"(smallest eigenvalue {smallest:.3e}); lower within or raise between"
        )

    return matrix


def correlated_returns(
    target_correlation: np.ndarray,
    n_obs: int,
    volatilities: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    target = np.asarray(target_correlation, dtype=float)

    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"target_correlation must be square; got {target.shape}")

    try:
        cholesky = np.linalg.cholesky(target)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "target_correlation is not positive definite and cannot be sampled"
        ) from exc

    n_series = target.shape[0]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_obs, n_series))

    returns = noise @ cholesky.T

    if volatilities is not None:
        returns = returns * np.asarray(volatilities, dtype=float)

    return returns


def synthetic_panel(
    target_correlation: np.ndarray,
    n_obs: int,
    names: list[str] | None = None,
    freq: str = "1d",
    volatilities: np.ndarray | None = None,
    seed: int | None = None,
) -> ReturnsPanel:
    values = correlated_returns(
        target_correlation, n_obs, volatilities=volatilities, seed=seed
    )

    if names is None:
        names = [f"S{i}" for i in range(values.shape[1])]

    return ReturnsPanel(values=values, names=tuple(names), freq=freq)


def independent_panel(
    n_obs: int,
    n_series: int,
    names: list[str] | None = None,
    freq: str = "1d",
    seed: int | None = None,
) -> ReturnsPanel:
    return synthetic_panel(
        target_correlation=np.eye(n_series),
        n_obs=n_obs,
        names=names,
        freq=freq,
        seed=seed,
    )
