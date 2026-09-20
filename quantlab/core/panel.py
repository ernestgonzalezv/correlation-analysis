from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 252 * 1440,
    "5m": 252 * 288,
    "15m": 252 * 96,
    "30m": 252 * 48,
    "1h": 252 * 24,
    "4h": 252 * 6,
    "1d": 252,
    "1w": 52,
    "1mo": 12,
}

VALID_KINDS = ("returns", "pnl")


@dataclass(frozen=True)
class ReturnsPanel:
    values: np.ndarray
    names: tuple[str, ...]
    freq: str = "1d"
    kind: str = "returns"
    index: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", tuple(self.names))

        if values.ndim != 2:
            raise ValueError(
                f"values must be a 2D matrix (T, N); got ndim={values.ndim}"
            )

        n_rows, n_cols = values.shape

        if n_rows < 2:
            raise ValueError(f"at least 2 observations are required; got {n_rows}")

        if len(self.names) != n_cols:
            raise ValueError(
                f"names has {len(self.names)} entries but values has {n_cols} columns"
            )

        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate names are not allowed: {self.names}")

        if self.freq not in PERIODS_PER_YEAR:
            raise ValueError(
                f"unknown freq '{self.freq}'; valid: {sorted(PERIODS_PER_YEAR)}"
            )

        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}; got '{self.kind}'")

        if not np.isfinite(values).all():
            n_bad = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"values contains {n_bad} non-finite entries (NaN or inf); "
                "clean the data in the adapter layer"
            )

        if self.index is not None:
            index = np.asarray(self.index)
            object.__setattr__(self, "index", index)
            if len(index) != n_rows:
                raise ValueError(
                    f"index has length {len(index)} but values has {n_rows} rows"
                )

    @property
    def T(self) -> int:
        return self.values.shape[0]

    @property
    def N(self) -> int:
        return self.values.shape[1]

    @property
    def periods_per_year(self) -> float:
        return PERIODS_PER_YEAR[self.freq]

    @property
    def observation_ratio(self) -> float:
        return self.T / self.N

    def select(self, names: Sequence[str]) -> "ReturnsPanel":
        missing = [n for n in names if n not in self.names]
        if missing:
            raise KeyError(f"columns not found: {missing}")

        idx = [self.names.index(n) for n in names]
        return replace(self, values=self.values[:, idx], names=tuple(names))

    def slice_rows(self, mask: np.ndarray) -> "ReturnsPanel":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.T,):
            raise ValueError(f"mask must have shape ({self.T},); got {mask.shape}")

        new_index = None if self.index is None else self.index[mask]
        return replace(self, values=self.values[mask], index=new_index)

    def column(self, name: str) -> np.ndarray:
        if name not in self.names:
            raise KeyError(f"column '{name}' does not exist; available: {self.names}")
        return self.values[:, self.names.index(name)]

    def __repr__(self) -> str:
        return (
            f"ReturnsPanel(T={self.T}, N={self.N}, freq='{self.freq}', "
            f"kind='{self.kind}', names={self.names})"
        )
