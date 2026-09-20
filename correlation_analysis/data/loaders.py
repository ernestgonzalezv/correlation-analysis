from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from correlation_analysis.core.panel import ReturnsPanel
from correlation_analysis.core.returns import panel_from_pnl, panel_from_prices

PARQUET_SUFFIXES = (".parquet", ".pq")
CSV_SUFFIXES = (".csv", ".txt")


@dataclass(frozen=True)
class LoadResult:
    panel: ReturnsPanel
    rows_read: int
    rows_dropped: int
    columns_dropped: tuple[str, ...]

    @property
    def completeness(self) -> float:
        return 1.0 - self.rows_dropped / self.rows_read if self.rows_read else 0.0


def read_frame(path: str | Path, index_column: str | None = None) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    suffix = path.suffix.lower()

    if suffix in PARQUET_SUFFIXES:
        frame = pd.read_parquet(path)
    elif suffix in CSV_SUFFIXES:
        frame = pd.read_csv(path)
    else:
        raise ValueError(
            f"unsupported extension '{suffix}'; expected one of "
            f"{CSV_SUFFIXES + PARQUET_SUFFIXES}"
        )

    if index_column is not None:
        if index_column not in frame.columns:
            raise KeyError(
                f"index column '{index_column}' not found; available: "
                f"{list(frame.columns)}"
            )
        frame = frame.set_index(index_column)

    return frame


def frame_to_panel(
    frame: pd.DataFrame,
    freq: str = "1d",
    kind: str = "returns",
    method: str = "log",
    periods_per_year: float | None = None,
) -> LoadResult:
    rows_read = len(frame)

    numeric = frame.select_dtypes(include="number")
    dropped_columns = tuple(c for c in frame.columns if c not in numeric.columns)

    if numeric.shape[1] < 1:
        raise ValueError(
            f"no numeric columns found; available: {list(frame.columns)}"
        )

    complete = numeric.dropna(axis=0, how="any")
    rows_dropped = rows_read - len(complete)

    if len(complete) < 2:
        raise ValueError(
            f"only {len(complete)} complete rows remain after dropping missing "
            f"values, out of {rows_read} read"
        )

    values = complete.to_numpy(dtype=float)
    names = [str(c) for c in complete.columns]
    index = complete.index.to_numpy()

    if kind == "pnl":
        panel = panel_from_pnl(
            values, names, freq=freq, index=index, periods_per_year=periods_per_year
        )
    elif kind == "returns":
        panel = panel_from_prices(
            values,
            names,
            freq=freq,
            index=index,
            method=method,
            periods_per_year=periods_per_year,
        )
    else:
        raise ValueError(f"kind must be 'returns' or 'pnl'; got '{kind}'")

    return LoadResult(
        panel=panel,
        rows_read=rows_read,
        rows_dropped=rows_dropped,
        columns_dropped=dropped_columns,
    )


def load_panel(
    path: str | Path,
    freq: str = "1d",
    kind: str = "returns",
    method: str = "log",
    index_column: str | None = None,
    periods_per_year: float | None = None,
) -> LoadResult:
    frame = read_frame(path, index_column=index_column)
    return frame_to_panel(
        frame,
        freq=freq,
        kind=kind,
        method=method,
        periods_per_year=periods_per_year,
    )


def write_panel(panel: ReturnsPanel, path: str | Path) -> Path:
    path = Path(path)
    frame = pd.DataFrame(
        panel.values,
        columns=list(panel.names),
        index=panel.index if panel.index is not None else np.arange(panel.T),
    )

    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        frame.to_parquet(path)
    elif suffix in CSV_SUFFIXES:
        frame.to_csv(path)
    else:
        raise ValueError(f"unsupported extension '{suffix}'")

    return path
