from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = (".parquet", ".pq", ".csv")


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    rows: int
    first: pd.Timestamp
    last: pd.Timestamp


def discover_symbols(directory: str | Path) -> list[str]:
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    names = sorted(
        path.stem
        for path in directory.iterdir()
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not names:
        raise FileNotFoundError(
            f"no files with suffixes {SUPPORTED_SUFFIXES} found in {directory}"
        )

    return names


def _read_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_parquet(path)


def read_symbol(
    directory: str | Path,
    symbol: str,
    value_column: str = "close",
    date_column: str = "date",
) -> pd.Series:
    directory = Path(directory)

    for suffix in SUPPORTED_SUFFIXES:
        path = directory / f"{symbol}{suffix}"
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"no file for symbol '{symbol}' in {directory}")

    frame = _read_one(path)

    if date_column not in frame.columns:
        if frame.index.name == date_column:
            frame = frame.reset_index()
        else:
            raise KeyError(
                f"date column '{date_column}' not found in {path.name}; "
                f"available: {list(frame.columns)}"
            )

    if value_column not in frame.columns:
        raise KeyError(
            f"value column '{value_column}' not found in {path.name}; "
            f"available: {list(frame.columns)}"
        )

    series = frame.set_index(date_column)[value_column].astype(float)
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None).normalize()
    series.name = symbol

    return series[~series.index.duplicated(keep="last")].sort_index()


def coverage(
    directory: str | Path,
    symbols: list[str] | None = None,
    value_column: str = "close",
    date_column: str = "date",
) -> list[SymbolCoverage]:
    symbols = symbols or discover_symbols(directory)
    report = []

    for symbol in symbols:
        series = read_symbol(directory, symbol, value_column, date_column).dropna()
        if series.empty:
            continue
        report.append(
            SymbolCoverage(symbol, len(series), series.index[0], series.index[-1])
        )

    return report


def load_symbol_frame(
    directory: str | Path,
    symbols: list[str] | None = None,
    value_column: str = "close",
    date_column: str = "date",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    symbols = symbols or discover_symbols(directory)

    columns = {
        symbol: read_symbol(directory, symbol, value_column, date_column)
        for symbol in symbols
    }

    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = date_column

    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]

    return frame
