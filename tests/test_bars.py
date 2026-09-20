import numpy as np
import pandas as pd
import pytest

from correlation_analysis.data.bars import (
    coverage,
    discover_symbols,
    load_symbol_frame,
    read_symbol,
)


@pytest.fixture
def bar_directory(tmp_path):
    rng = np.random.default_rng(120)
    dates = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")

    for symbol in ["EURUSD", "GBPUSD", "XAUUSD"]:
        closes = 100.0 * np.exp(
            np.cumsum(rng.normal(scale=0.01, size=len(dates)))
        )
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": closes,
                "close": closes,
                "n_bars": 1440,
            }
        )
        frame.to_parquet(tmp_path / f"{symbol}.parquet")

    return tmp_path


def test_discovers_symbols(bar_directory):
    assert discover_symbols(bar_directory) == ["EURUSD", "GBPUSD", "XAUUSD"]


def test_rejects_a_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="not a directory"):
        discover_symbols(tmp_path / "absent")


def test_rejects_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files with suffixes"):
        discover_symbols(tmp_path)


def test_reads_a_single_symbol(bar_directory):
    series = read_symbol(bar_directory, "EURUSD")
    assert len(series) == 300
    assert series.name == "EURUSD"
    assert series.index.tz is None


def test_rejects_an_unknown_symbol(bar_directory):
    with pytest.raises(FileNotFoundError, match="no file for symbol"):
        read_symbol(bar_directory, "USDZZZ")


def test_rejects_an_unknown_value_column(bar_directory):
    with pytest.raises(KeyError, match="value column"):
        read_symbol(bar_directory, "EURUSD", value_column="vwap")


def test_rejects_an_unknown_date_column(bar_directory):
    with pytest.raises(KeyError, match="date column"):
        read_symbol(bar_directory, "EURUSD", date_column="timestamp")


def test_duplicate_dates_keep_the_last_observation(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True),
            "close": [1.0, 2.0, 3.0],
        }
    )
    frame.to_parquet(tmp_path / "DUP.parquet")

    series = read_symbol(tmp_path, "DUP")

    assert len(series) == 2
    assert series.iloc[0] == 2.0


def test_builds_an_aligned_frame(bar_directory):
    frame = load_symbol_frame(bar_directory)
    assert frame.shape == (300, 3)
    assert list(frame.columns) == ["EURUSD", "GBPUSD", "XAUUSD"]


def test_selects_a_symbol_subset(bar_directory):
    frame = load_symbol_frame(bar_directory, symbols=["XAUUSD", "EURUSD"])
    assert list(frame.columns) == ["XAUUSD", "EURUSD"]


def test_date_window_is_applied(bar_directory):
    frame = load_symbol_frame(bar_directory, start="2020-06-01", end="2020-08-31")
    assert frame.index.min() >= pd.Timestamp("2020-06-01")
    assert frame.index.max() <= pd.Timestamp("2020-08-31")


def test_coverage_reports_rows_and_span(bar_directory):
    report = {c.symbol: c for c in coverage(bar_directory)}
    assert report["EURUSD"].rows == 300
    assert report["EURUSD"].first < report["EURUSD"].last


def test_frame_feeds_the_analysis_end_to_end(bar_directory):
    from correlation_analysis import analyze
    from correlation_analysis.data.loaders import frame_to_panel

    frame = load_symbol_frame(bar_directory).dropna(how="any")
    report = analyze(frame_to_panel(frame, freq="1d").panel)

    assert report.panel.N == 3
    assert 1.0 <= report.effective_bets <= 3.0
