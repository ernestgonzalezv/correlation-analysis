import numpy as np
import pandas as pd
import pytest

from correlation_analysis.data.loaders import (
    frame_to_panel,
    load_panel,
    read_frame,
    write_panel,
)
from correlation_analysis.data.synthetic import equicorrelation_matrix, synthetic_panel


@pytest.fixture
def price_frame():
    rng = np.random.default_rng(80)
    steps = rng.normal(loc=0.0002, scale=0.01, size=(300, 3))
    prices = 100.0 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(
        prices,
        columns=["EURUSD", "GBPUSD", "XAUUSD"],
        index=pd.date_range("2023-01-02", periods=300, freq="B"),
    )


def test_reads_csv(tmp_path, price_frame):
    path = tmp_path / "prices.csv"
    price_frame.to_csv(path, index_label="date")
    assert read_frame(path, index_column="date").shape == (300, 3)


def test_reads_parquet(tmp_path, price_frame):
    path = tmp_path / "prices.parquet"
    price_frame.to_parquet(path)
    assert read_frame(path).shape == (300, 3)


def test_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        read_frame(tmp_path / "absent.csv")


def test_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "prices.xlsx"
    path.write_text("nothing")
    with pytest.raises(ValueError, match="unsupported extension"):
        read_frame(path)


def test_rejects_a_missing_index_column(tmp_path, price_frame):
    path = tmp_path / "prices.csv"
    price_frame.to_csv(path, index_label="date")
    with pytest.raises(KeyError, match="not found"):
        read_frame(path, index_column="timestamp")


def test_prices_become_a_returns_panel(price_frame):
    result = frame_to_panel(price_frame, freq="1d", kind="returns")
    assert result.panel.kind == "returns"
    assert result.panel.T == 299
    assert result.panel.names == ("EURUSD", "GBPUSD", "XAUUSD")


def test_pnl_is_wrapped_without_conversion():
    frame = pd.DataFrame(
        {"MBR": [120.0, -45.0, 0.0, 88.0], "Cal": [10.0, 5.0, -2.0, 7.0]}
    )
    result = frame_to_panel(frame, kind="pnl")
    assert result.panel.kind == "pnl"
    assert result.panel.T == 4


def test_incomplete_rows_are_dropped_and_counted(price_frame):
    frame = price_frame.copy()
    frame.iloc[5, 1] = np.nan
    frame.iloc[20, 0] = np.nan

    result = frame_to_panel(frame)

    assert result.rows_read == 300
    assert result.rows_dropped == 2
    assert result.completeness == pytest.approx(298 / 300)


def test_non_numeric_columns_are_reported(price_frame):
    frame = price_frame.copy()
    frame["session"] = "london"

    result = frame_to_panel(frame)

    assert result.columns_dropped == ("session",)
    assert result.panel.N == 3


def test_rejects_a_frame_without_numeric_columns():
    frame = pd.DataFrame({"session": ["london", "ny"], "venue": ["a", "b"]})
    with pytest.raises(ValueError, match="no numeric columns"):
        frame_to_panel(frame)


def test_rejects_a_frame_left_too_short_by_missing_values():
    frame = pd.DataFrame({"A": [1.0, np.nan, np.nan], "B": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="complete rows remain"):
        frame_to_panel(frame)


def test_rejects_an_unknown_kind(price_frame):
    with pytest.raises(ValueError, match="kind must be"):
        frame_to_panel(price_frame, kind="levels")


def test_custom_calendar_survives_the_round_trip(price_frame):
    result = frame_to_panel(price_frame, freq="1d", periods_per_year=365.0)
    assert result.panel.periods_per_year == 365.0


def test_index_is_carried_into_the_panel(price_frame):
    panel = frame_to_panel(price_frame).panel
    assert panel.index is not None
    assert len(panel.index) == panel.T


def test_load_panel_round_trips_through_disk(tmp_path, price_frame):
    path = tmp_path / "prices.csv"
    price_frame.to_csv(path, index_label="date")

    result = load_panel(path, index_column="date", freq="1d")

    assert result.panel.T == 299
    assert result.rows_dropped == 0


def test_write_panel_round_trips(tmp_path):
    panel = synthetic_panel(equicorrelation_matrix(3, rho=0.4), n_obs=200, seed=81)
    path = write_panel(panel, tmp_path / "panel.parquet")

    reloaded = frame_to_panel(read_frame(path), kind="pnl").panel

    assert reloaded.N == panel.N
    assert np.allclose(reloaded.values, panel.values)


def test_write_panel_rejects_an_unknown_extension(tmp_path):
    panel = synthetic_panel(np.eye(2), n_obs=50, seed=82)
    with pytest.raises(ValueError, match="unsupported extension"):
        write_panel(panel, tmp_path / "panel.xlsx")
