from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from correlation_analysis.core.panel import PERIODS_PER_YEAR, ReturnsPanel


@pytest.fixture
def panel():
    rng = np.random.default_rng(42)
    return ReturnsPanel(
        values=rng.normal(size=(100, 3)),
        names=("EURUSD", "GBPUSD", "GOLD"),
        freq="1d",
    )


def test_basic_construction(panel):
    assert panel.T == 100
    assert panel.N == 3
    assert panel.names == ("EURUSD", "GBPUSD", "GOLD")
    assert panel.kind == "returns"


def test_values_are_cast_to_float():
    panel = ReturnsPanel(values=[[1, 2], [3, 4]], names=("A", "B"))
    assert panel.values.dtype == np.float64


def test_panel_is_immutable(panel):
    with pytest.raises(FrozenInstanceError):
        panel.freq = "1h"


def test_rejects_1d_array():
    with pytest.raises(ValueError, match="2D"):
        ReturnsPanel(values=np.zeros(10), names=("A",))


def test_rejects_single_observation():
    with pytest.raises(ValueError, match="at least 2 observations"):
        ReturnsPanel(values=np.zeros((1, 2)), names=("A", "B"))


def test_rejects_misaligned_names():
    with pytest.raises(ValueError, match="entries but values"):
        ReturnsPanel(values=np.zeros((10, 3)), names=("A", "B"))


def test_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate names"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "A"))


def test_rejects_unknown_freq():
    with pytest.raises(ValueError, match="unknown freq"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "B"), freq="3s")


def test_rejects_invalid_kind():
    with pytest.raises(ValueError, match="kind must be"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "B"), kind="prices")


def test_rejects_nan():
    values = np.zeros((10, 2))
    values[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ReturnsPanel(values=values, names=("A", "B"))


def test_rejects_inf():
    values = np.zeros((10, 2))
    values[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        ReturnsPanel(values=values, names=("A", "B"))


def test_rejects_misaligned_index():
    with pytest.raises(ValueError, match="index has length"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "B"), index=np.arange(5))


def test_daily_annualization_factor(panel):
    assert panel.periods_per_year == 252


def test_m15_annualization_factor():
    panel = ReturnsPanel(np.zeros((10, 1)), ("A",), freq="15m")
    assert panel.periods_per_year == 252 * 96


def test_annualization_factors_are_monotonic():
    order = ["1mo", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1m"]
    factors = [PERIODS_PER_YEAR[f] for f in order]
    assert factors == sorted(factors)


def test_observation_ratio(panel):
    assert panel.observation_ratio == pytest.approx(100 / 3)


def test_select_filters_and_reorders(panel):
    subset = panel.select(["GOLD", "EURUSD"])
    assert subset.names == ("GOLD", "EURUSD")
    assert np.allclose(subset.values[:, 0], panel.column("GOLD"))
    assert np.allclose(subset.values[:, 1], panel.column("EURUSD"))


def test_select_preserves_metadata(panel):
    subset = panel.select(["GOLD"])
    assert subset.freq == panel.freq
    assert subset.kind == panel.kind


def test_select_rejects_unknown_column(panel):
    with pytest.raises(KeyError, match="not found"):
        panel.select(["EURUSD", "BITCOIN"])


def test_slice_rows(panel):
    mask = np.zeros(100, dtype=bool)
    mask[:10] = True
    subset = panel.slice_rows(mask)
    assert subset.T == 10
    assert np.allclose(subset.values, panel.values[:10])


def test_slice_rows_trims_index():
    p = ReturnsPanel(values=np.zeros((10, 2)), names=("A", "B"), index=np.arange(10))
    subset = p.slice_rows(np.array([True, False] * 5))
    assert np.array_equal(subset.index, np.array([0, 2, 4, 6, 8]))


def test_slice_rows_rejects_wrong_mask(panel):
    with pytest.raises(ValueError, match="mask must have shape"):
        panel.slice_rows(np.ones(5, dtype=bool))


def test_column_lookup(panel):
    assert np.allclose(panel.column("GBPUSD"), panel.values[:, 1])


def test_column_rejects_unknown_name(panel):
    with pytest.raises(KeyError, match="does not exist"):
        panel.column("SILVER")


def test_operations_do_not_mutate_source(panel):
    original = panel.values.copy()
    panel.select(["GOLD"])
    panel.slice_rows(np.ones(100, dtype=bool))
    assert np.allclose(panel.values, original)


def test_calendar_is_derived_from_named_constants():
    from correlation_analysis.core.panel import (
        MINUTES_PER_TRADING_DAY,
        TRADING_DAYS_PER_YEAR,
    )

    assert PERIODS_PER_YEAR["1m"] == TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY
    hourly = TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY / 60
    assert PERIODS_PER_YEAR["1h"] == hourly
    assert PERIODS_PER_YEAR["1d"] == TRADING_DAYS_PER_YEAR


def test_custom_calendar_overrides_the_lookup():
    panel = ReturnsPanel(
        np.zeros((10, 1)), ("BTC",), freq="1d", periods_per_year=365.0
    )
    assert panel.periods_per_year == 365.0


def test_custom_calendar_allows_an_unlisted_freq_label():
    panel = ReturnsPanel(
        np.zeros((10, 1)), ("A",), freq="8h", periods_per_year=1095.0
    )
    assert panel.periods_per_year == 1095.0


def test_custom_calendar_must_be_positive():
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        ReturnsPanel(np.zeros((10, 1)), ("A",), periods_per_year=0.0)


def test_unlisted_freq_without_override_is_rejected():
    with pytest.raises(ValueError, match="unknown freq"):
        ReturnsPanel(np.zeros((10, 1)), ("A",), freq="8h")
