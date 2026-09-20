import numpy as np
import pytest

from quantlab.core.returns import (
    log_returns,
    panel_from_pnl,
    panel_from_prices,
    simple_returns,
)




def test_log_return_known_value():
    prices = np.array([[100.0], [102.0]])
    assert log_returns(prices)[0, 0] == pytest.approx(np.log(1.02))


def test_simple_return_known_value():
    prices = np.array([[100.0], [102.0]])
    assert simple_returns(prices)[0, 0] == pytest.approx(0.02)


def test_log_and_simple_agree_for_small_moves():
    prices = np.array([[100.0], [100.1], [100.05]])
    assert np.allclose(log_returns(prices), simple_returns(prices), atol=1e-5)


def test_constant_price_returns_zero():
    prices = np.full((5, 2), 50.0)
    assert np.allclose(log_returns(prices), 0.0)




def test_log_returns_are_additive():
    prices = np.array([[100.0], [105.0], [103.0], [110.0]])
    suma = log_returns(prices).sum()
    total = np.log(110.0 / 100.0)
    assert suma == pytest.approx(total)


def test_simple_returns_are_not_additive():
    prices = np.array([[100.0], [150.0], [75.0]])
    assert simple_returns(prices).sum() != pytest.approx(75.0 / 100.0 - 1.0)




def test_returns_have_one_row_less():
    prices = np.abs(np.random.default_rng(0).normal(100, 1, size=(50, 4)))
    assert log_returns(prices).shape == (49, 4)




def test_log_returns_reject_negative_prices():
    prices = np.array([[100.0], [-5.0]])
    with pytest.raises(ValueError, match="<= 0"):
        log_returns(prices)


def test_log_returns_reject_zero_price():
    prices = np.array([[100.0], [0.0]])
    with pytest.raises(ValueError, match="<= 0"):
        log_returns(prices)


def test_log_returns_reject_nan():
    prices = np.array([[100.0], [np.nan]])
    with pytest.raises(ValueError, match="NaN or inf"):
        log_returns(prices)


def test_log_returns_reject_1d():
    with pytest.raises(ValueError, match="2D"):
        log_returns(np.array([100.0, 101.0]))


def test_log_returns_reject_single_row():
    with pytest.raises(ValueError, match="at least 2 price rows"):
        log_returns(np.array([[100.0]]))




def test_panel_from_prices_aligns_index():
    prices = np.array([[100.0], [102.0], [101.0]])
    index = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])

    panel = panel_from_prices(prices, ["EURUSD"], index=index)

    assert panel.T == 2
    assert list(panel.index) == ["2024-01-02", "2024-01-03"]


def test_panel_from_prices_sets_returns_kind():
    prices = np.array([[100.0], [102.0], [101.0]])
    assert panel_from_prices(prices, ["A"]).kind == "returns"


def test_panel_from_prices_propagates_freq():
    prices = np.array([[100.0], [102.0], [103.0]])
    assert panel_from_prices(prices, ["A"], freq="15m").freq == "15m"


def test_panel_from_prices_rejects_invalid_method():
    prices = np.array([[100.0], [102.0], [101.0]])
    with pytest.raises(ValueError, match="'log' or 'simple'"):
        panel_from_prices(prices, ["A"], method="raro")


def test_panel_requires_two_observations():
    prices = np.array([[100.0], [102.0]])
    with pytest.raises(ValueError, match="at least 2 observations"):
        panel_from_prices(prices, ["A"])


def test_panel_from_pnl_accepts_negatives_and_zero():
    pnl = np.array([[120.0, -80.0], [-45.0, 0.0], [0.0, 210.0]])

    panel = panel_from_pnl(pnl, ["MBR", "MeanRev"])

    assert panel.kind == "pnl"
    assert panel.T == 3
    assert panel.N == 2
