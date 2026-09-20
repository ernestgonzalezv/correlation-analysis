"""Tests de conversion precios -> retornos.

Los tests de valor conocido usan numeros calculables a mano. Si el
motor no reproduce ln(1.02) para un precio que sube 2%, no hay nada
mas que discutir.
"""

import numpy as np
import pytest

from quantlab.core.returns import (
    log_returns,
    panel_from_pnl,
    panel_from_prices,
    simple_returns,
)


# ----------------------------------------------------------------------
# Valores conocidos
# ----------------------------------------------------------------------


def test_log_return_valor_conocido():
    prices = np.array([[100.0], [102.0]])
    assert log_returns(prices)[0, 0] == pytest.approx(np.log(1.02))


def test_simple_return_valor_conocido():
    prices = np.array([[100.0], [102.0]])
    assert simple_returns(prices)[0, 0] == pytest.approx(0.02)


def test_log_y_simple_coinciden_para_cambios_chicos():
    """Para movimientos pequeños ln(1+x) ~ x. Verifica que no hay un
    error de escala grueso en ninguna de las dos."""
    prices = np.array([[100.0], [100.1], [100.05]])
    assert np.allclose(log_returns(prices), simple_returns(prices), atol=1e-5)


def test_retorno_de_precio_constante_es_cero():
    prices = np.full((5, 2), 50.0)
    assert np.allclose(log_returns(prices), 0.0)


# ----------------------------------------------------------------------
# Propiedad clave: aditividad
# ----------------------------------------------------------------------


def test_log_returns_son_aditivos():
    """La razon de usar log-retornos: la suma de los retornos del
    periodo es el log del cambio total. Si esto falla, toda la
    agregacion y anualizacion aguas abajo esta mal."""
    prices = np.array([[100.0], [105.0], [103.0], [110.0]])
    suma = log_returns(prices).sum()
    total = np.log(110.0 / 100.0)
    assert suma == pytest.approx(total)


def test_simple_returns_NO_son_aditivos():
    """Contraste explicito: los retornos simples se componen, no se
    suman. Este test documenta por que el default es log."""
    prices = np.array([[100.0], [150.0], [75.0]])
    assert simple_returns(prices).sum() != pytest.approx(75.0 / 100.0 - 1.0)


# ----------------------------------------------------------------------
# Forma
# ----------------------------------------------------------------------


def test_retornos_tienen_una_fila_menos():
    prices = np.abs(np.random.default_rng(0).normal(100, 1, size=(50, 4)))
    assert log_returns(prices).shape == (49, 4)


# ----------------------------------------------------------------------
# Validacion
# ----------------------------------------------------------------------


def test_log_returns_rechaza_precios_negativos():
    prices = np.array([[100.0], [-5.0]])
    with pytest.raises(ValueError, match="<= 0"):
        log_returns(prices)


def test_log_returns_rechaza_precio_cero():
    prices = np.array([[100.0], [0.0]])
    with pytest.raises(ValueError, match="<= 0"):
        log_returns(prices)


def test_log_returns_rechaza_nan():
    prices = np.array([[100.0], [np.nan]])
    with pytest.raises(ValueError, match="NaN o inf"):
        log_returns(prices)


def test_log_returns_rechaza_1d():
    with pytest.raises(ValueError, match="2D"):
        log_returns(np.array([100.0, 101.0]))


def test_log_returns_rechaza_una_sola_fila():
    with pytest.raises(ValueError, match="al menos 2 filas"):
        log_returns(np.array([[100.0]]))


# ----------------------------------------------------------------------
# Construccion de paneles
# ----------------------------------------------------------------------


def test_panel_from_prices_alinea_el_index():
    """El index debe perder su primer elemento igual que los retornos.
    Un desfase de una fila aqui corre todas las fechas y es invisible
    en los numeros."""
    prices = np.array([[100.0], [102.0], [101.0]])
    index = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])

    panel = panel_from_prices(prices, ["EURUSD"], index=index)

    assert panel.T == 2
    assert list(panel.index) == ["2024-01-02", "2024-01-03"]


def test_panel_from_prices_marca_kind_returns():
    prices = np.array([[100.0], [102.0], [101.0]])
    assert panel_from_prices(prices, ["A"]).kind == "returns"


def test_panel_from_prices_propaga_freq():
    prices = np.array([[100.0], [102.0], [103.0]])
    assert panel_from_prices(prices, ["A"], freq="15m").freq == "15m"


def test_panel_from_prices_rechaza_method_invalido():
    prices = np.array([[100.0], [102.0], [101.0]])
    with pytest.raises(ValueError, match="'log' o 'simple'"):
        panel_from_prices(prices, ["A"], method="raro")


def test_panel_exige_al_menos_dos_observaciones():
    """Con una sola observacion no existe varianza (ddof=1 divide por
    cero). El panel lo rechaza en la frontera en vez de propagar NaN."""
    prices = np.array([[100.0], [102.0]])
    with pytest.raises(ValueError, match="al menos 2 observaciones"):
        panel_from_prices(prices, ["A"])


def test_panel_from_pnl_admite_negativos_y_cero():
    """El PnL cruza cero constantemente. Esta es la puerta de entrada
    de las estrategias al mismo motor que analiza instrumentos."""
    pnl = np.array([[120.0, -80.0], [-45.0, 0.0], [0.0, 210.0]])

    panel = panel_from_pnl(pnl, ["MBR", "MeanRev"])

    assert panel.kind == "pnl"
    assert panel.T == 3
    assert panel.N == 2
