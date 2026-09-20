"""Tests del formato canonico.

El panel es la pieza de la que cuelga todo el sistema. Si acepta datos
invalidos en silencio, todos los numeros de aguas abajo son basura sin
que nadie se entere. Por eso la mayoria de estos tests verifican que
RECHAZA cosas, no que las acepta.
"""

import numpy as np
import pytest

from quantlab.core.panel import PERIODOS_POR_ANIO, ReturnsPanel


@pytest.fixture
def panel_simple():
    rng = np.random.default_rng(42)
    return ReturnsPanel(
        values=rng.normal(size=(100, 3)),
        names=("EURUSD", "GBPUSD", "GOLD"),
        freq="1d",
    )


# ----------------------------------------------------------------------
# Construccion valida
# ----------------------------------------------------------------------


def test_construccion_basica(panel_simple):
    assert panel_simple.T == 100
    assert panel_simple.N == 3
    assert panel_simple.names == ("EURUSD", "GBPUSD", "GOLD")
    assert panel_simple.kind == "returns"


def test_values_se_convierte_a_float():
    panel = ReturnsPanel(values=[[1, 2], [3, 4]], names=("A", "B"))
    assert panel.values.dtype == np.float64


def test_panel_es_inmutable(panel_simple):
    with pytest.raises(Exception):
        panel_simple.freq = "1h"


# ----------------------------------------------------------------------
# Validacion: lo que tiene que rechazar
# ----------------------------------------------------------------------


def test_rechaza_array_1d():
    with pytest.raises(ValueError, match="2D"):
        ReturnsPanel(values=np.zeros(10), names=("A",))


def test_rechaza_menos_de_dos_observaciones():
    with pytest.raises(ValueError, match="al menos 2 observaciones"):
        ReturnsPanel(values=np.zeros((1, 2)), names=("A", "B"))


def test_rechaza_nombres_desalineados():
    with pytest.raises(ValueError, match="nombres pero values"):
        ReturnsPanel(values=np.zeros((10, 3)), names=("A", "B"))


def test_rechaza_nombres_duplicados():
    with pytest.raises(ValueError, match="duplicados"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "A"))


def test_rechaza_freq_desconocida():
    with pytest.raises(ValueError, match="desconocida"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "B"), freq="3s")


def test_rechaza_kind_invalido():
    with pytest.raises(ValueError, match="kind debe ser"):
        ReturnsPanel(values=np.zeros((10, 2)), names=("A", "B"), kind="precios")


def test_rechaza_nan():
    values = np.zeros((10, 2))
    values[3, 1] = np.nan
    with pytest.raises(ValueError, match="no finitos"):
        ReturnsPanel(values=values, names=("A", "B"))


def test_rechaza_inf():
    values = np.zeros((10, 2))
    values[0, 0] = np.inf
    with pytest.raises(ValueError, match="no finitos"):
        ReturnsPanel(values=values, names=("A", "B"))


def test_rechaza_index_de_largo_incorrecto():
    with pytest.raises(ValueError, match="index tiene largo"):
        ReturnsPanel(
            values=np.zeros((10, 2)), names=("A", "B"), index=np.arange(5)
        )


# ----------------------------------------------------------------------
# Anualizacion
# ----------------------------------------------------------------------


def test_periodos_por_anio_diario(panel_simple):
    assert panel_simple.periodos_por_anio == 252


def test_periodos_por_anio_m15():
    panel = ReturnsPanel(values=np.zeros((10, 1)), names=("A",), freq="15m")
    assert panel.periodos_por_anio == 252 * 96


def test_factores_de_anualizacion_son_monotonos():
    """Mas granular -> mas periodos por año. Un orden roto aqui produce
    Sharpes inflados sin lanzar ninguna excepcion."""
    orden = ["1mo", "1w", "1d", "4h", "1h", "30m", "15m", "5m", "1m"]
    factores = [PERIODOS_POR_ANIO[f] for f in orden]
    assert factores == sorted(factores)


def test_ratio_observaciones(panel_simple):
    assert panel_simple.ratio_observaciones == pytest.approx(100 / 3)


# ----------------------------------------------------------------------
# Operaciones
# ----------------------------------------------------------------------


def test_select_reordena_y_filtra(panel_simple):
    sub = panel_simple.select(["GOLD", "EURUSD"])
    assert sub.names == ("GOLD", "EURUSD")
    assert sub.N == 2
    assert np.allclose(sub.values[:, 0], panel_simple.column("GOLD"))
    assert np.allclose(sub.values[:, 1], panel_simple.column("EURUSD"))


def test_select_preserva_metadata(panel_simple):
    sub = panel_simple.select(["GOLD"])
    assert sub.freq == panel_simple.freq
    assert sub.kind == panel_simple.kind


def test_select_rechaza_columna_inexistente(panel_simple):
    with pytest.raises(KeyError, match="no encontradas"):
        panel_simple.select(["EURUSD", "BITCOIN"])


def test_slice_rows(panel_simple):
    mask = np.zeros(100, dtype=bool)
    mask[:10] = True
    sub = panel_simple.slice_rows(mask)
    assert sub.T == 10
    assert sub.N == 3
    assert np.allclose(sub.values, panel_simple.values[:10])


def test_slice_rows_recorta_index():
    index = np.arange(10)
    panel = ReturnsPanel(
        values=np.zeros((10, 2)), names=("A", "B"), index=index
    )
    mask = np.array([True, False] * 5)
    sub = panel.slice_rows(mask)
    assert sub.T == 5
    assert np.array_equal(sub.index, np.array([0, 2, 4, 6, 8]))


def test_slice_rows_rechaza_mask_mal_formada(panel_simple):
    with pytest.raises(ValueError, match="mask debe tener forma"):
        panel_simple.slice_rows(np.ones(5, dtype=bool))


def test_column_por_nombre(panel_simple):
    assert np.allclose(panel_simple.column("GBPUSD"), panel_simple.values[:, 1])


def test_column_rechaza_nombre_inexistente(panel_simple):
    with pytest.raises(KeyError, match="no existe"):
        panel_simple.column("PLATA")


def test_no_muta_el_panel_original(panel_simple):
    original = panel_simple.values.copy()
    panel_simple.select(["GOLD"])
    panel_simple.slice_rows(np.ones(100, dtype=bool))
    assert np.allclose(panel_simple.values, original)
