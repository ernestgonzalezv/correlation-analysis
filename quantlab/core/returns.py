"""Conversion de precios a retornos.

Por que nunca se correlacionan precios directamente:

1. No son comparables entre si. EUR/USD vale ~1.08 y el oro ~2600.
2. Peor: dos series con tendencia dan correlacion alta SIEMPRE, aunque
   no tengan nada que ver, porque las dos suben. Es correlacion espuria.

Los retornos resuelven ambos problemas y ademas son aproximadamente
estacionarios, que es lo que asume toda la estadistica posterior.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from quantlab.core.panel import ReturnsPanel


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Retornos logaritmicos: r_t = ln(P_t / P_{t-1}).

    Se usan log-retornos y no cambios porcentuales simples porque los
    log-retornos son aditivos en el tiempo: el retorno de la semana es
    la suma de los retornos de los 5 dias. Eso simplifica agregacion,
    anualizacion y composicion.

    Devuelve una matriz con una fila menos que la entrada.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 2:
        raise ValueError(f"prices debe ser 2D (T, N); recibido ndim={prices.ndim}")

    if prices.shape[0] < 2:
        raise ValueError("se necesitan al menos 2 filas de precios")

    if not np.isfinite(prices).all():
        raise ValueError("prices contiene NaN o inf; limpialo antes de convertir")

    if (prices <= 0).any():
        n_malos = int((prices <= 0).sum())
        raise ValueError(
            f"prices contiene {n_malos} valores <= 0; el logaritmo no existe ahi. "
            "Para series que cruzan cero (PnL, spreads) usa diferencias simples."
        )

    return np.log(prices[1:] / prices[:-1])


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Retornos simples: r_t = P_t / P_{t-1} - 1."""
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 2:
        raise ValueError(f"prices debe ser 2D (T, N); recibido ndim={prices.ndim}")

    if prices.shape[0] < 2:
        raise ValueError("se necesitan al menos 2 filas de precios")

    if (prices[:-1] == 0).any():
        raise ValueError("hay precios en cero; no se puede dividir")

    return prices[1:] / prices[:-1] - 1.0


def panel_from_prices(
    prices: np.ndarray,
    names: Sequence[str],
    freq: str = "1d",
    index: np.ndarray | None = None,
    method: str = "log",
) -> ReturnsPanel:
    """Construye un ReturnsPanel a partir de una matriz de precios.

    El index se recorta en su primer elemento para quedar alineado con
    los retornos, que tienen una observacion menos que los precios.
    """
    if method == "log":
        values = log_returns(prices)
    elif method == "simple":
        values = simple_returns(prices)
    else:
        raise ValueError(f"method debe ser 'log' o 'simple'; dado '{method}'")

    index_alineado = None if index is None else np.asarray(index)[1:]

    return ReturnsPanel(
        values=values,
        names=tuple(names),
        freq=freq,
        kind="returns",
        index=index_alineado,
    )


def panel_from_pnl(
    pnl: np.ndarray,
    names: Sequence[str],
    freq: str = "1d",
    index: np.ndarray | None = None,
) -> ReturnsPanel:
    """Construye un ReturnsPanel a partir de PnL diario de estrategias.

    El PnL ya es un cambio, no un nivel: no se convierte, se envuelve.
    Esta es la puerta por la que entran las estrategias al mismo motor
    que analiza instrumentos.
    """
    return ReturnsPanel(
        values=np.asarray(pnl, dtype=float),
        names=tuple(names),
        freq=freq,
        kind="pnl",
        index=None if index is None else np.asarray(index),
    )
