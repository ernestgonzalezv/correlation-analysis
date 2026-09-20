"""Formato canonico: ReturnsPanel.

Es la unica estructura que atraviesa todo el sistema.

    PRODUCTORES                                    CONSUMIDORES
    yfinance, MT5, CSV,  ----> ReturnsPanel ---->  covarianza, PCA,
    PnL de backtest,                               filtro de ruido,
    PnL en vivo, sintetico                         metricas de riesgo

Al motor no le importa si una columna es EUR/USD o la estrategia MBR.
Son series de retornos. Esa indiferencia es toda la escalabilidad
del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

# Periodos por año segun timeframe.
#
# Base 252 dias habiles. Forex opera 24h durante 5 dias, por eso las
# intradiarias multiplican por las horas del dia completo.
#
# Este diccionario existe para que nadie tenga que recordar el factor
# de anualizacion a mano: anualizar un Sharpe con sqrt() equivocado es
# el bug mas silencioso y mas comun del quant retail.
PERIODOS_POR_ANIO: dict[str, float] = {
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

TIPOS_VALIDOS = ("returns", "pnl")


@dataclass(frozen=True)
class ReturnsPanel:
    """Panel de series temporales alineadas.

    Parameters
    ----------
    values
        Matriz (T, N). Filas = observaciones en el tiempo,
        columnas = series (instrumentos o estrategias).
    names
        Nombre de cada columna. Largo N.
    freq
        Timeframe de las observaciones. Clave de PERIODOS_POR_ANIO.
    kind
        "returns" para retornos porcentuales/log, "pnl" para
        ganancias en moneda. La matematica de correlacion es la
        misma; la distincion evita mezclar unidades por accidente.
    index
        Marcas de tiempo opcionales. Largo T.
    """

    values: np.ndarray
    names: tuple[str, ...]
    freq: str = "1d"
    kind: str = "returns"
    index: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Validacion
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", tuple(self.names))

        if values.ndim != 2:
            raise ValueError(
                f"values debe ser una matriz 2D (T, N); recibido ndim={values.ndim}"
            )

        n_filas, n_cols = values.shape

        if n_filas < 2:
            raise ValueError(f"se necesitan al menos 2 observaciones; hay {n_filas}")

        if len(self.names) != n_cols:
            raise ValueError(
                f"names tiene {len(self.names)} nombres pero values tiene "
                f"{n_cols} columnas"
            )

        if len(set(self.names)) != len(self.names):
            raise ValueError(f"hay nombres duplicados en names: {self.names}")

        if self.freq not in PERIODOS_POR_ANIO:
            raise ValueError(
                f"freq '{self.freq}' desconocida; "
                f"validas: {sorted(PERIODOS_POR_ANIO)}"
            )

        if self.kind not in TIPOS_VALIDOS:
            raise ValueError(f"kind debe ser uno de {TIPOS_VALIDOS}; dado '{self.kind}'")

        if not np.isfinite(values).all():
            n_malos = int((~np.isfinite(values)).sum())
            raise ValueError(
                f"values contiene {n_malos} valores no finitos (NaN o inf). "
                "Limpia los datos en la capa de adaptadores, no aqui."
            )

        if self.index is not None:
            index = np.asarray(self.index)
            object.__setattr__(self, "index", index)
            if len(index) != n_filas:
                raise ValueError(
                    f"index tiene largo {len(index)} pero values tiene "
                    f"{n_filas} filas"
                )

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def T(self) -> int:
        """Numero de observaciones."""
        return self.values.shape[0]

    @property
    def N(self) -> int:
        """Numero de series."""
        return self.values.shape[1]

    @property
    def periodos_por_anio(self) -> float:
        """Factor de anualizacion correspondiente a la frecuencia."""
        return PERIODOS_POR_ANIO[self.freq]

    @property
    def ratio_observaciones(self) -> float:
        """T/N. Cuantas observaciones hay por cada serie.

        Regla practica: por debajo de 10 la matriz de covarianza es
        mayormente ruido. Ver core.noise para el criterio formal.
        """
        return self.T / self.N

    # ------------------------------------------------------------------
    # Operaciones
    # ------------------------------------------------------------------

    def select(self, names: Sequence[str]) -> "ReturnsPanel":
        """Devuelve un panel nuevo con solo las columnas pedidas."""
        faltantes = [n for n in names if n not in self.names]
        if faltantes:
            raise KeyError(f"columnas no encontradas: {faltantes}")

        idx = [self.names.index(n) for n in names]
        return replace(self, values=self.values[:, idx], names=tuple(names))

    def slice_rows(self, mask: np.ndarray) -> "ReturnsPanel":
        """Devuelve un panel nuevo con las filas donde mask es True."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.T,):
            raise ValueError(f"mask debe tener forma ({self.T},); dada {mask.shape}")

        nuevo_index = None if self.index is None else self.index[mask]
        return replace(self, values=self.values[mask], index=nuevo_index)

    def column(self, name: str) -> np.ndarray:
        """Serie de una columna por nombre."""
        if name not in self.names:
            raise KeyError(f"columna '{name}' no existe; hay {self.names}")
        return self.values[:, self.names.index(name)]

    def __repr__(self) -> str:
        return (
            f"ReturnsPanel(T={self.T}, N={self.N}, freq='{self.freq}', "
            f"kind='{self.kind}', names={self.names})"
        )
