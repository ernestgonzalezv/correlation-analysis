from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import StringIO

import certifi
import pandas as pd

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MISSING_VALUE_TOKEN = "."
REQUEST_TIMEOUT_SECONDS = 20
DOWNLOAD_ATTEMPTS = 3
USER_AGENT = "correlation-analysis/0.1 (research)"


@dataclass(frozen=True)
class Series:
    name: str
    series_id: str
    invert: bool = False
    description: str = ""


FX_MAJORS: tuple[Series, ...] = (
    Series("EURUSD", "DEXUSEU", description="US dollars per euro"),
    Series("GBPUSD", "DEXUSUK", description="US dollars per pound"),
    Series("AUDUSD", "DEXUSAL", description="US dollars per Australian dollar"),
    Series("JPYUSD", "DEXJPUS", invert=True, description="inverted from yen per dollar"),
    Series("CADUSD", "DEXCAUS", invert=True, description="inverted from CAD per dollar"),
    Series("CHFUSD", "DEXSZUS", invert=True, description="inverted from CHF per dollar"),
)

CROSS_ASSET: tuple[Series, ...] = (
    Series("SP500", "SP500", description="S&P 500 index"),
    Series("NASDAQ", "NASDAQCOM", description="Nasdaq Composite index"),
    Series("BRENT", "DCOILBRENTEU", description="Brent crude, USD per barrel"),
    Series("VIX", "VIXCLS", description="CBOE volatility index"),
)

DEFAULT_UNIVERSE: tuple[Series, ...] = FX_MAJORS + CROSS_ASSET


def _download(url: str) -> str:
    context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None

    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS, context=context
            ) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise ValueError(f"FRED returned HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc

    raise ConnectionError(
        f"could not download {url} after {DOWNLOAD_ATTEMPTS} attempts "
        f"({last_error}). On a slow link, fetch the file separately and read it "
        "with parse_series."
    )


def parse_series(payload: str, series_id: str) -> pd.Series:
    frame = pd.read_csv(
        StringIO(payload),
        parse_dates=[0],
        index_col=0,
        na_values=[MISSING_VALUE_TOKEN],
    )

    if frame.shape[1] != 1:
        raise ValueError(
            f"expected a single value column for {series_id}; got {list(frame.columns)}"
        )

    series = frame.iloc[:, 0].astype(float)
    series.index.name = "date"

    if series.dropna().empty:
        raise ValueError(f"FRED returned no observations for {series_id}")

    return series


def series_url(
    series_id: str, start: str | None = None, end: str | None = None
) -> str:
    params = [f"id={series_id}"]
    if start is not None:
        params.append(f"cosd={start}")
    if end is not None:
        params.append(f"coed={end}")

    return f"{FRED_CSV_URL}?{'&'.join(params)}"


def fetch_series(
    series_id: str, start: str | None = None, end: str | None = None
) -> pd.Series:
    payload = _download(series_url(series_id, start=start, end=end))
    return parse_series(payload, series_id)


def fetch_frame(
    universe: tuple[Series, ...] = DEFAULT_UNIVERSE,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if not universe:
        raise ValueError("universe must contain at least one series")

    columns = {}

    for spec in universe:
        values = fetch_series(spec.series_id, start=start, end=end)
        columns[spec.name] = 1.0 / values if spec.invert else values

    frame = pd.DataFrame(columns)
    frame.index.name = "date"

    return frame.sort_index()


def describe(universe: tuple[Series, ...] = DEFAULT_UNIVERSE) -> str:
    width = max(len(spec.name) for spec in universe)
    lines = []

    for spec in universe:
        marker = " (inverted)" if spec.invert else ""
        lines.append(
            f"{spec.name:<{width}}  {spec.series_id:<14} {spec.description}{marker}"
        )

    return "\n".join(lines)
