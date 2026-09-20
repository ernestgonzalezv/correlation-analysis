import pytest

from correlation_analysis.data.fred import (
    CROSS_ASSET,
    DEFAULT_UNIVERSE,
    FX_MAJORS,
    describe,
    parse_series,
    series_url,
)

SAMPLE = "observation_date,DEXUSEU\n2024-01-02,1.1000\n2024-01-03,.\n2024-01-04,1.1200\n"


def test_url_carries_the_series_id():
    assert "id=DEXUSEU" in series_url("DEXUSEU")


def test_url_carries_the_date_window():
    url = series_url("DEXUSEU", start="2015-01-01", end="2020-12-31")
    assert "cosd=2015-01-01" in url
    assert "coed=2020-12-31" in url


def test_url_omits_absent_dates():
    url = series_url("DEXUSEU")
    assert "cosd=" not in url and "coed=" not in url


def test_parses_values_and_dates():
    series = parse_series(SAMPLE, "DEXUSEU")
    assert len(series) == 3
    assert series.iloc[0] == pytest.approx(1.10)


def test_missing_token_becomes_nan():
    assert int(parse_series(SAMPLE, "DEXUSEU").isna().sum()) == 1


def test_rejects_an_all_missing_series():
    payload = "observation_date,X\n2024-01-02,.\n2024-01-03,.\n"
    with pytest.raises(ValueError, match="no observations"):
        parse_series(payload, "X")


def test_rejects_a_multi_column_payload():
    payload = "observation_date,A,B\n2024-01-02,1.0,2.0\n"
    with pytest.raises(ValueError, match="single value column"):
        parse_series(payload, "AB")


def test_universe_names_are_unique():
    names = [spec.name for spec in DEFAULT_UNIVERSE]
    assert len(names) == len(set(names))


def test_default_universe_is_the_union_of_its_parts():
    assert DEFAULT_UNIVERSE == FX_MAJORS + CROSS_ASSET


def test_inverted_quotes_are_flagged():
    inverted = {spec.name for spec in FX_MAJORS if spec.invert}
    assert inverted == {"JPYUSD", "CADUSD", "CHFUSD"}


def test_describe_lists_every_series():
    text = describe()
    assert all(spec.series_id in text for spec in DEFAULT_UNIVERSE)
