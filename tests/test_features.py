"""Tests for the helper functions, on tiny hand-made frames (no data download)."""

import numpy as np
import pandas as pd
import pytest

from energy_eda import data, features as F


def make_frame(start="2020-01-06 00:00", hours=24 * 14, seed=0):
    """Two weeks of synthetic hourly data on a UTC index (2020-01-06 is a Monday)."""
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="timestamp_utc")
    rng = np.random.default_rng(seed)
    wind = rng.uniform(1_000, 40_000, hours)
    return pd.DataFrame({
        "price_eur_mwh": 60 - wind / 1_000 + rng.normal(0, 1, hours),
        "load_mw": np.full(hours, 60_000.0),
        "wind_generation_mw": wind,
        "solar_generation_mw": np.full(hours, 5_000.0),
    }, index=idx)


def test_residual_load_subtracts_wind_and_solar():
    df = make_frame(hours=3)
    rl = F.residual_load(df)
    expected = 60_000 - df["wind_generation_mw"] - 5_000
    pd.testing.assert_series_equal(rl, expected.rename("residual_load_mw"))


def test_residual_load_treats_missing_solar_as_zero():
    df = make_frame(hours=2)
    df.loc[df.index[0], "solar_generation_mw"] = np.nan
    rl = F.residual_load(df)
    assert rl.iloc[0] == pytest.approx(60_000 - df["wind_generation_mw"].iloc[0])


def test_hour_weekday_table_uses_berlin_local_time():
    # 2020-01-06 23:00 UTC is 00:00 on Tuesday in Berlin (UTC+1 in winter).
    idx = pd.DatetimeIndex(["2020-01-06 23:00"], tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": [42.0]}, index=idx)
    t = F.hour_weekday_table(df)
    assert t.shape == (24, 7)
    assert t.loc[0, "Tue"] == 42.0
    assert t.drop(columns="Tue").isna().all().all()


def test_hour_weekday_table_handles_summer_time():
    # In July Berlin is UTC+2: 17:00 UTC is the 19:00 local evening peak.
    idx = pd.DatetimeIndex(["2020-07-06 17:00"], tz="UTC")  # Monday
    t = F.hour_weekday_table(pd.DataFrame({"price_eur_mwh": [55.0]}, index=idx))
    assert t.loc[19, "Mon"] == 55.0


def test_wind_deciles_are_balanced_and_show_merit_order():
    df = make_frame()
    dec = F.wind_deciles(df)
    assert list(dec.index) == list(range(1, 11))
    assert dec["hours"].sum() == len(df)
    assert dec["hours"].max() - dec["hours"].min() <= 1
    # price was built to fall with wind, so the windiest decile must be cheapest
    assert dec["mean_price"].is_monotonic_decreasing


def test_negative_hours_by_month_counts_and_keeps_zero_months():
    idx = pd.date_range("2020-01-01", "2020-03-31 22:00", freq="h", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": 30.0}, index=idx)
    df.loc["2020-01-15 10:00":"2020-01-15 12:00", "price_eur_mwh"] = -5.0   # 3 hours
    df.loc["2020-03-01 12:00", "price_eur_mwh"] = -1.0                       # 1 hour
    neg = F.negative_hours_by_month(df)
    assert neg.to_dict() == {"2020-01": 3, "2020-02": 0, "2020-03": 1}


def test_negative_hours_by_month_drops_partial_trailing_month():
    # The last UTC hour below is already 1 April in Berlin; that 1-hour "month" must not appear.
    idx = pd.date_range("2020-03-01", "2020-03-31 23:00", freq="h", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": 30.0}, index=idx)
    assert list(F.negative_hours_by_month(df).index) == ["2020-03"]


def test_fill_short_gaps_only_fills_short_gaps():
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    s = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0, 10.0], index=idx)
    out = F.fill_short_gaps(s, limit=2)
    assert out.iloc[1] == pytest.approx(2.0)
    assert out.iloc[4:8].isna().sum() == 2  # a 4-hour gap is only partly filled (limit=2)


def test_sample_loads_with_expected_shape():
    df = data.load_sample()
    assert len(df) == 17_544
    assert list(df.columns) == data.COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_unique and df.index.is_monotonic_increasing
