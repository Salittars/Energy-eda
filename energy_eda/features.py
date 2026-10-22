"""Small, tested helpers that turn the hourly table into analysis features."""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Europe/Berlin"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def residual_load(df: pd.DataFrame) -> pd.Series:
    """Load minus wind minus solar (MW): the demand that dispatchable plants must cover.

    If solar is missing for an hour, only wind is subtracted for that hour.
    """
    solar = df["solar_generation_mw"].fillna(0) if "solar_generation_mw" in df else 0
    return (df["load_mw"] - df["wind_generation_mw"] - solar).rename("residual_load_mw")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add residual load and local-time calendar columns (Europe/Berlin)."""
    out = df.copy()
    out["residual_load_mw"] = residual_load(out)
    local = out.index.tz_convert(LOCAL_TZ)
    out["hour_local"] = local.hour
    out["weekday_local"] = local.dayofweek
    out["month"] = local.strftime("%Y-%m")
    return out


def hour_weekday_table(df: pd.DataFrame, value: str = "price_eur_mwh") -> pd.DataFrame:
    """Mean of ``value`` by local hour (rows 0-23) and weekday (columns Mon-Sun)."""
    local = df.index.tz_convert(LOCAL_TZ)
    table = (
        df.assign(hour=local.hour, weekday=local.dayofweek)
        .pivot_table(index="hour", columns="weekday", values=value, aggfunc="mean")
        .reindex(index=range(24), columns=range(7))
    )
    table.columns = WEEKDAYS
    return table


def wind_deciles(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Price statistics per wind-generation decile (1 = calmest, n = windiest)."""
    d = df.dropna(subset=["wind_generation_mw", "price_eur_mwh"]).copy()
    d["wind_decile"] = pd.qcut(d["wind_generation_mw"], n, labels=range(1, n + 1)).astype(int)
    return d.groupby("wind_decile").agg(
        wind_min_gw=("wind_generation_mw", lambda s: s.min() / 1000),
        wind_max_gw=("wind_generation_mw", lambda s: s.max() / 1000),
        mean_price=("price_eur_mwh", "mean"),
        median_price=("price_eur_mwh", "median"),
        negative_hours=("price_eur_mwh", lambda s: int((s < 0).sum())),
        hours=("price_eur_mwh", "size"),
    )


def negative_hours_by_month(df: pd.DataFrame, min_hours: int = 24 * 27) -> pd.Series:
    """Count of hours with a negative day-ahead price per calendar month (local time).

    Months with no negative hours are kept as 0 so the series has no holes.
    Months covered by fewer than ``min_hours`` hours are dropped: the last two
    UTC hours of 30 Sep 2020 are already 1 Oct in Berlin, and without this rule
    they create a fake "October 2020" month with a 0 count.
    """
    local = pd.Index(df.index.tz_convert(LOCAL_TZ).strftime("%Y-%m"), name="month")
    neg = (df["price_eur_mwh"] < 0).groupby(local).sum().astype(int)
    hours = df["price_eur_mwh"].groupby(local).size()
    full = pd.period_range(local.min(), local.max(), freq="M").strftime("%Y-%m")
    neg = neg.reindex(full, fill_value=0)
    keep = hours.reindex(full, fill_value=0) >= min_hours
    out = neg[keep].rename("negative_hours")
    out.index.name = "month"
    return out


def fill_short_gaps(s: pd.Series, limit: int = 3) -> pd.Series:
    """Time-interpolate at most ``limit`` consecutive missing hours (ACF and STL need no NaNs).

    Longer gaps are only partly filled and the rest stays NaN, so they remain visible.
    """
    return s.interpolate(method="time", limit=limit, limit_area="inside")
