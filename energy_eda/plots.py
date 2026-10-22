"""One function per chart. Each returns (figure, takeaway, numbers).

The takeaway is a one-sentence finding built from the data, so the numbers in
the captions always match the figure that was just drawn.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

from . import features as F  # noqa: E402

PRICE = "price_eur_mwh"
PRICE_LABEL = "Day-ahead price (EUR/MWh)"
SOURCE_NOTE = "Data: OPSD / ENTSO-E (DE-LU day-ahead, DE load, wind, solar), DWD weather. Oct 2018 - Sep 2020."
BLUE, ORANGE, GREEN, RED, GREY = "#1f4e79", "#e07b00", "#2e8b57", "#c0392b", "#9aa5b1"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def _finish(fig, title: str, subtitle: str) -> None:
    """Title + subtitle in a fixed 0.85-inch band above the axes, source note below."""
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.25 / h, 1, 1 - 0.85 / h))
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(0.01, 1 - 0.48 / h, subtitle, ha="left", va="top", fontsize=10, color="#444")
    fig.text(0.99, 0.02 / h, SOURCE_NOTE, ha="right", va="bottom", fontsize=7.5, color="#777")


# 1 -----------------------------------------------------------------------
def price_timeseries(df: pd.DataFrame):
    p = df[PRICE]
    r7 = p.rolling("7D", min_periods=24 * 5).mean()
    r30 = p.rolling("30D", min_periods=24 * 20).mean()
    y1 = p.loc["2018-10-01":"2019-09-30"].mean()
    y2 = p.loc["2019-10-01":"2020-09-30"].mean()
    low_day = r30.loc["2019-10-31":].idxmin()
    nums = {
        "mean_y1": y1, "mean_y2": y2, "change_pct": (y2 / y1 - 1) * 100,
        "r30_min": r30.loc[low_day], "r30_min_date": low_day.strftime("%Y-%m-%d"),
        "min_price": p.min(), "min_time": p.idxmin().strftime("%Y-%m-%d %H:%M UTC"),
        "max_price": p.max(), "max_time": p.idxmax().strftime("%Y-%m-%d %H:%M UTC"),
    }

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(p.index, p, color=GREY, lw=0.35, alpha=0.7, label="Hourly price")
    ax.plot(r7.index, r7, color=BLUE, lw=1.4, label="7-day rolling mean")
    ax.plot(r30.index, r30, color=ORANGE, lw=2.4, label="30-day rolling mean")
    ax.axhline(0, color="black", lw=0.8)
    for start, end, m in [("2018-10-01", "2019-09-30", y1), ("2019-10-01", "2020-09-30", y2)]:
        ax.hlines(m, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                  colors=RED, linestyles="--", lw=1.2)
        ax.text(pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=150), -72,
                f"Oct-Sep mean: {m:.1f} EUR/MWh", color=RED, fontsize=9.5, fontweight="bold",
                ha="center", bbox=dict(facecolor="white", edgecolor=RED, boxstyle="round,pad=0.25"))
    ax.annotate(f"{nums['min_price']:.0f} EUR/MWh\n{p.idxmin():%d %b %Y}",
                xy=(p.idxmin(), p.min()), xytext=(30, 2), textcoords="offset points",
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#555"))
    ax.set_ylabel(PRICE_LABEL)
    ax.set_ylim(-100, 210)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", ncol=3, frameon=False)
    takeaway = (
        f"The average day-ahead price fell from {y1:.1f} EUR/MWh (Oct 2018-Sep 2019) to "
        f"{y2:.1f} EUR/MWh (Oct 2019-Sep 2020), a {abs(nums['change_pct']):.0f}% drop, with the "
        f"30-day mean bottoming at {nums['r30_min']:.1f} EUR/MWh around {low_day:%d %b %Y}."
    )
    _finish(fig, "German day-ahead power price, hourly with rolling means",
            "Rolling means strip out the hour-to-hour noise and show the regime shift in spring 2020")
    return fig, takeaway, nums


# 2 -----------------------------------------------------------------------
def seasonality_heatmap(df: pd.DataFrame):
    t = F.hour_weekday_table(df)
    wk = t[F.WEEKDAYS[:5]].stack()
    peak_h, peak_d = wk.idxmax()
    low_h, low_d = t.stack().idxmin()
    nums = {
        "weekday_peak": wk.max(), "weekday_peak_hour": int(peak_h), "weekday_peak_day": peak_d,
        "low": t.stack().min(), "low_hour": int(low_h), "low_day": low_d,
        "weekday_mean": t[F.WEEKDAYS[:5]].values.mean(), "sunday_mean": t["Sun"].mean(),
    }
    fig, ax = plt.subplots(figsize=(8, 8.5))
    sns.heatmap(t, cmap="rocket_r", annot=True, fmt=".0f", annot_kws={"size": 7.5},
                cbar_kws={"label": "Mean price (EUR/MWh)", "shrink": 0.7}, linewidths=0.3, ax=ax)
    ax.grid(False)
    ax.set_ylabel("Hour of day (Berlin local time)")
    ax.set_xlabel("")
    ax.set_yticklabels([f"{h:02d}:00" for h in range(24)], rotation=0)
    takeaway = (
        f"Prices follow the working week: the highest average is {nums['weekday_peak']:.1f} EUR/MWh "
        f"on {peak_d} at {peak_h:02d}:00, the lowest {nums['low']:.1f} EUR/MWh on {low_d} at "
        f"{low_h:02d}:00 when demand is low and solar output peaks; Sundays average "
        f"{nums['sunday_mean']:.1f} vs {nums['weekday_mean']:.1f} EUR/MWh on weekdays."
    )
    _finish(fig, "Daily and weekly price seasonality",
            "Mean day-ahead price by hour (local time) and weekday")
    return fig, takeaway, nums


# 3 -----------------------------------------------------------------------
def price_vs_wind_residual(df: pd.DataFrame):
    d = df.dropna(subset=[PRICE]).copy()
    d["wind_gw"] = d["wind_generation_mw"] / 1000
    d["rl_gw"] = d["residual_load_mw"] / 1000
    r_wind = d[PRICE].corr(d["wind_gw"])
    r_rl = d[PRICE].corr(d["rl_gw"])
    slope, intercept = np.polyfit(d["rl_gw"], d[PRICE], 1)
    nums = {"r_wind": r_wind, "r_residual": r_rl, "r2_residual": r_rl ** 2,
            "slope_eur_per_gw": slope, "r2_wind": r_wind ** 2}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    sample = d.sample(frac=1.0, random_state=0)
    sc = axes[0].scatter(sample["wind_gw"], sample[PRICE], c=sample["rl_gw"], cmap="viridis",
                         s=3, alpha=0.45, rasterized=True)
    cb = fig.colorbar(sc, ax=axes[0], pad=0.01)
    cb.set_label("Residual load (GW)")
    cb.solids.set_alpha(1)
    axes[0].set_xlabel("Wind generation (GW)")
    axes[0].set_ylabel(PRICE_LABEL)
    axes[0].set_title(f"Price vs wind  (r = {r_wind:.2f})", loc="left", fontsize=11)

    hb = axes[1].hexbin(d["rl_gw"], d[PRICE], gridsize=60, cmap="Blues", mincnt=1, bins="log")
    xs = np.linspace(d["rl_gw"].min(), d["rl_gw"].max(), 50)
    axes[1].plot(xs, intercept + slope * xs, color=RED, lw=2,
                 label=f"OLS fit: {slope:.2f} EUR/MWh per GW, R² = {r_rl**2:.2f}")
    fig.colorbar(hb, ax=axes[1], pad=0.01).set_label("Hours (log scale)")
    axes[1].set_xlabel("Residual load = load - wind - solar (GW)")
    axes[1].set_title(f"Price vs residual load  (r = {r_rl:.2f})", loc="left", fontsize=11)
    axes[1].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylim(-100, 210)
    takeaway = (
        f"Wind alone explains only {r_wind**2:.0%} of price variance (r = {r_wind:.2f}), but residual "
        f"load explains {r_rl**2:.0%} (r = {r_rl:.2f}); each extra GW of residual load adds about "
        f"{slope:.2f} EUR/MWh."
    )
    _finish(fig, "What drives the price: wind vs residual load",
            "Colour shows that the same wind output gives very different prices depending on demand left for other plants")
    return fig, takeaway, nums


# 4 -----------------------------------------------------------------------
def merit_order_deciles(df: pd.DataFrame):
    dec = F.wind_deciles(df)
    d = df.dropna(subset=[PRICE, "wind_generation_mw"]).copy()
    d["wind_decile"] = pd.qcut(d["wind_generation_mw"], 10, labels=range(1, 11)).astype(int)
    nums = {
        "calm_mean": dec.loc[1, "mean_price"], "windy_mean": dec.loc[10, "mean_price"],
        "calm_range": (dec.loc[1, "wind_min_gw"], dec.loc[1, "wind_max_gw"]),
        "windy_range": (dec.loc[10, "wind_min_gw"], dec.loc[10, "wind_max_gw"]),
        "windy_negative_hours": int(dec.loc[10, "negative_hours"]),
        "total_negative_hours": int(dec["negative_hours"].sum()),
    }
    nums["drop_pct"] = (1 - nums["windy_mean"] / nums["calm_mean"]) * 100

    fig, ax = plt.subplots(figsize=(11, 5.2))
    palette = sns.color_palette("crest", 10)
    sns.boxplot(data=d, x="wind_decile", y=PRICE, hue="wind_decile", palette=palette, legend=False,
                showfliers=False, width=0.6, ax=ax)
    ax.plot(range(10), dec["mean_price"].values, "o-", color=RED, lw=2, label="Mean price", zorder=5)
    for i, (m, lo, hi) in enumerate(zip(dec["mean_price"], dec["wind_min_gw"], dec["wind_max_gw"])):
        ax.text(i + 0.33, m, f"{m:.1f}", ha="left", va="center", color=RED, fontsize=8.5, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9), zorder=6)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"D{i}\n{lo:.0f}-{hi:.0f} GW" for i, lo, hi in
                        zip(dec.index, dec["wind_min_gw"], dec["wind_max_gw"])], fontsize=8.5)
    ax.set_xlabel("Wind generation decile (D1 = calmest 10% of hours, D10 = windiest)")
    ax.set_ylabel(PRICE_LABEL)
    ax.axhline(0, color="black", lw=0.7)
    ax.legend(loc="upper right", frameon=False)
    takeaway = (
        f"The merit-order effect: in the windiest 10% of hours ({nums['windy_range'][0]:.1f}-"
        f"{nums['windy_range'][1]:.1f} GW) the mean price is {nums['windy_mean']:.1f} EUR/MWh vs "
        f"{nums['calm_mean']:.1f} in the calmest 10%, {nums['drop_pct']:.0f}% lower, and that decile "
        f"holds {nums['windy_negative_hours']} of the {nums['total_negative_hours']} negative-price hours."
    )
    _finish(fig, "Merit-order effect: price by wind decile",
            "Boxes show the middle 50% of hours (whiskers 1.5 IQR, outliers hidden); red line is the mean")
    return fig, takeaway, nums


# 5 -----------------------------------------------------------------------
def negative_prices_by_month(df: pd.DataFrame):
    neg = F.negative_hours_by_month(df)
    total = int((df[PRICE] < 0).sum())
    y1 = int((df.loc["2018-10-01":"2019-09-30", PRICE] < 0).sum())
    y2 = int((df.loc["2019-10-01":"2020-09-30", PRICE] < 0).sum())
    top = neg.idxmax()
    negs = df.loc[df[PRICE] < 0]
    local = negs.index.tz_convert(F.LOCAL_TZ)
    weekend_share = (local.dayofweek >= 5).mean()
    nums = {"total": total, "y1": y1, "y2": y2, "top_month": top, "top_count": int(neg.max()),
            "weekend_share": weekend_share, "mean_negative_price": negs[PRICE].mean()}

    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = [RED if m == top else BLUE for m in neg.index]
    ax.bar(range(len(neg)), neg.values, color=colors)
    for i, v in enumerate(neg.values):
        if v:
            ax.text(i, v + 1, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(len(neg)))
    ax.set_xticklabels([pd.Period(m).strftime("%b\n%Y") if pd.Period(m).month in (1, 10) or i == 0
                        else pd.Period(m).strftime("%b") for i, m in enumerate(neg.index)], fontsize=8.5)
    ax.set_ylabel("Hours with price < 0")
    ax.set_xlabel("Month (Berlin local time)")
    takeaway = (
        f"{total} hours had negative prices ({y1} in the first year, {y2} in the second); the worst "
        f"month was {pd.Timestamp(top + "-01"):%B %Y} with {int(neg.max())} hours, and "
        f"{weekend_share:.0%} of all negative hours fell on weekends."
    )
    _finish(fig, "Negative-price hours per month",
            "Negative prices happen when inflexible and renewable output exceeds demand plus exports")
    return fig, takeaway, nums


# 6 -----------------------------------------------------------------------
def load_wind_distributions(df: pd.DataFrame):
    load = df["load_mw"] / 1000
    wind = df["wind_generation_mw"] / 1000
    rl = df["residual_load_mw"] / 1000
    nums = {
        "load_mean": load.mean(), "load_median": load.median(), "load_min": load.min(), "load_max": load.max(),
        "wind_mean": wind.mean(), "wind_median": wind.median(), "wind_skew": wind.skew(),
        "wind_p10": wind.quantile(0.1), "wind_p90": wind.quantile(0.9),
        "load_cv": load.std() / load.mean(), "wind_cv": wind.std() / wind.mean(),
        "rl_min": rl.min(), "rl_max": rl.max(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = [(load, "Load (GW)", BLUE), (wind, "Wind generation (GW)", GREEN), (rl, "Residual load (GW)", ORANGE)]
    for ax, (s, label, c) in zip(axes, specs):
        sns.histplot(s, bins=60, kde=True, color=c, stat="density", alpha=0.45, ax=ax, edgecolor=None)
        ax.axvline(s.mean(), color="black", ls="--", lw=1, label=f"mean {s.mean():.1f}")
        ax.axvline(s.median(), color=RED, ls=":", lw=1.5, label=f"median {s.median():.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.legend(frameon=False, fontsize=8.5)
    takeaway = (
        f"Load stays inside a narrow band ({nums['load_min']:.1f}-{nums['load_max']:.1f} GW, coefficient of "
        f"variation {nums['load_cv']:.2f}) with two humps for nights/weekends and working hours, while wind "
        f"is right-skewed (skew {nums['wind_skew']:.2f}, median {nums['wind_median']:.1f} vs mean "
        f"{nums['wind_mean']:.1f} GW, CV {nums['wind_cv']:.2f}) and runs from {nums['wind_p10']:.1f} to "
        f"{nums['wind_p90']:.1f} GW between its 10th and 90th percentiles."
    )
    _finish(fig, "Distributions of load, wind and residual load (hourly)",
            "Wind is the volatile input: its variability, not demand, moves residual load most")
    return fig, takeaway, nums


# 7 -----------------------------------------------------------------------
def price_acf_pacf(df: pd.DataFrame):
    p = F.fill_short_gaps(df[PRICE]).dropna()
    a = acf(p, nlags=24 * 8, fft=True)
    trough = int(np.argmin(a[1:25])) + 1
    nums = {"acf_1": a[1], "acf_trough_lag": trough, "acf_trough": a[trough], "acf_24": a[24],
            "acf_168": a[168]}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2))
    plot_acf(p, lags=24 * 8, ax=axes[0], alpha=0.05, title="", marker=".", markersize=3,
             vlines_kwargs={"colors": BLUE, "lw": 0.8}, color=BLUE)
    plot_pacf(p, lags=72, ax=axes[1], alpha=0.05, title="", method="ywm", marker=".", markersize=4,
              vlines_kwargs={"colors": BLUE, "lw": 0.8}, color=BLUE)
    for lag in (24, 48, 72, 96, 120, 144, 168, 192):
        axes[0].axvline(lag, color=ORANGE, lw=0.8, ls=":", zorder=0)
    axes[0].annotate(f"lag 24 h: {a[24]:.2f}", xy=(24, a[24]), xytext=(30, 0.95), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#555"))
    axes[0].annotate(f"lag 168 h (1 week): {a[168]:.2f}", xy=(168, a[168]), xytext=(120, 0.95),
                     fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    axes[0].set_title("Autocorrelation (ACF), lags up to 8 days", loc="left", fontsize=11)
    axes[1].set_title("Partial autocorrelation (PACF), lags up to 72 h", loc="left", fontsize=11)
    axes[1].set_xlabel("Lag (hours)")
    for ax in axes:
        ax.set_ylim(-0.5, 1.05)
    takeaway = (
        f"Price is highly persistent: correlation with the previous hour is {a[1]:.2f}, it falls to "
        f"{a[trough]:.2f} at {trough} h, climbs back to {a[24]:.2f} at 24 h and peaks again at {a[168]:.2f} "
        f"after one week, "
        f"so yesterday's and last week's same hour are natural forecasting features."
    )
    _finish(fig, "How much the price remembers: ACF and PACF of hourly price",
            "Shaded band = 95% confidence interval for zero correlation; dotted lines mark whole days")
    return fig, takeaway, nums


# 8 -----------------------------------------------------------------------
def stl_decomposition(df: pd.DataFrame):
    daily = F.fill_short_gaps(df[PRICE]).tz_convert(F.LOCAL_TZ).resample("D").mean().dropna()
    res = STL(daily, period=7, robust=True).fit()
    seas = res.seasonal
    by_dow = seas.groupby(seas.index.dayofweek).mean()
    var_total = daily.var()
    nums = {
        "trend_start": res.trend.iloc[0], "trend_end": res.trend.iloc[-1],
        "trend_min": res.trend.min(), "trend_min_date": res.trend.idxmin().strftime("%Y-%m-%d"),
        "sunday_effect": by_dow.loc[6], "saturday_effect": by_dow.loc[5],
        "weekday_effect": by_dow.loc[0:4].mean(),
        "resid_share": res.resid.var() / var_total, "seasonal_share": seas.var() / var_total,
        "trend_share": res.trend.var() / var_total,
    }
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    parts = [(daily, "Daily mean price", GREY), (res.trend, "Trend", ORANGE),
             (seas, "Weekly seasonal", BLUE), (res.resid, "Residual", "#555")]
    for ax, (s, name, c) in zip(axes, parts):
        if name == "Residual":
            ax.scatter(s.index, s, s=4, color=c)
            ax.axhline(0, color="black", lw=0.7)
        else:
            ax.plot(s.index, s, color=c, lw=1.1 if name != "Trend" else 2.2)
        ax.set_ylabel(f"{name}\n(EUR/MWh)", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    takeaway = (
        f"STL splits the daily price into a trend that falls from {nums['trend_start']:.1f} to a low of "
        f"{nums['trend_min']:.1f} EUR/MWh ({pd.Timestamp(nums['trend_min_date']):%b %Y}), a weekly cycle "
        f"worth {nums['sunday_effect']:.1f} EUR/MWh on Sundays and {nums['weekday_effect']:+.1f} on weekdays, "
        f"and a residual that still carries {nums['resid_share']:.0%} of the variance (weather and fuel shocks)."
    )
    _finish(fig, "STL decomposition of the daily average price (weekly period)",
            "Robust Seasonal-Trend decomposition using LOESS, period = 7 days")
    return fig, takeaway, nums


# 9 -----------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame):
    cols = {
        PRICE: "Price", "load_mw": "Load", "wind_generation_mw": "Wind gen.",
        "solar_generation_mw": "Solar gen.", "residual_load_mw": "Residual load",
        "wind_speed_100m_ms_mean": "Wind speed 100 m", "temperature_c_mean": "Temperature",
    }
    present = [c for c in cols if df[c].notna().any()]
    corr = df[present].rename(columns=cols).corr(method="pearson")
    nums = {f"price_vs_{cols[c]}": corr.loc["Price", cols[c]] for c in present if c != PRICE}
    if "wind_speed_100m_ms_mean" in present:
        nums["windspeed_vs_windgen"] = corr.loc["Wind speed 100 m", "Wind gen."]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    takeaway = (
        f"Residual load is the single best linear correlate of price (r = {nums['price_vs_Residual load']:.2f}), "
        f"ahead of load ({nums['price_vs_Load']:.2f}) and wind ({nums['price_vs_Wind gen.']:.2f}); "
        + (f"the 4-station wind speed tracks national wind output at r = {nums['windspeed_vs_windgen']:.2f}."
           if "windspeed_vs_windgen" in nums else "weather columns were not available in this source.")
    )
    _finish(fig, "Correlation matrix: price, fundamentals and weather",
            "Hourly Pearson correlations, Oct 2018 - Sep 2020")
    return fig, takeaway, nums


CHARTS = [
    ("01_price_timeseries", price_timeseries),
    ("02_seasonality_heatmap", seasonality_heatmap),
    ("03_price_vs_wind_residual_load", price_vs_wind_residual),
    ("04_merit_order_wind_deciles", merit_order_deciles),
    ("05_negative_prices_by_month", negative_prices_by_month),
    ("06_load_wind_distributions", load_wind_distributions),
    ("07_price_acf_pacf", price_acf_pacf),
    ("08_stl_decomposition", stl_decomposition),
    ("09_correlation_matrix", correlation_matrix),
]
