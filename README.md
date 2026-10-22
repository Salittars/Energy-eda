# energy-eda

**Proves:** "Matplotlib/Seaborn"

Exploratory and time-series analysis of two years of hourly German power-market data (day-ahead price, load, wind, solar, weather), turned into nine publication-quality charts with numeric takeaways, for an energy trader or analyst who wants to know what drives the price before building a forecast.

## Demo

**Merit-order effect:** the windiest 10% of hours are 54% cheaper than the calmest 10%.

![Price by wind decile](figures/04_merit_order_wind_deciles.png)

**Wind vs residual load:** residual load explains 68% of hourly price variance, wind alone 15%.

![Price vs wind and residual load](figures/03_price_vs_wind_residual_load.png)

**Seasonality:** weekday 19:00 is the most expensive hour, Sunday 14:00 the cheapest.

![Hour x weekday heatmap](figures/02_seasonality_heatmap.png)

All nine charts with their one-sentence takeaways: [`figures/takeaways.md`](figures/takeaways.md). Full walkthrough: [`notebooks/energy_eda.ipynb`](notebooks/energy_eda.ipynb) (executed, outputs kept).

## How it works

- **Data:** 17,544 hourly rows (1 Oct 2018 - 30 Sep 2020, UTC) from the cleaned `hourly` table of [energy-data-pipeline](https://github.com/Salittars/energy-data-pipeline) (DE-LU day-ahead price, German load and wind from OPSD/ENTSO-E, 4-station DWD weather), plus German solar from the same OPSD file. A 0.4 MB Parquet extract is committed in `data/sample/` so everything runs offline; `--source duckdb` reads the pipeline output directly and `--source opsd` downloads the OPSD file itself.
- **Features:** residual load (load - wind - solar), Berlin local hour and weekday (so daylight saving does not shift the daily peaks), wind-generation deciles, negative-price counts per local month. These helpers live in `energy_eda/features.py` and are covered by pytest.
- **Charts:** one function per chart in `energy_eda/plots.py` (Matplotlib + Seaborn, one shared style). Each returns the figure and a takeaway sentence built from the data, so captions can never drift from the numbers.
- **Time-series methods:** rolling means (7 and 30 days), ACF/PACF up to 8 days of lags (statsmodels), robust STL decomposition of the daily mean price with a weekly period, Pearson correlation matrix.
- **Output:** `python -m energy_eda report` writes nine PNGs plus `figures/takeaways.md` and `figures/results.json` (every number quoted below) in about 5 seconds.

## Results

| Chart | Finding |
|---|---|
| Price + rolling means | Mean price fell from **41.7** EUR/MWh (Oct 2018-Sep 2019) to **29.9** (Oct 2019-Sep 2020), **-28%**; the 30-day mean bottomed at **16.0** EUR/MWh in early May 2020. Range -90.01 (8 Jun 2019) to 200.04 EUR/MWh (21 Sep 2020). |
| Hour x weekday heatmap | Highest average **52.7** EUR/MWh (Thursday 19:00), lowest **14.3** (Sunday 14:00, low demand + midday solar). Sundays average **25.2** vs **39.0** on weekdays. |
| Price vs wind / residual load | Wind alone: r = **-0.39** (R² 0.15). Residual load: r = **0.83** (R² 0.68), about **+1.19 EUR/MWh per extra GW**. |
| Merit order by wind decile | Windiest decile (30.7-46.1 GW): **19.4** EUR/MWh vs **42.3** in the calmest (0.1-3.3 GW), **-54%**. The windiest decile holds **265 of 484** negative-price hours. |
| Negative prices by month | **484** negative hours (195 in year one, 289 in year two); worst month **February 2020** with **84** hours (winter storms); **65%** fell on weekends; mean price in those hours -16.2 EUR/MWh. |
| Distributions | Load stays in 31.9-76.9 GW (CV **0.18**, two humps for night/weekend vs working hours); wind is right-skewed (skew **0.90**, median 11.7 vs mean 14.4 GW, CV **0.72**). |
| ACF / PACF | Lag-1 autocorrelation **0.95**; falls to **0.40** at 18 h, back up to **0.64** at 24 h and **0.59** at 168 h (one week). PACF: two big spikes (lag 1 **+0.95**, lag 2 **-0.43**), then a daily echo at 25 h (-0.28). |
| STL (daily, weekly period) | Trend from 55.6 down to **17.5** EUR/MWh (late March 2020) and back to 42.2; weekly cycle **-8.6** EUR/MWh on Sundays, -4.0 Saturdays, **+2.5** on weekdays; trend 44% / weekly 15% / residual **34%** of daily variance. |
| Correlation matrix | Price vs residual load 0.83, load 0.49, wind -0.39, solar -0.19, temperature -0.13; 4-station wind speed vs national wind output **0.73**. |

**What failed at first:** the first version of the negative-prices chart grouped hours by Berlin local month and showed an extra, empty "October 2020" bar. The data ends at 30 Sep 2020 23:00 UTC, but 22:00 and 23:00 UTC are already 1 October in Berlin, so two hours created a fake month. Converting to local time was right (it puts the daily peaks in the correct place across daylight saving), so the fix was to keep local time and drop any month covered by fewer than 27 days of hours. A regression test (`test_negative_hours_by_month_drops_partial_trailing_month`) now guards it, together with two tests that pin the winter (UTC+1) and summer (UTC+2) hour mapping.

## Run it

```bash
git clone <this repo> && cd energy-eda
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m energy_eda report        # 9 charts -> figures/, from the offline sample (or: make report)
python -m pytest -q                # 9 tests (or: make test)
jupyter notebook notebooks/energy_eda.ipynb
```

To run on the full pipeline output instead of the sample, build [energy-data-pipeline](https://github.com/Salittars/energy-data-pipeline) first (`python -m energy_pipeline build`), then:

```bash
python -m energy_eda report --source duckdb \
    --duckdb ../energy-data-pipeline/data/energy.duckdb \
    --opsd-csv ../energy-data-pipeline/data/raw/opsd_time_series_60min.csv   # adds solar
python -m energy_eda report --source opsd       # or: download OPSD directly (~130 MB, no weather columns)
python -m energy_eda make-sample --duckdb ... --opsd-csv ...   # rebuild data/sample/
```

Data: Open Power System Data, Time series package 2020-10-06 (ENTSO-E Transparency Platform underneath), and Deutscher Wetterdienst open data. Please follow their attribution terms.

## What I'd do next

- Turn the findings into a day-ahead price model: residual-load forecast, local hour/weekday and price lags at 24 h and 168 h as features, gradient boosting vs a linear baseline, scored with a rolling-origin backtest.
- Add gas (TTF) and carbon (EUA) prices to explain the residual 34% of the STL variance and the 2020 trend, and extend past September 2020 with the ENTSO-E API to cover the 2021-2022 price shock.
