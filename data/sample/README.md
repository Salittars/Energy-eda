# Offline sample

`de_hourly_2018-10_2020-09.parquet` (~0.4 MB) is a compact, cleaned extract of the
hourly German power-market table: 17,544 hours, 1 Oct 2018 to 30 Sep 2020 (UTC).

| Column | Unit | Source |
|---|---|---|
| `price_eur_mwh` | EUR/MWh | DE-LU day-ahead price, Open Power System Data (OPSD) |
| `load_mw` | MW | ENTSO-E actual load via OPSD |
| `wind_generation_mw` | MW | ENTSO-E actual wind (on + offshore) via OPSD |
| `solar_generation_mw` | MW | ENTSO-E actual solar via OPSD |
| `wind_speed_100m_ms_mean` | m/s | 4-station average (Hamburg, Berlin, Frankfurt, Munich), DWD observations scaled to 100 m |
| `temperature_c_mean` | °C | same 4 stations, DWD |

It was produced by `python -m energy_eda make-sample` from the `hourly` table of the
[energy-data-pipeline](https://github.com/Salittars/energy-data-pipeline) project plus the
OPSD solar column. Values are rounded (MW to 1 MW, price to 0.01 EUR/MWh) to keep it small.

Data: Open Power System Data, Time series package 2020-10-06, and
Deutscher Wetterdienst open data.
