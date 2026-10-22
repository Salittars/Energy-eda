"""Load the hourly German power-market table from one of three sources.

1. ``sample`` (default): the compact Parquet extract committed in data/sample/,
   so everything runs offline.
2. ``duckdb``: the ``hourly`` table built by the energy-data-pipeline project
   (github.com/Salittars/energy-data-pipeline). Solar is not in that table, so
   it is merged from the OPSD file when one is available.
3. ``opsd``: download the Open Power System Data file directly (power data only,
   no weather columns).

Every loader returns the same frame: a UTC ``DatetimeIndex`` named
``timestamp_utc`` and the columns in ``COLUMNS`` (weather columns may be NaN).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
OPSD_COLUMNS = {
    "DE_wind_generation_actual": "wind_generation_mw",
    "DE_solar_generation_actual": "solar_generation_mw",
    "DE_load_actual_entsoe_transparency": "load_mw",
    "DE_LU_price_day_ahead": "price_eur_mwh",
}
START, END = "2018-10-01 00:00", "2020-09-30 23:00"

COLUMNS = [
    "price_eur_mwh",
    "load_mw",
    "wind_generation_mw",
    "solar_generation_mw",
    "wind_speed_100m_ms_mean",
    "temperature_c_mean",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "data" / "sample" / "de_hourly_2018-10_2020-09.parquet"


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Put any frame on the shared shape: UTC hourly index, fixed columns."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    df.index.name = "timestamp_utc"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.loc[START:END]
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS].astype("float64")


def read_opsd_csv(path: str | Path) -> pd.DataFrame:
    """Read the four German power columns from an OPSD 60-min CSV."""
    df = pd.read_csv(path, usecols=["utc_timestamp", *OPSD_COLUMNS])
    df = df.rename(columns=OPSD_COLUMNS)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    return df.set_index("utc_timestamp")


def download_opsd(dest: str | Path = REPO_ROOT / "data" / "opsd_time_series_60min.csv") -> Path:
    """Download the OPSD file (about 130 MB) unless it is already cached."""
    import requests

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(OPSD_URL, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def load_sample(path: str | Path = SAMPLE_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _finish(df.set_index("timestamp_utc"))


def load_opsd(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else download_opsd()
    return _finish(read_opsd_csv(path))


def load_duckdb(db_path: str | Path, opsd_csv: str | Path | None = None) -> pd.DataFrame:
    """Read the pipeline's ``hourly`` table; add solar from OPSD if given."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.sql(
            "SELECT timestamp_utc, price_eur_mwh, load_mw, wind_generation_mw, "
            "wind_speed_100m_ms_mean, temperature_c_mean FROM hourly ORDER BY 1"
        ).df()
    finally:
        con.close()
    df = df.set_index("timestamp_utc")
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    if opsd_csv and Path(opsd_csv).exists():
        solar = read_opsd_csv(opsd_csv)["solar_generation_mw"]
        df = df.join(solar, how="left")
    return _finish(df)


def load(source: str = "sample", duckdb_path: str | None = None, opsd_csv: str | None = None) -> pd.DataFrame:
    if source == "sample":
        return load_sample()
    if source == "duckdb":
        if not duckdb_path:
            raise ValueError("--duckdb PATH is required with --source duckdb")
        return load_duckdb(duckdb_path, opsd_csv)
    if source == "opsd":
        return load_opsd(opsd_csv)
    raise ValueError(f"unknown source: {source!r}")


def write_sample(df: pd.DataFrame, path: str | Path = SAMPLE_PATH) -> Path:
    """Save the compact extract committed with the repo (rounded to cut size)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in ["load_mw", "wind_generation_mw", "solar_generation_mw"]:
        out[col] = out[col].round(0).astype("float32")
    for col in ["price_eur_mwh", "wind_speed_100m_ms_mean", "temperature_c_mean"]:
        out[col] = out[col].round(2).astype("float32")
    out = out.reset_index()
    out["timestamp_utc"] = out["timestamp_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    out.to_parquet(path, index=False, compression="zstd")
    return path
