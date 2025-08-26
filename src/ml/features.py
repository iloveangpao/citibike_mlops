# import pandas as pd, argparse, os
# from datetime import timedelta

# def main(out):
#     pg_url = "postgresql://airflow:airflow@postgres:5432/airflow"
#     trips = pd.read_sql("SELECT * FROM trips_raw", pg_url)
#     station = pd.read_sql("SELECT * FROM station_status_raw", pg_url)

#     # …join, pivot, lags…
#     trips["date"] = pd.to_datetime(trips.started_at).dt.date
#     feats = trips.groupby(["start_station_id","date"]).size().to_frame("rides").reset_index()
#     # six lag features
#     for lag in [1,2,3,7]:
#         feats[f"lag_{lag}d"] = feats.groupby("start_station_id").rides.shift(lag)

#     feats.dropna().to_parquet(out)
#     print("features ready:", out)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--out", required=True)
#     args = p.parse_args()
#     main(args.out)


# src/ml/features.py
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from typing import Optional, List

import pandas as pd


def _log(msg: str) -> None:
    print(f"[features] {msg}", flush=True)


def _pick_csv_from_zip(zf: zipfile.ZipFile) -> str:
    """
    Pick a CSV member deterministically from a Citi Bike archive.
    Prefer names containing 'tripdata' and ending with '.csv'; otherwise first CSV.
    """
    names = zf.namelist()
    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        raise ValueError("No CSV files found inside ZIP.")
    # prefer ones that look like the official dump
    preferred: List[str] = [n for n in csvs if re.search(r"tripdata", n, re.I)]
    chosen = sorted(preferred or csvs)[0]
    _log(f"Selected CSV inside ZIP: {chosen}")
    return chosen


def _read_csv_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        member = _pick_csv_from_zip(zf)
        with zf.open(member) as fp:
            # robust decode; Citi Bike is UTF-8 but be defensive
            return pd.read_csv(io.TextIOWrapper(fp, encoding="utf-8", errors="ignore"))


def _read_csv_any(path: str) -> pd.DataFrame:
    path_l = path.lower()
    if path_l.endswith(".zip"):
        return _read_csv_from_zip(path)
    if path_l.endswith(".csv"):
        return pd.read_csv(path)
    if path_l.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    # last resort: let pandas guess
    return pd.read_csv(path)


def _std_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return the first existing column name (case-insensitive) from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize old vs new Citi Bike schemas into:
      started_at (datetime64[ns, UTC-naive]), ended_at (datetime64[ns]),
      start_station_id (string), end_station_id (string) if present.
    Also compute ride_minutes (float).
    """
    df = df.copy()

    # Unify time columns
    started = _std_col(df, "started_at", "starttime", "Start Time")
    ended = _std_col(df, "ended_at", "stoptime", "Stop Time")
    if started is None or ended is None:
        raise ValueError(
            f"Cannot find start/end time columns. Have: {list(df.columns)[:15]} ..."
        )

    # Parse datetimes
    df["_started_at"] = pd.to_datetime(df[started], errors="coerce", utc=False)
    df["_ended_at"] = pd.to_datetime(df[ended], errors="coerce", utc=False)

    # Normalize station ids (strings to preserve leading zeros / mixed types)
    start_id = _std_col(df, "start_station_id", "start station id", "Start Station ID")
    end_id = _std_col(df, "end_station_id", "end station id", "End Station ID")

    if start_id is not None:
        df["_start_station_id"] = df[start_id].astype("string")
    else:
        # Some rows/years lack station id; back off to station name (less ideal)
        start_name = _std_col(df, "start_station_name", "start station name", "Start Station Name")
        df["_start_station_id"] = df[start_name].astype("string") if start_name else pd.Series(pd.NA, index=df.index, dtype="string")

    if end_id is not None:
        df["_end_station_id"] = df[end_id].astype("string")

    # Compute ride duration minutes; prefer given tripduration when available
    tripdur = _std_col(df, "tripduration")  # seconds in older data
    if tripdur and pd.api.types.is_numeric_dtype(df[tripdur]):
        df["_ride_minutes"] = pd.to_numeric(df[tripdur], errors="coerce") / 60.0
    else:
        # fall back to difference
        df["_ride_minutes"] = (df["_ended_at"] - df["_started_at"]).dt.total_seconds() / 60.0

    # Clean rows with impossible/invalid values
    df = df.dropna(subset=["_started_at", "_ride_minutes", "_start_station_id"])
    df = df[(df["_ride_minutes"] >= 0.5) & (df["_ride_minutes"] <= 480)]  # 0.5 to 8 hours

    # Final normalized subset
    return df[["_started_at", "_start_station_id", "_ride_minutes"]].rename(
        columns={
            "_started_at": "started_at",
            "_start_station_id": "start_station_id",
            "_ride_minutes": "ride_minutes",
        }
    )


def build_features(raw: pd.DataFrame, include_weather: bool = False) -> pd.DataFrame:
    """
    Build feature set for Citi Bike demand prediction.
    
    Features:
    Base features (always included):
      - ds (date, string): YYYY-MM-DD
      - hour (int16): Hour of day (0-23)
      - start_station_id (string): Station identifier
      - trips (int32): Number of trips started
      - avg_duration_min (float32): Average trip duration in minutes
    
    Weather features (if include_weather=True):
      - temp_f (float32): Temperature in Fahrenheit
      - precip_probability (float32): Probability of precipitation
      - wind_speed_mph (float32): Wind speed in mph
    
    Args:
        raw: DataFrame with raw trip data
        include_weather: Whether to include weather features
    
    Returns:
        DataFrame with engineered features
    """
    # Normalize schema and create base features
    df = _normalize_schema(raw)
    df["ds"] = df["started_at"].dt.date.astype("string")
    df["hour"] = df["started_at"].dt.hour.astype("int16")

    # Group by station and time
    grp = (
        df.groupby(["start_station_id", "ds", "hour"], observed=True)
          .agg(
              trips=("ride_minutes", "size"),
              avg_duration_min=("ride_minutes", "mean"),
          )
          .reset_index()
    )

    # Add time-based features
    grp["day_of_week"] = pd.to_datetime(grp["ds"]).dt.dayofweek.astype("int8")
    grp["is_weekend"] = (grp["day_of_week"] >= 5).astype("int8")
    
    # Add lagged features (previous hour, day, week)
    for station in grp["start_station_id"].unique():
        mask = grp["start_station_id"] == station
        station_data = grp[mask].sort_values(["ds", "hour"])
        
        # Previous hour
        grp.loc[mask, "prev_hour_trips"] = station_data["trips"].shift(1)
        
        # Same hour yesterday
        grp.loc[mask, "prev_day_trips"] = station_data["trips"].shift(24)
        
        # Same hour last week
        grp.loc[mask, "prev_week_trips"] = station_data["trips"].shift(24 * 7)
    
    # If weather data is requested, fetch from database
    if include_weather:
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        
        # Get weather data from our database (assumes weather table exists)
        pg = PostgresHook("POSTGRES_CONN").get_conn()
        weather = pd.read_sql("""
            SELECT 
                date(timestamp) as ds,
                EXTRACT(HOUR FROM timestamp) as hour,
                temperature_f,
                precipitation_probability,
                wind_speed_mph
            FROM weather_data
            WHERE timestamp >= %s AND timestamp <= %s
        """, pg, params=[grp["ds"].min(), grp["ds"].max()])
        
        # Merge weather features
        grp = grp.merge(
            weather,
            on=["ds", "hour"],
            how="left"
        )
    
    # Ensure consistent dtypes for ML pipeline
    dtype_map = {
        "start_station_id": "string",
        "ds": "string",
        "hour": "int16",
        "day_of_week": "int8",
        "is_weekend": "int8",
        "trips": "int32",
        "avg_duration_min": "float32",
        "prev_hour_trips": "float32",
        "prev_day_trips": "float32",
        "prev_week_trips": "float32"
    }
    
    if include_weather:
        dtype_map.update({
            "temperature_f": "float32",
            "precipitation_probability": "float32",
            "wind_speed_mph": "float32"
        })
    
    for col, dtype in dtype_map.items():
        if col in grp.columns:
            grp[col] = grp[col].astype(dtype)
    
    return grp


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build features from a Citi Bike trips archive (zip/csv).")
    ap.add_argument("--trips_zip", required=True, help="Path to trips archive file (.zip/.csv.zip) or a .csv/.csv.gz")
    ap.add_argument("--out", required=True, help="Output Parquet path")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _log(f"Loading: {args.trips_zip}")
    raw = _read_csv_any(args.trips_zip)
    _log(f"Raw shape: {raw.shape}, columns: {list(raw.columns)[:10]}...")

    feats = build_features(raw)
    _log(f"Features shape: {feats.shape}")
    # Parquet requires pyarrow or fastparquet; we depend on pyarrow in pyproject.
    feats.to_parquet(args.out, index=False)
    _log(f"Wrote features: {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
