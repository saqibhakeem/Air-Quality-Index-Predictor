"""
backfill_historical.py

Populates data/raw_data.csv with historical rows so build_features.py and
the training pipeline have enough history to work with (rather than waiting
weeks for fetch_raw_data.py to accumulate it hour by hour).

Data sources:
- OpenWeather Air Pollution History API (free tier, hourly data available
  from 2020-11-27 onwards) -> pollutant concentrations, incl. pm2_5/pm10
- Open-Meteo Historical Weather API (free, NO API key required) -> temp,
  humidity, wind, pressure

The EPA AQI target (aqi_us_epa) is computed from pm2_5/pm10 using the same
epa_aqi.py module fetch_raw_data.py uses, so live and backfilled rows are
on an identical scale.

Usage:
    export OPENWEATHER_API_KEY="..."
    python backfill_historical.py --days 90
    # or
    python backfill_historical.py --start 2026-05-01 --end 2026-08-01
"""

import os
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd

from epa_aqi import compute_overall_aqi
from fetch_raw_data import CITY_NAME, LAT, LON, RAW_FIELDS, RAW_CSV_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
OPENWEATHER_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

CHUNK_DAYS = 30  # fetch pollution history in month-sized chunks to keep responses small
EARLIEST_AVAILABLE = datetime(2020, 11, 27, tzinfo=timezone.utc)  # OpenWeather history floor


def fetch_pollution_history(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly pollutant concentrations from OpenWeather, chunked by month."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        params = {
            "lat": LAT,
            "lon": LON,
            "start": int(chunk_start.timestamp()),
            "end": int(chunk_end.timestamp()),
            "appid": OPENWEATHER_API_KEY,
        }
        log.info("Fetching pollution history %s -> %s", chunk_start.date(), chunk_end.date())
        resp = requests.get(OPENWEATHER_HISTORY_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for item in data.get("list", []):
            comp = item["components"]
            rows.append({
                "timestamp_utc": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                "aqi_openweather": item["main"]["aqi"],
                "pm2_5": comp.get("pm2_5"),
                "pm10": comp.get("pm10"),
                "no2": comp.get("no2"),
                "so2": comp.get("so2"),
                "co": comp.get("co"),
                "o3": comp.get("o3"),
                "nh3": comp.get("nh3"),
            })
        frames.append(pd.DataFrame(rows))
        chunk_start = chunk_end

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_weather_history(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch hourly weather history from Open-Meteo (no API key needed)."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "timezone": "UTC",
    }
    log.info("Fetching weather history %s -> %s from Open-Meteo", start.date(), end.date())
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
        "temp_c": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"],
        "wind_speed_ms": hourly["wind_speed_10m"],
        "pressure_hpa": hourly["surface_pressure"],
    })
    return df


def build_backfill_rows(start: datetime, end: datetime) -> pd.DataFrame:
    pollution_df = fetch_pollution_history(start, end)
    weather_df = fetch_weather_history(start, end)

    if pollution_df.empty:
        raise RuntimeError("No pollution history returned - check date range / API key")

    # Align to the same hourly timestamps (OpenWeather and Open-Meteo both
    # report on the hour, but round defensively in case of second-level drift).
    pollution_df["timestamp_utc"] = pollution_df["timestamp_utc"].dt.floor("h")
    weather_df["timestamp_utc"] = weather_df["timestamp_utc"].dt.floor("h")

    merged = pollution_df.merge(weather_df, on="timestamp_utc", how="inner")

    merged["city"] = CITY_NAME
    merged["lat"] = LAT
    merged["lon"] = LON
    merged["weather_main"] = None  # not available from Open-Meteo without extra mapping
    merged["aqi_us_epa"] = merged.apply(
        lambda r: compute_overall_aqi(r["pm2_5"], r["pm10"]), axis=1
    )
    merged["aqi_aqicn"] = None  # AQICN has no historical endpoint - live-only field

    merged = merged.reindex(columns=RAW_FIELDS)
    return merged


def merge_into_raw_csv(new_rows: pd.DataFrame, path: Path = RAW_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["timestamp_utc"])
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    combined = combined.drop_duplicates(subset="timestamp_utc", keep="last")
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

    combined.to_csv(path, index=False)
    log.info("raw_data.csv now has %d total rows (%d new)", len(combined), len(new_rows))


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill historical AQI/weather data")
    parser.add_argument("--days", type=int, default=90, help="Number of past days to backfill")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (default: now)")
    return parser.parse_args()


def main():
    args = parse_args()
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=args.days)

    start = max(start, EARLIEST_AVAILABLE)
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")

    log.info("Backfilling %s for %s -> %s", CITY_NAME, start.date(), end.date())
    new_rows = build_backfill_rows(start, end)
    merge_into_raw_csv(new_rows)


if __name__ == "__main__":
    main()
