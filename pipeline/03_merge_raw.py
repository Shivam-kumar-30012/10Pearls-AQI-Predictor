"""
PIPELINE STEP 03 - Merge pollutants and weather

Joins the two raw sources on their hourly UTC timestamp into a single
table. This is the point the original pipeline got wrong: weather was
bolted on AFTER feature engineering, so no weather-derived feature
could ever reach the feature store, and the AQI was computed without
any weather context at all.

Merging here, before any transformation, means step 04 sees one
complete table and every downstream artefact stays consistent.

Reads:  data/raw/pollutants_historical.csv
        data/raw/weather_historical.csv
Writes: data/processed/raw_merged.csv

Run:    python pipeline/03_merge_raw.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    POLLUTANTS_RAW_CSV, WEATHER_RAW_CSV, MERGED_RAW_CSV, CITY_NAME,
)

POLLUTANT_COLS = ["pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3"]
WEATHER_COLS = ["temperature", "humidity", "pressure", "wind_speed",
                "wind_deg", "clouds", "precipitation",
                "boundary_layer_height"]


def load(path, label):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    before = len(df)
    df = (df.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True))
    print("  %-11s %6d rows  (%d dupes dropped)  %s -> %s"
          % (label, len(df), before - len(df),
             df.timestamp.min().date(), df.timestamp.max().date()))
    return df


def main():
    print("Loading raw sources:")
    poll = load(POLLUTANTS_RAW_CSV, "pollutants")
    weather = load(WEATHER_RAW_CSV, "weather")

    # Inner join: a row is only usable if we have BOTH pollutants and
    # weather for that hour. A left join would leave weather columns
    # empty on unmatched rows, which silently poisons later features.
    df = poll.merge(weather, on="timestamp", how="inner")
    print("\nInner join on timestamp -> %d rows" % len(df))

   
    expected = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq="h")
    missing = len(expected) - len(df)
    print("Expected hours in range: %d  |  missing: %d (%.2f%%)"
          % (len(expected), missing, missing / len(expected) * 100))

    gaps = df.timestamp.diff().dt.total_seconds().div(3600)
    big = gaps[gaps > 1]
    if len(big):
        print("Gaps > 1h: %d  |  largest: %.0f hours" % (len(big), big.max()))

    # ---- Missing value report ----
    print("\nMissing values per column:")
    nulls = df[POLLUTANT_COLS + WEATHER_COLS].isna().sum()
    any_missing = False
    for col, n in nulls.items():
        if n:
            print("  %-24s %5d (%.2f%%)" % (col, n, n / len(df) * 100))
            any_missing = True
    if not any_missing:
        print("  none")

    df["city"] = CITY_NAME

    df.to_csv(MERGED_RAW_CSV, index=False)
    print("\nSaved %d rows x %d cols -> %s"
          % (len(df), df.shape[1], MERGED_RAW_CSV))
    print("Range: %s -> %s" % (df.timestamp.min(), df.timestamp.max()))


if __name__ == "__main__":
    main()