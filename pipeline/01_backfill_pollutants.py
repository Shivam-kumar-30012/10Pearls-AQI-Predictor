"""
Pipeline Step 01 - Backfill historical pollutant data.

Pulls hourly air pollution readings for the configured city from
OpenWeather's Air Pollution History API, from BACKFILL_START_DATE
(2020-11-27, when their archive begins) up to now.

This script writes RAW data only - no AQI, no features, no cleaning.
Those happen in step 04. Keeping collection and transformation
separate means a change to the feature logic never requires
re-downloading five years of data.

"""

import sys
import time
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CITY_LAT, CITY_LON, CITY_NAME,
    OPENWEATHER_API_KEY, BACKFILL_START_DATE,
    POLLUTANTS_RAW_CSV, require_secrets,
)

API_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"


def build_year_chunks(start_date: str):
    """
    OpenWeather rejects ranges longer than about a year, so we split
    the request into calendar-year chunks. Generated from config
    rather than hardcoded, so extending the range needs no code edit.
    """
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    chunks = []
    cursor = start
    while cursor < now:
        year_end = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        end = min(year_end, now)
        chunks.append((int(cursor.timestamp()), int(end.timestamp())))
        cursor = end
    return chunks


def fetch_chunk(start_unix: int, end_unix: int) -> list:
    params = {
        "lat": CITY_LAT,
        "lon": CITY_LON,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json().get("list", [])


def to_row(reading: dict) -> dict:
    """
    Flattens one API reading. Raw pollutant values only - the
    timestamp is stored as an ISO string in UTC 
    """
    dt = datetime.fromtimestamp(reading["dt"], tz=timezone.utc)
    c = reading["components"]
    return {
        "timestamp": dt.isoformat(),
        "city": CITY_NAME,
        "pm2_5": c.get("pm2_5"),
        "pm10": c.get("pm10"),
        "no2": c.get("no2"),
        "o3": c.get("o3"),
        "co": c.get("co"),
        "so2": c.get("so2"),
        "nh3": c.get("nh3"),
    }


def main():
    require_secrets("OPENWEATHER_API_KEY")

    chunks = build_year_chunks(BACKFILL_START_DATE)
    print("Backfilling %s pollutants from %s"
          % (CITY_NAME, BACKFILL_START_DATE))
    print("%d chunks to fetch\n" % len(chunks))

    rows = []
    for i, (start, end) in enumerate(chunks, 1):
        label = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m")
        print("  [%d/%d] %s ..." % (i, len(chunks), label), end=" ")
        try:
            readings = fetch_chunk(start, end)
        except requests.exceptions.HTTPError as e:
            print("FAILED (%s)" % e)
            continue
        print("%d readings" % len(readings))
        rows.extend(to_row(r) for r in readings)
        time.sleep(1)          # be polite to the API

    if not rows:
        print("\nNo data collected - check your API key and connection.")
        return

    # Sort and de-duplicate: chunk boundaries can overlap by an hour
    rows.sort(key=lambda r: r["timestamp"])
    seen = set()
    unique = []
    for r in rows:
        if r["timestamp"] not in seen:
            seen.add(r["timestamp"])
            unique.append(r)

    dropped = len(rows) - len(unique)
    if dropped:
        print("\nDropped %d duplicate timestamps from chunk overlaps" % dropped)

    with open(POLLUTANTS_RAW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(unique[0].keys()))
        writer.writeheader()
        writer.writerows(unique)

    print("\nSaved %d rows -> %s" % (len(unique), POLLUTANTS_RAW_CSV))
    print("Range: %s  ->  %s"
          % (unique[0]["timestamp"], unique[-1]["timestamp"]))


if __name__ == "__main__":
    main()