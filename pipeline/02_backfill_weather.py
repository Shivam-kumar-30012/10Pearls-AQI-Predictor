"""
Pipeline Step 02 - Backfill historical weather data.

Uses Open-Meteo's ERA5 reanalysis archive rather than OpenWeather.
OpenWeather's free tier only reaches about five days back for historical
weather, while Open-Meteo is free, needs no API key, and covers any date
range from 1940 onward. Same location, same hourly resolution, same UTC
timestamps as the pollutant data, so step 03 can join them directly.

Weather matters because AQI is largely meteorology-driven: wind disperses
pollutants, humidity grows particles, and temperature inversions trap them
near the ground.


"""

import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CITY_LAT, CITY_LON, CITY_NAME,
    BACKFILL_START_DATE, WEATHER_RAW_CSV,
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo variable name -> our column name
VARIABLES = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_deg",
    "cloud_cover": "clouds",
    "precipitation": "precipitation",
    "boundary_layer_height": "boundary_layer_height",
}


def fetch_weather(start_date: str, end_date: str) -> dict:
    params = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(VARIABLES.keys()),
        "timezone": "UTC",
    }
    response = requests.get(API_URL, params=params, timeout=180)
    response.raise_for_status()
    return response.json()


def main():
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("Fetching %s weather: %s -> %s"
          % (CITY_NAME, BACKFILL_START_DATE, end_date))
    print("(Open-Meteo ERA5 reanalysis - free, no API key)\n")

    data = fetch_weather(BACKFILL_START_DATE, end_date)
    hourly = data["hourly"]
    times = hourly["time"]
    print("Got %d hourly readings\n" % len(times))

    if not times:
        print("No data returned - check dates and connection.")
        return

    rows = []
    for i, t in enumerate(times):
        # Open-Meteo returns "2020-11-27T00:00"; add the UTC marker so
        # it parses identically to the pollutant timestamps
        row = {"timestamp": t + ":00+00:00" if len(t) == 16 else t}
        for api_name, our_name in VARIABLES.items():
            series = hourly.get(api_name)
            row[our_name] = series[i] if series else None
        rows.append(row)

    with open(WEATHER_RAW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved %d rows -> %s" % (len(rows), WEATHER_RAW_CSV))
    print("Range: %s  ->  %s" % (rows[0]["timestamp"], rows[-1]["timestamp"]))
    print("Columns:", ", ".join(rows[0].keys()))


if __name__ == "__main__":
    main()