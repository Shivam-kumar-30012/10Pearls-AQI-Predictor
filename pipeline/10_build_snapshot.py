"""
Build the dashboard snapshot.

Writes a small JSON file holding everything the dashboard needs: the
current reading, the 72-hour forecast, 30 days of history, and the model
metrics.

Why this exists
---------------
Streamlit Cloud runs from the GitHub repo, where data/ is gitignored and
the runner has no local CSV. The dashboard reads Hopsworks first, but
that free tier is unreliable - reads time out, quota runs out. A ~100 KB
snapshot committed to the repo means the dashboard always renders
something, even when the store is unreachable.

The daily training workflow regenerates this, so it is never more than a
day stale.

Run:  python pipeline/10_build_snapshot.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

import config
from aqi_calculator import aqi_category

OUT = ROOT / "data" / "dashboard_snapshot.json"
HISTORY_DAYS = 30


def main():
    import importlib
    predict = importlib.import_module("09_predict")

    print("Generating forecast...")
    forecast, info = predict.forecast(verbose=True)
    daily = predict.daily_summary(forecast)

    print("\nLoading history...")
    df, source = predict.load_features(verbose=False)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=HISTORY_DAYS)
    hist = df[df["timestamp"] >= cutoff][
        ["timestamp", "aqi", "pm2_5", "pm10", "no2", "o3", "co", "so2",
         "temperature", "humidity", "wind_speed", "wind_deg"]
    ].copy()
    print("  %d hours of history" % len(hist))

    latest = df.iloc[-1]

    # Trend: now versus six hours ago, which is long enough to be
    # meaningful and short enough to still be "now".
    six_ago = df["aqi"].iloc[-7] if len(df) > 7 else latest["aqi"]
    trend = float(latest["aqi"] - six_ago)

    snapshot = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "predicted_from": info["predicted_from"].isoformat(),
        "data_source": info["data_source"],
        "city": config.CITY_NAME,
        "current": {
            "aqi": int(round(latest["aqi"])),
            "category": aqi_category(latest["aqi"])[0],
            "dominant_pollutant": str(latest.get("dominant_pollutant", "unknown")),
            "trend_6h": round(trend, 1),
            "temperature": round(float(latest["temperature"]), 1),
            "humidity": int(round(latest["humidity"])),
            "wind_speed": round(float(latest["wind_speed"]), 1),
            "wind_deg": int(round(latest["wind_deg"])),
            "pollutants": {
                p: round(float(latest[p]), 1)
                for p in ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
            },
        },
        "forecast": [
            {
                "timestamp": r["timestamp"].isoformat(),
                "hours_ahead": int(r["hours_ahead"]),
                "day": int(r["day"]),
                "aqi": int(r["aqi"]),
                "category": r["category"],
            }
            for _, r in forecast.iterrows()
        ],
        "daily": [
            {
                "day": int(r["day"]),
                "date": str(r["date"]),
                "mean_aqi": int(r["mean_aqi"]),
                "min_aqi": int(r["min_aqi"]),
                "max_aqi": int(r["max_aqi"]),
                "category": r["category"],
            }
            for _, r in daily.iterrows()
        ],
        "history": [
            {
                "timestamp": r["timestamp"].isoformat(),
                "aqi": int(round(r["aqi"])),
            }
            for _, r in hist.iterrows()
            if pd.notna(r["aqi"])
        ],
        "pollutant_history": {
            p: [round(float(v), 1) for v in hist[p].tail(48).fillna(0)]
            for p in ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
        },
        "models": info["models"],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(snapshot, f, indent=1)

    size_kb = OUT.stat().st_size / 1024
    print("\nSaved %s (%.0f KB)" % (OUT.name, size_kb))
    print("  current AQI %d (%s)"
          % (snapshot["current"]["aqi"], snapshot["current"]["category"]))
    print("  forecast: %d hours" % len(snapshot["forecast"]))
    print("  history: %d hours" % len(snapshot["history"]))


if __name__ == "__main__":
    main()