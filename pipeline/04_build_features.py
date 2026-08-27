"""
PIPELINE STEP 04 - Build features


What this does, in order:

  1. Reindex onto a COMPLETE hourly grid.

  2. Compute AQI correctly, using EPA averaging windows

  3. Build features: time, cyclical, lags, rolling stats, trends,
     weather, and weather-derived (wind u/v, ventilation).

  4. Save an HOURLY feature table. 

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MERGED_RAW_CSV, FEATURES_CSV, PM_AVERAGING_HOURS, GAS_AVERAGING_HOURS,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aqi_calculator import calculate_aqi

MAX_INTERPOLATE_HOURS = 3

POLLUTANTS = ["pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3"]
WEATHER = ["temperature", "humidity", "pressure", "wind_speed",
           "wind_deg", "clouds", "precipitation"]



# 1. Complete hourly grid

def build_hourly_grid(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")   # indexing rows as timestamps rather than 0,1,2.....

    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    before = len(df)
    df = df.reindex(full)
    df.index.name = "timestamp"
    print("  Reindexed %d -> %d rows (%d missing hours inserted)"
          % (before, len(df), len(df) - before))

    numeric = [c for c in POLLUTANTS + WEATHER if c in df.columns]
    filled = df[numeric].isna().sum().sum()
    df[numeric] = df[numeric].interpolate(
        method="time", limit=MAX_INTERPOLATE_HOURS, limit_direction="both")
    remaining = df[numeric].isna().sum().sum()
    print("  Interpolated gaps <=%dh: %d values filled, %d still NaN"
          % (MAX_INTERPOLATE_HOURS, filled - remaining, remaining))

    return df.reset_index() # reindecx according to the rows as 0,1,2,3... 



# 2. Correct AQI



def add_aqi(df):
    pm25_avg = df["pm2_5"].rolling(PM_AVERAGING_HOURS, min_periods=18).mean()
    pm10_avg = df["pm10"].rolling(PM_AVERAGING_HOURS, min_periods=18).mean()
    co_avg = df["co"].rolling(GAS_AVERAGING_HOURS, min_periods=6).mean()
    o3_avg = df["o3"].rolling(GAS_AVERAGING_HOURS, min_periods=6).mean()

    results = [
        calculate_aqi(pm25=a, pm10=b, co=c, o3=d, return_dominant=True)
        for a, b, c, d in zip(pm25_avg, pm10_avg, co_avg, o3_avg)
    ]
    df["aqi"] = [r[0] for r in results]
    df["dominant_pollutant"] = [r[1] for r in results]

    valid = df["aqi"].notna().sum()  #counting non blank aqis
    print("  AQI computed for %d rows | mean %.1f | std %.1f"
          % (valid, df["aqi"].mean(), df["aqi"].std()))
    print("  Dominant pollutant:",
          df["dominant_pollutant"].value_counts().to_dict())
    return df



# 3. Features
#

# adding time features.
def add_time_features(df):
    ts = df["timestamp"].dt
    df["hour"] = ts.hour
    df["day"] = ts.day
    df["month"] = ts.month
    df["day_of_week"] = ts.dayofweek
    df["is_weekend"] = (ts.dayofweek >= 5).astype(int)

    # Cyclical: 23:00 and 00:00 are adjacent in time but far apart as
    # plain integers. Sin/cos puts them next to each other on a circle.

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Pakistan's seasons drive very different pollution regimes:
    # winter inversions trap smoke, monsoon rain washes air clean.

    season = pd.cut(df["month"], bins=[0, 2, 5, 9, 11, 12],
                    labels=["winter", "summer", "monsoon",
                            "post_monsoon", "winter2"], ordered=False)
    season = season.astype(str).replace("winter2", "winter")
    for s in ["winter", "summer", "monsoon", "post_monsoon"]:
        df["season_%s" % s] = (season == s).astype(int) #comparing the whole col at once. 
    return df


def add_lag_features(df):
    for h in [1, 3, 6, 12, 24, 48, 72, 168]:
        df["aqi_lag_%dh" % h] = df["aqi"].shift(h)

    for w, name in [(24, "1d"), (72, "3d"), (168, "7d")]:
        r = df["aqi"].rolling(w, min_periods=max(2, w // 4))
        df["aqi_roll_mean_%s" % name] = r.mean()
        df["aqi_roll_std_%s" % name] = r.std()

    df["aqi_roll_max_1d"] = df["aqi"].rolling(24, min_periods=6).max()
    df["aqi_roll_min_1d"] = df["aqi"].rolling(24, min_periods=6).min()
    df["aqi_range_1d"] = df["aqi_roll_max_1d"] - df["aqi_roll_min_1d"]

    # Trend: is pollution building or clearing right now?
    df["aqi_trend_6h"] = df["aqi"] - df["aqi_lag_6h"]
    df["aqi_trend_24h"] = df["aqi"] - df["aqi_lag_24h"]
    df["aqi_trend_72h"] = df["aqi"] - df["aqi_lag_72h"]

    # Anomaly: today versus the same period a week ago
    df["aqi_vs_7d"] = df["aqi"] - df["aqi_roll_mean_7d"]

    for p in ["pm2_5", "pm10", "o3", "co"]:
        df["%s_lag_24h" % p] = df[p].shift(24)
        df["%s_roll_24h" % p] = df[p].rolling(24, min_periods=6).mean()
    return df


def add_weather_features(df):
    # Wind direction is circular: 359 deg and 1 deg are nearly the same
    # direction but maximally distant as raw numbers. Decomposing into
    # u/v components lets the model use direction properly

    rad = np.deg2rad(df["wind_deg"])
    df["wind_u"] = -df["wind_speed"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(rad)

    # Dispersion proxies
    df["ventilation"] = df["wind_speed"] * (100 - df["humidity"]) / 100
    df["calm_flag"] = (df["wind_speed"] < 1.5).astype(int)

    # Pressure rising usually means settled, stagnant air
    df["pressure_trend_24h"] = df["pressure"] - df["pressure"].shift(24)
    df["temp_range_24h"] = (df["temperature"].rolling(24, min_periods=6).max()
                            - df["temperature"].rolling(24, min_periods=6).min())

    # Rain scavenges particulates - cumulative recent rainfall matters
    # more than the current hour's reading
    df["precip_24h"] = df["precipitation"].rolling(24, min_periods=6).sum()
    df["precip_72h"] = df["precipitation"].rolling(72, min_periods=12).sum()

    for c in ["temperature", "humidity", "wind_speed", "pressure"]:
        df["%s_roll_24h" % c] = df[c].rolling(24, min_periods=6).mean()
    return df


def main():
    print("Loading merged raw data...")
    df = pd.read_csv(MERGED_RAW_CSV)
    print("  %d rows" % len(df))

    print("\n[1/4] Building complete hourly grid")
    df = build_hourly_grid(df)

    print("\n[2/4] Computing AQI (EPA averaging + corrected breakpoints)")
    df = add_aqi(df)

    print("\n[3/4] Building features")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_weather_features(df)

    print("\n[4/4] Finalising")
    # Drop the warm-up period where long rolling windows are undefined.
    # No forecast targets are created here: step 06 builds those when
    # it expands rows across the hours_ahead dimension, which keeps
    # this table a clean point-in-time snapshot for the feature store.
    before = len(df)
    df = df.dropna(subset=["aqi", "aqi_lag_168h", "aqi_roll_mean_7d"])
    df = df.reset_index(drop=True)
    print("  Dropped %d warm-up/gap rows -> %d remain" % (before - len(df), len(df)))


    if "boundary_layer_height" in df.columns:
        df = df.drop(columns=["boundary_layer_height"])
        print("  Dropped boundary_layer_height (~9% missing in ERA5)")

    df["city"] = df["city"].ffill().bfill()

    df.to_csv(FEATURES_CSV, index=False)
    print("\nSaved %d rows x %d cols -> %s" % (len(df), df.shape[1], FEATURES_CSV))
    print("Range: %s -> %s" % (df.timestamp.min(), df.timestamp.max()))
    print("\nAQI: mean %.1f | std %.1f | min %.0f | max %.0f"
          % (df.aqi.mean(), df.aqi.std(), df.aqi.min(), df.aqi.max()))
    print("corr(aqi, aqi 24h later): %.3f" % df.aqi.corr(df.aqi.shift(-24)))


if __name__ == "__main__":
    main()