"""
Rebuilds target_aqi and the day1/2/3 targets using the CORRECTED
EPA calculator, then reports a before/after comparison.

Run from your project folder, with aqi_calculator_fixed.py alongside it:
    python rebuild_targets.py

Reads:  aqi_training_data_with_weather.csv
Writes: aqi_training_data_fixed.csv
"""

import pandas as pd
import numpy as np
from aqi_calculator_fixed import compute_aqi_series

IN_CSV = "aqi_training_data_with_weather.csv"
OUT_CSV = "aqi_training_data_fixed.csv"

df = pd.read_csv(IN_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
print("Loaded %d rows" % len(df))

old_aqi = df["target_aqi"].copy()

# ---- Recompute AQI with correct breakpoints AND correct averaging ----
print("Recomputing AQI with EPA averaging windows (24h PM, 8h CO/O3)...")
df = compute_aqi_series(df)

# ---- Rebuild every derived column from the corrected AQI ----
print("Rebuilding lags, rolling means, and forecast targets...")
df["target_aqi"] = df["aqi_correct"]

for h in [1, 3, 6, 12, 24, 48, 72]:
    df["aqi_lag_%dh" % h] = df["target_aqi"].shift(h)

df["aqi_rolling_3d"] = df["target_aqi"].rolling(72, min_periods=1).mean()
df["aqi_rolling_7d"] = df["target_aqi"].rolling(168, min_periods=1).mean()

df["target_day1"] = df["target_aqi"].shift(-24)
df["target_day2"] = df["target_aqi"].shift(-48)
df["target_day3"] = df["target_aqi"].shift(-72)

before = len(df)
df = df.dropna(subset=["target_aqi", "target_day1", "target_day2", "target_day3",
                       "aqi_lag_72h"]).reset_index(drop=True)
print("Dropped %d edge rows -> %d remain" % (before - len(df), len(df)))

# ---- Before / after ----
print("\n" + "=" * 58)
print("BEFORE vs AFTER")
print("=" * 58)
print("%-28s %12s %12s" % ("", "OLD", "NEW"))
print("%-28s %12d %12d" % ("rows == 500",
                           (old_aqi == 500).sum(), (df.target_aqi == 500).sum()))
print("%-28s %12.1f %12.1f" % ("mean AQI", old_aqi.mean(), df.target_aqi.mean()))
print("%-28s %12.1f %12.1f" % ("std AQI", old_aqi.std(), df.target_aqi.std()))

old_d = old_aqi.diff().abs()
new_d = df.target_aqi.diff().abs()
print("%-28s %12.1f %12.1f" % ("median hourly jump",
                               old_d.median(), new_d.median()))
print("%-28s %12.1f %12.1f" % ("mean hourly jump",
                               old_d.mean(), new_d.mean()))
print("%-28s %12d %12d" % ("jumps > 100 AQI/hr",
                           (old_d > 100).sum(), (new_d > 100).sum()))

corr_old = pd.Series(old_aqi).corr(pd.Series(old_aqi).shift(-24))
corr_new = df.target_aqi.corr(df.target_day1)
print("%-28s %12.3f %12.3f" % ("corr(today, tomorrow)", corr_old, corr_new))

print("\nDominant pollutant breakdown:")
print(df["dominant_pollutant"].value_counts())

df.to_csv(OUT_CSV, index=False)
print("\nSaved %s  (%d rows x %d cols)" % (OUT_CSV, len(df), df.shape[1]))