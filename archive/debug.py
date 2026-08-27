import pandas as pd
import numpy as np

df = pd.read_csv("aqi_training_data.csv")

# 1. Check timestamp is sorted (chronological order matters!)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("Timestamp sorted?", df["timestamp"].is_monotonic_increasing)
print("Date range:", df["timestamp"].min(), "→", df["timestamp"].max())

# 2. Check for any remaining NaN/nulls
print("\nNulls per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 3. Check target distribution
print("\nTarget (target_day1) stats:")
print(df["target_day1"].describe())

# 4. CRITICAL: check if test set has a different distribution than train
# This is the #1 reason tree models fail
split_idx = int(len(df) * 0.8)
train_target = df["target_day1"].iloc[:split_idx]
test_target  = df["target_day1"].iloc[split_idx:]

print("\n=== TRAIN target distribution ===")
print(train_target.describe())
print("\n=== TEST target distribution ===")
print(test_target.describe())

# 5. Check feature variance
feature_cols = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",
    "pm10", "no2", "o3", "co", "so2", "nh3", "pm2_5",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    "aqi_rolling_3d", "aqi_rolling_7d",
]
print("\n=== Feature stats (look for weird scales) ===")
print(df[feature_cols].describe().T[["mean", "std", "min", "max"]])

# 6. Look at a few raw rows - is the data even continuous hour-by-hour?
print("\n=== Time gaps in data ===")
gaps = df["timestamp"].diff().dt.total_seconds() / 3600
print("Hour gaps > 1:", (gaps > 1).sum())
print("Largest gap (hours):", gaps.max())