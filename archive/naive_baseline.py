import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("aqi_training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

split = int(len(df) * 0.8)

# Naive predictor: tomorrow's AQI ≈ today's 3-day rolling average
naive_pred = df["aqi_rolling_3d"].iloc[split:]
actual = df["target_day1"].iloc[split:]

print("=== NAIVE BASELINE (predict tomorrow = today's 3d rolling avg) ===")
print(f"  RMSE: {np.sqrt(mean_squared_error(actual, naive_pred)):.2f}")
print(f"  MAE:  {mean_absolute_error(actual, naive_pred):.2f}")
print(f"  R²:   {r2_score(actual, naive_pred):.4f}")

# Even simpler: tomorrow = same hour yesterday
yesterday_pred = df["target_aqi"].iloc[split:].shift(24).dropna()
yesterday_actual = df["target_day1"].iloc[split:].iloc[24:]

print("\n=== YESTERDAY-AT-SAME-TIME BASELINE ===")
print(f"  RMSE: {np.sqrt(mean_squared_error(yesterday_actual, yesterday_pred)):.2f}")
print(f"  MAE:  {mean_absolute_error(yesterday_actual, yesterday_pred):.2f}")
print(f"  R²:   {r2_score(yesterday_actual, yesterday_pred):.4f}")