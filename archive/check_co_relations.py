"""
Check how strongly each feature correlates with target_day1.
Helps diagnose why our models are underperforming.
"""
import pandas as pd

df = pd.read_csv("aqi_training_data.csv")

feature_cols = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm10", "no2", "o3", "co", "so2", "nh3", "pm2_5",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    "aqi_rolling_3d", "aqi_rolling_7d",
]

correlations = df[feature_cols + ["target_day1"]].corr()["target_day1"].sort_values(ascending=False)
print("Correlation with target_day1 (1.0 = perfect, 0 = none):")
print(correlations)