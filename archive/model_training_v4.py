"""
Model training v4 - serious attempt to break the 0.35 ceiling.

Key changes from v3:
1. AQI TREND features (NEW): is AQI rising/falling over 24h windows
2. AQI distribution features (NEW): max/min/median/std of last 24h
3. Persistence forecast features: assumes tomorrow = today, model learns to correct
4. Weather FORECAST PROXY (NEW): use the weather from 24h ago as a proxy
   for "tomorrow's weather" (the only historical signal we have)
5. Keep seasonal dummies from v3
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor

df = pd.read_csv("aqi_training_data_with_weather.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} rows")

# ---- Seasonal features ----
def get_season(month):
    if month in [12, 1, 2]: return "winter"
    elif month in [3, 4, 5]: return "summer"
    elif month in [6, 7, 8, 9]: return "monsoon"
    else: return "post_monsoon"

df["season"] = df["month"].apply(get_season)
season_dummies = pd.get_dummies(df["season"], prefix="season").astype(int)
df = pd.concat([df, season_dummies], axis=1)

# ---- NEW: AQI trend features (is AQI going up or down?) ----
df["aqi_trend_6h"]   = df["aqi_lag_1h"] - df["aqi_lag_6h"]
df["aqi_trend_12h"]  = df["aqi_lag_1h"] - df["aqi_lag_12h"]
df["aqi_trend_24h"]  = df["aqi_lag_1h"] - df["aqi_lag_24h"]
df["aqi_trend_48h"]  = df["aqi_lag_24h"] - df["aqi_lag_48h"]
df["aqi_trend_72h"]  = df["aqi_lag_24h"] - df["aqi_lag_72h"]

# ---- NEW: AQI distribution over last 24h ----
last_24h = df["target_aqi"].rolling(window=24, min_periods=12)
df["aqi_max_24h"]    = last_24h.max()
df["aqi_min_24h"]    = last_24h.min()
df["aqi_median_24h"] = last_24h.median()
df["aqi_std_24h"]    = last_24h.std()
df["aqi_range_24h"]  = df["aqi_max_24h"] - df["aqi_min_24h"]

# ---- NEW: Weather 24h-ago proxy ----
# We don't have tomorrow's weather, but we have weather 24h ago.
# If AQI is autocorrelated, the wind/humidity pattern 24h back
# tells us something about the air mass.
weather_cols_to_shift = ["temperature", "humidity", "pressure",
                          "wind_speed", "wind_deg", "clouds"]
for col in weather_cols_to_shift:
    df[f"{col}_lag24"] = df[col].shift(24)
    df[f"{col}_diff24"] = df[col] - df[f"{col}_lag24"]

# ---- NEW: Persistence forecast features ----
# "If nothing changes, tomorrow = today's 7-day rolling avg"
# Model can learn corrections to this naive baseline.
df["persistence_24h"] = df["aqi_rolling_7d"]   # baseline assumption
df["persistence_48h"] = df["aqi_rolling_7d"]   # 48h ahead (less reliable)
df["persistence_72h"] = df["aqi_rolling_7d"]   # 72h ahead

# ---- Features ----
feature_cols = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",

    # Season
    "season_winter", "season_summer", "monsoon" if "monsoon" in df.columns else "season_monsoon",
    "season_post_monsoon",
]
# Fix season names if needed
feature_cols = [c for c in feature_cols if c in df.columns] + [
    "season_winter", "season_summer", "season_monsoon", "season_post_monsoon"
]
feature_cols = list(dict.fromkeys(feature_cols))  # dedupe preserving order

feature_cols += [
    # Pollutants
    "pm10", "no2", "o3", "co", "so2", "nh3", "pm2_5",

    # Weather (current)
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds",

    # Weather 24h ago (NEW)
    "temperature_lag24", "humidity_lag24", "pressure_lag24",
    "wind_speed_lag24", "wind_deg_lag24", "clouds_lag24",
    "temperature_diff24", "humidity_diff24", "pressure_diff24",
    "wind_speed_diff24", "wind_deg_diff24", "clouds_diff24",

    # AQI lags
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",

    # AQI rolling
    "aqi_rolling_3d", "aqi_rolling_7d",

    # AQI trends (NEW)
    "aqi_trend_6h", "aqi_trend_12h", "aqi_trend_24h",
    "aqi_trend_48h", "aqi_trend_72h",

    # AQI distribution (NEW)
    "aqi_max_24h", "aqi_min_24h", "aqi_median_24h",
    "aqi_std_24h", "aqi_range_24h",

    # Persistence forecast (NEW)
    "persistence_24h", "persistence_48h", "persistence_72h",
]

target_col = "target_day1"

X = df[feature_cols]
y = df[target_col]
print(f"Features: {len(feature_cols)}")
print(f"Target: {target_col}")

# Drop NaN
before = len(X)
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]
print(f"After dropna: {len(X)} rows (dropped {before - len(X)})")

# Chronological split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---- Models ----
models = {
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    ),
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 500)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R^2: {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print("\n" + "=" * 50)
print("COMPARISON (sorted by R^2)")
print("=" * 50)
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_model = models[best_name]
joblib.dump(best_model, "model_day1_v4.pkl")

# Save feature list
import json
with open("feature_cols_v4.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\nBest model: {best_name} (R^2 = {results_df.iloc[0]['r2']:.4f})")
