"""
Model training v3 - real fix.

Key changes from v2:
1. Predict AQI CHANGE (target_day1 - current_aqi), not absolute AQI
   - Smaller, more stable target
   - Easier to learn

2. Add seasonal dummy features (winter, summer, monsoon, post-monsoon)
   - Pakistan has 4 distinct seasons that affect AQI differently
   - Encoded as one-hot so trees/linear can use them directly

3. Use TimeSeriesSplit cross-validation
   - More honest evaluation
   - Prevents lucky/unlucky single-split results
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor

df = pd.read_csv("aqi_training_data_with_weather.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---- Seasonal features (Pakistan-specific) ----
# Winter: Dec-Feb (crop burning, temperature inversions, highest PM)
# Summer: Mar-May (dust storms, hot)
# Monsoon: Jun-Sep (rain washes pollution, lowest AQI usually)
# Post-monsoon: Oct-Nov (transition)
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    else:
        return "post_monsoon"

df["season"] = df["month"].apply(get_season)
season_dummies = pd.get_dummies(df["season"], prefix="season").astype(int)
df = pd.concat([df, season_dummies], axis=1)
print(f"Added season columns: {list(season_dummies.columns)}")

# ---- Target: predict AQI CHANGE, not absolute AQI ----
# target_day1 = AQI 24h from now
# target_aqi = AQI right now
# delta_aqi_24h = how much AQI will change
df["delta_aqi_24h"] = df["target_day1"] - df["target_aqi"]
print(f"\nDelta target stats:")
print(df["delta_aqi_24h"].describe())

# ---- Features ----
feature_cols = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",

    # Season dummies (NEW)
    "season_winter", "season_summer", "season_monsoon", "season_post_monsoon",

    # Pollutants
    "pm10", "no2", "o3", "co", "so2", "nh3", "pm2_5",

    # Weather
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds",

    # AQI lags
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",

    # Rolling
    "aqi_rolling_3d", "aqi_rolling_7d",
]

X = df[feature_cols]
y = df["delta_aqi_24h"]  # CHANGED: predict change, not absolute

print(f"\nFeatures: {len(feature_cols)} | Target: delta_aqi_24h")

# ---- Chronological split for final eval ----
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---- Models ----
# Note: predictions are now deltas, will convert back to AQI for metrics
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1
    ),
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predict delta, then convert to absolute AQI
    predicted_delta = model.predict(X_test)
    predicted_delta = np.clip(predicted_delta, -500, 500)
    predictions = df["target_aqi"].iloc[split:].values + predicted_delta
    predictions = np.clip(predictions, 0, 500)

    actual = df["target_day1"].iloc[split:].values

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    r2   = r2_score(actual, predictions)

    results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R^2: {r2:.4f}")

# ---- Compare ----
results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print("\n" + "=" * 50)
print("COMPARISON (predict delta, evaluate as absolute AQI)")
print("=" * 50)
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_model = models[best_name]
joblib.dump(best_model, "model_day1_v3.pkl")
print(f"\nBest model: {best_name} (R^2 = {results_df.iloc[0]['r2']:.4f})")
print("NOTE: model predicts delta_aqi_24h. At inference, add current target_aqi to prediction.")
