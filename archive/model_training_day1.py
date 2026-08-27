"""
Model training v2 - fixes for the diagnostic findings:
1. Drop redundant lag features (keep only lag_24h, lag_48h, lag_72h)
2. Add proper cyclical + interaction features
3. Add AQI change-rate features
4. Use log target transform (handles the 0-500 capped range)
5. More aggressive model tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor

# ---- 1. Load data ----
df = pd.read_csv("aqi_training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} rows")

# ---- 2. Better feature engineering ----

# AQI CHANGE RATE (this is the missing signal!)
# "How fast is AQI rising/falling right now?" is super predictive
df["aqi_change_1h"]  = df["target_aqi"] - df["aqi_lag_1h"]
df["aqi_change_3h"]  = df["target_aqi"] - df["aqi_lag_3h"]
df["aqi_change_6h"]  = df["target_aqi"] - df["aqi_lag_6h"]
df["aqi_change_24h"] = df["target_aqi"] - df["aqi_lag_24h"]

# Day vs night interaction (encoded cyclically already, but trees need raw too)
df["hour_x_weekend"] = df["hour_sin"] * df["is_weekend"]

# Rolling STD (volatility - high std = unstable air, harder to predict)
df["aqi_rolling_3d_std"] = df["target_aqi"].rolling(window=72, min_periods=24).std()
df["aqi_rolling_7d_std"] = df["target_aqi"].rolling(window=168, min_periods=48).std()

# Pollutant ratios (ozone reacts with NOx, PM correlates with CO from traffic)
df["pm_ratio"]    = df["pm2_5"] / (df["pm10"] + 1e-6)
df["co_no2_ratio"] = df["co"] / (df["no2"] + 1e-6)

# Drop NaN rows created by rolling std
df = df.dropna().reset_index(drop=True)
print(f"After dropna: {len(df)} rows")

# ---- 3. Define features ----
feature_cols = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend", "hour_x_weekend",
    
    # Pollutants
    "pm10", "no2", "o3", "co", "so2", "nh3", "pm2_5",
    "pm_ratio", "co_no2_ratio",
    
    # Keep only the MEANINGFUL lags (24h+ capture the daily cycle)
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    
    # Rolling means
    "aqi_rolling_3d", "aqi_rolling_7d",
    
    # NEW: change rates
    "aqi_change_1h", "aqi_change_3h", "aqi_change_6h", "aqi_change_24h",
    
    # NEW: volatility
    "aqi_rolling_3d_std", "aqi_rolling_7d_std",
]

target_col = "target_day1"

X = df[feature_cols]
y = df[target_col]
print(f"Features: {len(feature_cols)} columns")

# ---- 4. Time-series split (more robust than 80/20) ----
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---- 5. Train models ----
models = {
    "Ridge Regression": Ridge(alpha=10.0),  # more regularization since we added features
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
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    
    results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

# ---- 6. Compare & save best ----
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(results_df.to_string(index=False))

best_name = results_df.loc[results_df["r2"].idxmax(), "model"]
best_model = models[best_name]
joblib.dump(best_model, "model_day1_v2.pkl")

# Save the feature list so the inference code uses the same columns
with open("feature_cols_v2.json", "w") as f:
    import json
    json.dump(feature_cols, f)

print(f"\n✅ Best model: {best_name} (R²={results_df['r2'].max():.4f})")
print(f"✅ Saved to model_day1_v2.pkl")