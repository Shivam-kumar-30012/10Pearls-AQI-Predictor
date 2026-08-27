"""
Train Model 1: predicts AQI 24 hours (1 day) ahead.
Uses aqi_training_data_with_weather.csv (37 columns including weather).
Trains and compares 4 models:
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
Evaluates with RMSE, MAE, R^2 and saves the best.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor
# ---------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------
df = pd.read_csv("aqi_training_data_with_weather.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
# ---------------------------------------------------------------
# Step 2: Define features (X) and target (y)
# ---------------------------------------------------------------
feature_cols = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week", "is_weekend",
    # Pollutants
    "pm10", "no2", "o3", "co", "so2", "nh3", "pm2_5",
    # Weather (NEW)
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds",
    # AQI lags
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    # Rolling means
    "aqi_rolling_3d", "aqi_rolling_7d",
]
target_col = "target_day1"
X = df[feature_cols]
y = df[target_col]
print(f"Features: {len(feature_cols)} columns")
print(f"Target: {target_col}")

# Step 3: Train/test split (chronological - no shuffle)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
# ---------------------------------------------------------------
# Step 4: Train and evaluate models
# ---------------------------------------------------------------
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
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, 500)  # AQI bounded 0-500
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae  = mean_absolute_error(y_test, predictions)
    r2   = r2_score(y_test, predictions)
    results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R^2:  {r2:.4f}")
# ---------------------------------------------------------------
# Step 5: Compare and pick the best
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print("\n" + "=" * 50)
print("COMPARISON (sorted by R^2)")
print("=" * 50)
print(results_df.to_string(index=False))
best_model_name = results_df.iloc[0]["model"]
print(f"\nBest model (by R^2): {best_model_name}")
# ---------------------------------------------------------------
# Step 6: Save the best model to disk
# ---------------------------------------------------------------
best_model = models[best_model_name]
joblib.dump(best_model, "model_day1.pkl")
print(f"Saved best model to model_day1.pkl")