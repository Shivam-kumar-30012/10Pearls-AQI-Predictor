"""
PIPELINE STEP 06 - Train the forecast models

We train 3 models:

    model_day1.pkl  ->  predicts hours  1 to 24 ahead
    model_day2.pkl  ->  predicts hours 25 to 48 ahead
    model_day3.pkl  ->  predicts hours 49 to 72 ahead

Together they cover every hour of the next 3 days (72 predictions).

For each one we try 3 algorithms (Ridge, Random Forest, XGBoost) and
keep whichever scores best.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# Which hours each model is responsible for
HORIZONS = {
    1: (1, 24),
    2: (25, 48),
    3: (49, 72),
}

# Features multiplied by hours_ahead. See make_training_data for why.
INTERACT_WITH = ["aqi", "aqi_trend_24h", "aqi_trend_72h",
                 "aqi_roll_std_1d", "aqi_vs_7d"]


def load_data():
    """
    Read the features, preferring whichever source is FRESHEST.

    Two things make this necessary:

    1. Hopsworks' free tier limits row reads. When the quota is spent,
       reads fail while writes and the model registry keep working - so
       we can still train from the local CSV and register the result.

    2. The local CSV can be AHEAD of the store. Step 08 writes the CSV
       before attempting its upload, so rows whose upload failed exist
       locally but not in Hopsworks. Training on the store in that case
       would silently use older data than we have.

    So: try Hopsworks, then check whether the CSV is newer, and use the
    fresher one. The source is printed clearly, so the Actions log shows
    where any given model version came from.

    Pass --local to force the CSV when developing offline.
    """
    source = None

    if "--local" in sys.argv:
        print("--local flag: reading the local CSV")
        df = pd.read_csv(config.FEATURES_CSV)
        source = "local CSV (forced)"
    else:
        try:
            import hopsworks
            print("Reading from Hopsworks feature store...")
            project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
            store = project.get_feature_store()
            group = store.get_feature_group(config.FEATURE_GROUP_NAME,
                                            version=config.FEATURE_GROUP_VERSION)
            df = group.read()
            source = "Hopsworks %s v%d" % (config.FEATURE_GROUP_NAME,
                                           config.FEATURE_GROUP_VERSION)
        except Exception as error:
            print("  Hopsworks read failed: %s" % type(error).__name__)
            print("  Falling back to the local CSV.")
            if not config.FEATURES_CSV.exists():
                print("\n  No local CSV either. Cannot train.")
                sys.exit(1)
            df = pd.read_csv(config.FEATURES_CSV)
            source = "local CSV (Hopsworks unreachable)"

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # If we read the store, check the CSV is not further ahead
    if source.startswith("Hopsworks") and config.FEATURES_CSV.exists():
        local = pd.read_csv(config.FEATURES_CSV)
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        if local["timestamp"].max() > df["timestamp"].max():
            behind = local["timestamp"].max() - df["timestamp"].max()
            print("  Local CSV is %.0f hours ahead of the store - using it"
                  % (behind.total_seconds() / 3600))
            df = local.sort_values("timestamp").reset_index(drop=True)
            source = "local CSV (ahead of store)"

    print("\nDATA SOURCE: %s" % source)
    print("Loaded %d rows | %s -> %s\n"
          % (len(df), df.timestamp.min().date(), df.timestamp.max().date()))

    # Warn if the data looks stale - worth knowing before spending
    # 15 minutes training on it.
    age = (pd.Timestamp.now(tz="UTC") - df.timestamp.max()).total_seconds() / 3600
    if age > 48:
        print("WARNING: newest row is %.0f hours old. The hourly collector"
              % age)
        print("may not be running. Training will continue anyway.\n")

    return df


def extra_feature_names(feature_names):
    """
    Names of the columns make_training_data adds, in the same order.

    Kept in one place so the saved feature_cols.json and the prediction
    script can never disagree with what was actually trained on.
    """
    return (["hours_ahead", "sqrt_hours_ahead"]
            + ["%s_x_hours" % c for c in INTERACT_WITH if c in feature_names])


def make_training_data(df, feature_names, first_hour, last_hour):
    """
    Turn the feature table into training examples.

    One row of the table becomes several training examples - one for
    each hour we want to predict. We add 'hours_ahead' as an extra
    feature so the model knows how far into the future it is looking.

    Example: the row for Monday 9am becomes
        (Monday 9am features, hours_ahead=1)  -> AQI at 10am
        (Monday 9am features, hours_ahead=2)  -> AQI at 11am
        and so on up to hours_ahead=24

    INTERACTION FEATURES
    --------------------
    A first version used hours_ahead on its own, and the model almost
    ignored it - Ridge gave it a weight of -0.2 against -145 for
    aqi_lag_1h. The forecast came out as a flat line: the same AQI for
    every hour of a day.

    The reason is that hours_ahead alone can only shift every prediction
    up or down by a fixed amount. It cannot express "the current value
    matters less the further out you go", which is the thing that
    actually happens.

    So we multiply hours_ahead by the features that should decay with
    lead time:

        aqi x hours_ahead        - today's level matters less at +24h
        trend x hours_ahead      - a rising trend compounds over time
        volatility x hours_ahead - unstable air drifts further

    and add sqrt(hours_ahead), because forecast error grows roughly with
    the square root of lead time rather than linearly.
    """
    features = df[feature_names].to_numpy(dtype=np.float32)
    aqi = df["aqi"].to_numpy(dtype=np.float32)
    total_rows = len(df)

    # Positions of the features we want to interact with hours_ahead
    interact_idx = [feature_names.index(c) for c in INTERACT_WITH
                    if c in feature_names]

    all_X = []
    all_y = []
    all_hours = []
    all_row_numbers = []

    for hours_ahead in range(first_hour, last_hour + 1):
        # The answer for row i is the AQI 'hours_ahead' rows later.
        # The last few rows have no answer, so we stop early.
        usable = total_rows - hours_ahead

        X = features[:usable]
        y = aqi[hours_ahead:hours_ahead + usable]

        # Skip any example with a missing value
        complete = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        all_X.append(X[complete])
        all_y.append(y[complete])
        all_hours.append(np.full(complete.sum(), hours_ahead, dtype=np.float32))
        all_row_numbers.append(np.flatnonzero(complete))

    # Stack everything into one big table
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    hours = np.concatenate(all_hours)
    row_numbers = np.concatenate(all_row_numbers)

    # Build the extra columns that depend on hours_ahead
    extra = [hours, np.sqrt(hours)]
    for idx in interact_idx:
        extra.append(X[:, idx] * hours)

    X = np.column_stack([X] + extra)

    print("  %d rows x %d hours = %d training examples (%d features)"
          % (total_rows, last_hour - first_hour + 1, len(y), X.shape[1]))
    return X, y, hours, row_numbers


def measure(name, actual, predicted, seconds=None):
    """Calculate RMSE, MAE and R2, print them, and return them."""
    predicted = np.clip(predicted, 0, 700)   # AQI can't be negative

    result = {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }

    timing = "" if seconds is None else "  (%.0fs)" % seconds
    print("    %-16s RMSE %6.2f | MAE %6.2f | R2 %.4f%s"
          % (name, result["rmse"], result["mae"], result["r2"], timing))
    return result


def train_one_horizon(df, day, feature_names):
    """Train and compare all 3 algorithms for one day."""
    first_hour, last_hour = HORIZONS[day]

    print("=" * 60)
    print("DAY +%d   (predicting hours %d to %d ahead)"
          % (day, first_hour, last_hour))
    print("=" * 60)

    X, y, hours, row_numbers = make_training_data(
        df, feature_names, first_hour, last_hour)

    # Split by DATE, oldest 80% to train and newest 20% to test.
    # We split on the original row number, not the expanded rows. This
    # keeps all 24 hours of a day together. Otherwise the model could
    # train on 3pm Tuesday and be tested on 4pm Tuesday, which is
    # almost the same data - the score would look better than it is.
    split_point = int(len(df) * config.TEST_SPLIT_RATIO)
    is_train = row_numbers < split_point
    is_test = ~is_train

    X_train, X_test = X[is_train], X[is_test]
    y_train, y_test = y[is_train], y[is_test]
    hours_test = hours[is_test]

    print("  train: %d examples | test: %d examples" % (len(y_train), len(y_test)))
    print("  train ends %s | test starts %s\n"
          % (df["timestamp"].iloc[split_point - 1].date(),
             df["timestamp"].iloc[split_point].date()))

    scores = []

    #  Baseline: just guess that AQI stays the same
    # Any real model must beat this, otherwise it has learned nothing.
    print("  Baseline:")
    aqi_column = feature_names.index("aqi")
    scores.append(measure("Persistence", y_test, X_test[:, aqi_column]))

    # --- The 3 algorithms ---
    print("\n  Models:")
    algorithms = {
        # Ridge needs all features on a similar scale, so we
        # put a StandardScaler in front of it.
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),

        # max_samples=0.3 means each tree looks at 30% of the rows.
        # With ~900,000 rows that is still plenty, and it makes
        # training about 3x faster.
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=50,
            max_samples=0.3,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
    }

    trained = {}
    predictions = {}

    for name, algorithm in algorithms.items():
        start = time.time()
        algorithm.fit(X_train, y_train)
        predicted = algorithm.predict(X_test)

        scores.append(measure(name, y_test, predicted, time.time() - start))
        trained[name] = algorithm
        predictions[name] = predicted

    #  Pick the winner
    table = pd.DataFrame(scores).sort_values("r2", ascending=False)
    print("\n" + table.to_string(index=False))

    winner = table.iloc[0]["model"]
    winner_r2 = float(table.iloc[0]["r2"])
    baseline_r2 = float(table[table["model"] == "Persistence"]["r2"].iloc[0])

    if winner == "Persistence":
        print("\n  WARNING: no model beat the baseline. Nothing saved.")
        return None

    print("\n  Winner: %s (R2 %.4f, which is %+.4f better than the baseline)"
          % (winner, winner_r2, winner_r2 - baseline_r2))

    # Save it
    filename = config.MODELS_DIR / ("model_day%d.pkl" % day)
    joblib.dump(trained[winner], filename)
    print("  Saved %s" % filename.name)

    # How accuracy changes with distance
    accuracy_by_hour = {}
    for h in range(first_hour, last_hour + 1):
        this_hour = hours_test == h
        if this_hour.sum() > 30:
            accuracy_by_hour[h] = round(
                float(r2_score(y_test[this_hour],
                               predictions[winner][this_hour])), 4)

    print("  R2 at hour %d: %.3f  ->  hour %d: %.3f\n"
          % (first_hour, accuracy_by_hour[first_hour],
             last_hour, accuracy_by_hour[last_hour]))

    # How much the forecast VARIES across the block. A model that
    # predicts the same number for all 24 hours is not forecasting,
    # it is copying the current value - this makes that visible.
    spread = float(np.std(predictions[winner]))
    print("  Prediction spread (std): %.1f AQI" % spread)

    return {
        "day": day,
        "hours": [first_hour, last_hour],
        "winner": winner,
        "r2": winner_r2,
        "baseline_r2": baseline_r2,
        "improvement": round(winner_r2 - baseline_r2, 4),
        "prediction_spread": round(spread, 2),
        "all_scores": scores,
        "r2_by_hour": accuracy_by_hour,
    }


def main():
    df = load_data()

    # Use every numeric column as a feature, except the ones that
    # aren't really data (text and identifiers).
    skip = {"timestamp", "city", "dominant_pollutant"}
    feature_names = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in skip]
    extras = extra_feature_names(feature_names)
    print("Using %d base features + %d hours_ahead terms = %d total\n"
          % (len(feature_names), len(extras), len(feature_names) + len(extras)))

    all_results = {}
    for day in [1, 2, 3]:
        result = train_one_horizon(df, day, feature_names)
        if result:
            all_results["day%d" % day] = result

    # The prediction script rebuilds its input from this list, so it must
    # describe every column in the order make_training_data produced them,
    # including the interaction terms.
    with open(config.MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_names + extras, f, indent=2)

    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, result in all_results.items():
        print("  %-6s %-14s R2 %.4f  (baseline %.4f, improvement %+.4f, spread %.1f)"
              % (key, result["winner"], result["r2"],
                 result["baseline_r2"], result["improvement"],
                 result["prediction_spread"]))
    print("\nSaved models, metrics.json and feature_cols.json")


if __name__ == "__main__":
    main()