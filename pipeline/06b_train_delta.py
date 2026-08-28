"""
PIPELINE STEP 06b - Delta-target model (day +1)

Trains a model that predicts the CHANGE in AQI rather than its level,
and saves it as delta_model_day1.pkl. Step 07 registers whichever file
each horizon actually uses.

Why a delta target
------------------
The absolute models produced almost flat forecasts - about a 5-point
swing across 24 hours, where the real data moves 11 or more. The cause
is that Ridge minimises squared error and aqi_lag_1h correlates ~1.0
with the target, so predicting close to the current value is the safest
way to keep RMSE low. The model optimises the metric at the cost of
being useful.

Predicting the change removes that hiding place: "about the same as
now" becomes zero, so the model has to commit to a direction and a
size. At prediction time the current AQI is added back.

What actually happened
----------------------
Only day +1 benefits. XGBoost on the delta target scored R2 0.9317
against 0.9296 for absolute Ridge, with MAE down from 10.15 to 8.28
and swing up from 5 to 7.8.

Days 2 and 3 do not. For a LINEAR model, predicting (y - x) with x
among the features is mathematically equivalent to predicting y, so
Ridge scored identically either way (0.7158 vs 0.7158). XGBoost was
clearly worse at those horizons (0.61 and 0.48). They keep their
absolute models.

An earlier session tried a delta target and got R2 0.16, concluding it
was hopeless. That was on the CORRUPTED AQI, where the target had
spurious 500s in it, so the test was invalid.

How it is judged
----------------
R2 alone cannot answer this, because the absolute models score well
while under-reacting. So SWING is also measured - how far predictions
move across a block compared with how far reality moves.

Run:  python pipeline/06b_train_delta.py
      python pipeline/06b_train_delta.py --local
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

HORIZONS = {1: (1, 24), 2: (25, 48), 3: (49, 72)}

INTERACT_WITH = ["aqi", "aqi_trend_24h", "aqi_trend_72h",
                 "aqi_roll_std_1d", "aqi_vs_7d"]

# Below this, something is wrong with the data source and training
# would produce a worse model than the one already registered.
MIN_ROWS = 5000


def load_data():
    """
    Read the features, preferring whichever source is FRESHEST.

    Same logic as 06_train.py. This originally read the CSV directly,
    which failed on a GitHub runner where data/ is gitignored.
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
            group = project.get_feature_store().get_feature_group(
                config.FEATURE_GROUP_NAME,
                version=config.FEATURE_GROUP_VERSION)
            df = group.read()
            source = "Hopsworks %s v%d" % (config.FEATURE_GROUP_NAME,
                                           config.FEATURE_GROUP_VERSION)
        except Exception as error:
            print("  Hopsworks read failed: %s" % type(error).__name__)
            if not config.FEATURES_CSV.exists():
                print("  No local CSV either. Cannot train.")
                sys.exit(1)
            print("  Falling back to the local CSV.")
            df = pd.read_csv(config.FEATURES_CSV)
            source = "local CSV (Hopsworks unreachable)"

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if source.startswith("Hopsworks") and config.FEATURES_CSV.exists():
        local = pd.read_csv(config.FEATURES_CSV)
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        # A newer timestamp is not enough on its own. On a CI runner the
        # collector creates the CSV from scratch, so it holds only the
        # few hours it just fetched - "ahead" in time while containing
        # 0.1% of the data. A run once trained on 42 rows this way.
        # Require it to be at least as complete as the store too.
        if (local["timestamp"].max() > df["timestamp"].max()
                and len(local) >= len(df)):
            print("  Local CSV is ahead of the store - using it")
            df = local.sort_values("timestamp").reset_index(drop=True)
            source = "local CSV (ahead of store)"

    print("\nDATA SOURCE: %s" % source)
    print("Loaded %d rows | %s -> %s\n"
          % (len(df), df.timestamp.min().date(), df.timestamp.max().date()))
    return df


def extra_feature_names(feature_names):
    return (["hours_ahead", "sqrt_hours_ahead"]
            + ["%s_x_hours" % c for c in INTERACT_WITH if c in feature_names])


def make_training_data(df, feature_names, first_hour, last_hour):
    """
    Same expansion as the main script, but the answer is the CHANGE
    from the current AQI rather than the future AQI itself.

    current_aqi is returned as well, so predictions can be converted
    back to absolute AQI and compared on the same scale.
    """
    features = df[feature_names].to_numpy(dtype=np.float32)
    aqi = df["aqi"].to_numpy(dtype=np.float32)
    total_rows = len(df)

    interact_idx = [feature_names.index(c) for c in INTERACT_WITH
                    if c in feature_names]

    all_X, all_y, all_hours, all_rows, all_current = [], [], [], [], []

    for hours_ahead in range(first_hour, last_hour + 1):
        usable = total_rows - hours_ahead

        X = features[:usable]
        current = aqi[:usable]
        future = aqi[hours_ahead:hours_ahead + usable]
        delta = future - current          # the change, not the level

        complete = ~np.isnan(X).any(axis=1) & ~np.isnan(delta)

        all_X.append(X[complete])
        all_y.append(delta[complete])
        all_current.append(current[complete])
        all_hours.append(np.full(complete.sum(), hours_ahead, dtype=np.float32))
        all_rows.append(np.flatnonzero(complete))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    hours = np.concatenate(all_hours)
    rows = np.concatenate(all_rows)
    current = np.concatenate(all_current)

    extra = [hours, np.sqrt(hours)]
    for idx in interact_idx:
        extra.append(X[:, idx] * hours)
    X = np.column_stack([X] + extra)

    print("  %d examples, %d features | delta: mean %.1f, std %.1f"
          % (len(y), X.shape[1], y.mean(), y.std()))
    return X, y, hours, rows, current


def swing_check(predicted_aqi, actual_aqi, hours, rows):
    """
    How far do the predictions move across a 24-hour block, compared
    with how far reality moves?

    R2 rewards staying close to the current value, so a model can score
    well while drawing a flat line. Swing measures whether it commits.
    """
    d = pd.DataFrame({"row": rows, "hour": hours,
                      "pred": predicted_aqi, "actual": actual_aqi})
    g = d.groupby("row")          # each group is one 24-hour forecast
    return ((g["pred"].max() - g["pred"].min()).median(),
            (g["actual"].max() - g["actual"].min()).median())


def measure(name, actual, predicted, seconds=None):
    predicted = np.clip(predicted, 0, 700)
    result = {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
    timing = "" if seconds is None else "  (%.0fs)" % seconds
    print("    %-22s RMSE %6.2f | MAE %6.2f | R2 %.4f%s"
          % (name, result["rmse"], result["mae"], result["r2"], timing))
    return result


def train_one(df, day, feature_names):
    first_hour, last_hour = HORIZONS[day]
    print("=" * 62)
    print("DAY +%d  (hours %d-%d)  -  DELTA TARGET" % (day, first_hour, last_hour))
    print("=" * 62)

    X, y_delta, hours, rows, current = make_training_data(
        df, feature_names, first_hour, last_hour)

    split = int(len(df) * config.TEST_SPLIT_RATIO)
    tr, te = rows < split, rows >= split

    X_tr, X_te = X[tr], X[te]
    yd_tr, yd_te = y_delta[tr], y_delta[te]
    cur_te = current[te]
    hours_te, rows_te = hours[te], rows[te]

    # The true future AQI, so scores compare with the absolute models
    y_abs_te = cur_te + yd_te

    print("  train %d | test %d\n" % (len(yd_tr), len(yd_te)))

    scores = []
    print("  Baseline:")
    scores.append(measure("Persistence", y_abs_te, cur_te))

    print("\n  Delta models (converted back to absolute AQI):")
    algorithms = {
        "Ridge (delta)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "XGBoost (delta)": XGBRegressor(
            n_estimators=600, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_SEED, n_jobs=-1, tree_method="hist"),
    }

    trained, predictions = {}, {}
    for name, algo in algorithms.items():
        t0 = time.time()
        algo.fit(X_tr, yd_tr)
        pred_abs = cur_te + algo.predict(X_te)      # convert back
        scores.append(measure(name, y_abs_te, pred_abs, time.time() - t0))
        trained[name] = algo
        predictions[name] = pred_abs

    print("\n  SWING across each 24-hour block (median):")
    _, actual_swing = swing_check(list(predictions.values())[0],
                                  y_abs_te, hours_te, rows_te)
    print("    %-22s %6.1f AQI  <- what really happens" % ("Actual", actual_swing))
    pers_swing, _ = swing_check(cur_te, y_abs_te, hours_te, rows_te)
    print("    %-22s %6.1f AQI" % ("Persistence", pers_swing))
    for name in predictions:
        s, _ = swing_check(predictions[name], y_abs_te, hours_te, rows_te)
        print("    %-22s %6.1f AQI" % (name, s))

    table = pd.DataFrame(scores).sort_values("r2", ascending=False)
    print("\n" + table.to_string(index=False))

    winner = table.iloc[0]["model"]
    if winner == "Persistence":
        print("\n  No delta model beat the baseline. Nothing saved.")
        return None

    best_swing, _ = swing_check(predictions[winner], y_abs_te, hours_te, rows_te)

    path = config.MODELS_DIR / ("delta_model_day%d.pkl" % day)
    joblib.dump(trained[winner], path)
    print("\n  Winner: %s (R2 %.4f, swing %.1f vs actual %.1f)"
          % (winner, table.iloc[0]["r2"], best_swing, actual_swing))
    print("  Saved %s" % path.name)

    return {
        "day": day,
        "hours": [first_hour, last_hour],
        "winner": winner,
        "r2": float(table.iloc[0]["r2"]),
        "baseline_r2": float(table[table.model == "Persistence"].r2.iloc[0]),
        "predicted_swing": round(float(best_swing), 1),
        "actual_swing": round(float(actual_swing), 1),
        "all_scores": scores,
    }


def main():
    df = load_data()

    # A run once trained on 42 rows because the CI runner's freshly
    # created CSV looked "newer" than the store. The resulting models
    # scored R2 -0.17 and were nearly registered over the good ones.
    if len(df) < MIN_ROWS:
        print("Only %d rows available - refusing to train." % len(df))
        print("Something is wrong with the data source. The currently")
        print("registered models stay live.")
        sys.exit(1)

    skip = {"timestamp", "city", "dominant_pollutant"}
    feature_names = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in skip]
    extras = extra_feature_names(feature_names)
    print("%d base + %d extra = %d features\n"
          % (len(feature_names), len(extras), len(feature_names) + len(extras)))

    results = {}
    for day in [1, 2, 3]:
        r = train_one(df, day, feature_names)
        if r:
            results["day%d" % day] = r
        print()

    with open(config.MODELS_DIR / "feature_cols_delta.json", "w") as f:
        json.dump(feature_names + extras, f, indent=2)
    with open(config.MODELS_DIR / "metrics_delta.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 62)
    print("COMPARISON WITH THE ABSOLUTE MODELS")
    print("=" * 62)

    path = config.MODELS_DIR / "metrics.json"
    absolute = json.loads(path.read_text()) if path.exists() else {}

    print("  %-6s %-12s %-12s %-10s %-10s"
          % ("", "R2 abs", "R2 delta", "swing", "actual"))
    for key in ["day1", "day2", "day3"]:
        if key not in results:
            continue
        abs_r2 = absolute.get(key, {}).get("r2", float("nan"))
        print("  %-6s %-12.4f %-12.4f %-10.1f %-10.1f"
              % (key, abs_r2, results[key]["r2"],
                 results[key]["predicted_swing"],
                 results[key]["actual_swing"]))

    print("\nStep 07 registers the delta model for day +1 only; days 2 and 3")
    print("keep their absolute models, which scored better there.")


if __name__ == "__main__":
    main()