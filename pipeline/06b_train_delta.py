"""
EXPERIMENT - Delta-target models

Separate from the main pipeline. Everything it writes is prefixed
`delta_`, so deleting those files returns the project to its current
state. 06_train.py and the registered v2 models are untouched.

The problem this addresses
--------------------------
The current models produce almost flat forecasts: about a 5-point swing
across 24 hours. The real data moves far more than that:

    median 24h change      14 points
    median daily swing     18 points
    last 72 hours          92 -> 131 -> 108  (39 points)

So the models under-react. The cause is that Ridge minimises squared
error, and aqi_lag_1h correlates ~1.0 with the target - so predicting
close to the current value is the safest way to keep RMSE low. The
model optimises the metric at the cost of being useful. This is
regression to the mean.

What this tries instead
-----------------------
Predict the CHANGE from the current AQI rather than the level:

    target = aqi(t + h) - aqi(t)

Then add the current AQI back at prediction time. The model can no
longer hedge toward "about the same as now", because "the same" is now
zero - it has to commit to a direction and a size.

Note: an earlier session tried a delta target and got R2 0.16. That was
on the CORRUPTED AQI, where the target had spurious 500s injected into
it, so the test was invalid. Worth retrying on the correct target.

How it is judged
----------------
R2 alone will not answer this, because the current models score well
while under-reacting. So we also measure SWING - how much the
predictions move across a block compared with how much reality moves.
A model that scores slightly lower but swings realistically is the
better forecaster.

Run:  python pipeline/06b_train_delta.py
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

def load_data():
    """
    Read the features, preferring whichever source is FRESHEST.

    This was written as a local experiment and originally read the CSV
    directly. On a GitHub runner there is no CSV - the runner is wiped
    each run and data/ is gitignored - so it needs the same Hopsworks
    path as 06_train.py.
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

    # The CSV can be ahead of the store when an upload failed
    if source.startswith("Hopsworks") and config.FEATURES_CSV.exists():
        local = pd.read_csv(config.FEATURES_CSV)
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        if local["timestamp"].max() > df["timestamp"].max():
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

    We also return current_aqi for each example, so predictions can be
    converted back to absolute AQI for a fair comparison.
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
        delta = future - current          # <-- the change, not the level

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
    How much do the predictions move across a 24-hour block, compared
    with how much reality moves?

    This is the metric that matters here. R2 rewards staying close to
    the current value, so a model can score well while producing a flat
    line. Swing measures whether it actually commits.
    """
    df = pd.DataFrame({"row": rows, "hour": hours,
                       "pred": predicted_aqi, "actual": actual_aqi})
    # Group by source row: each group is one 24-hour forecast block
    g = df.groupby("row")
    pred_swing = (g["pred"].max() - g["pred"].min())
    actual_swing = (g["actual"].max() - g["actual"].min())
    return pred_swing.median(), actual_swing.median()


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

    # The true future AQI, for comparing on the same scale as the
    # main models
    y_abs_te = cur_te + yd_te

    print("  train %d | test %d\n" % (len(yd_tr), len(yd_te)))

    scores = []

    print("  Baseline:")
    # Persistence means predicting zero change
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
        pred_delta = algo.predict(X_te)
        pred_abs = cur_te + pred_delta          # convert back
        scores.append(measure(name, y_abs_te, pred_abs, time.time() - t0))
        trained[name] = algo
        predictions[name] = pred_abs

    # ---- The swing comparison ----
    print("\n  SWING across each 24-hour block (median):")
    _, actual_swing = swing_check(predictions[list(predictions)[0]],
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
        print("\n  No delta model beat the baseline.")
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
    print("COMPARISON WITH CURRENT MODELS")
    print("=" * 62)

    current_path = config.MODELS_DIR / "metrics.json"
    current = json.loads(current_path.read_text()) if current_path.exists() else {}

    print("  %-6s %-12s %-12s %-10s %-10s" %
          ("", "R2 now", "R2 delta", "swing", "actual"))
    for key in ["day1", "day2", "day3"]:
        if key not in results:
            continue
        now_r2 = current.get(key, {}).get("r2", float("nan"))
        print("  %-6s %-12.4f %-12.4f %-10.1f %-10.1f"
              % (key, now_r2, results[key]["r2"],
                 results[key]["predicted_swing"],
                 results[key]["actual_swing"]))

    print("\nKeep the delta models only if they swing realistically AND")
    print("hold R2 close to the current ones. Nothing has been replaced -")
    print("the files are all prefixed delta_.")


if __name__ == "__main__":
    main()