"""
PIPELINE STEP 09 - Make predictions

Produces a 72-hour AQI forecast: one prediction for every hour of the
next three days.

How it works
------------
The three models each cover a 24-hour block and take `hours_ahead` as
a feature, so each one is called 24 times with a different value:

    model_day1  ->  hours  1..24
    model_day2  ->  hours 25..48
    model_day3  ->  hours 49..72

The input is the most recent complete row of features. Every prediction
in a block uses that same row - only `hours_ahead` changes.

Why the feature order matters
-----------------------------
A model expects its 70 inputs in exactly the order it was trained on.
Give it the same numbers in a different order and it will not error -
it will just return nonsense, treating your temperature value as wind
speed. So we read feature_cols.json (saved next to the model) and build
the input to match it exactly.

Where things are loaded from
----------------------------
Models come from the Hopsworks Model Registry, newest version first, so
the dashboard automatically picks up whatever the daily training run
last produced. Features come from the feature store, falling back to
the local CSV. Both fall back to local files if Hopsworks is
unreachable - its free tier limits row reads while leaving the registry
available.

Run:  python pipeline/09_predict.py
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config
from aqi_calculator import aqi_category

HORIZONS = {1: (1, 24), 2: (25, 48), 3: (49, 72)}

# A DELTA model predicts the CHANGE from the current AQI rather than the
# future AQI itself; the caller adds the current value back.
#
# Only day+1 uses one. An XGBoost delta model scored R2 0.9317 against
# 0.9296 for the absolute Ridge, with MAE down from 10.15 to 8.28 and a
# forecast swing of 7.8 AQI against reality's 11 - much closer than the
# absolute model's 5.
#
# Days 2 and 3 stay absolute: for a LINEAR model, predicting (y - x)
# with x among the features is mathematically equivalent to predicting
# y, so Ridge scored identically either way (0.7158 vs 0.7158). XGBoost
# was clearly worse at those horizons.
#
# Which type a horizon uses is read from the REGISTRY (model_type.json,
# written by step 07) rather than hardcoded here. A model should declare
# what it is - otherwise every consumer keeps its own list and they
# drift apart the first time a horizon changes type.
DELTA_FALLBACK = {1}          # used only if the registry is unreachable


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------
def load_features(verbose=True):
    """Most recent features. Hopsworks first, then the local CSV."""
    try:
        import hopsworks
        if verbose:
            print("Reading features from Hopsworks...")
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
        fg = project.get_feature_store().get_feature_group(
            config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        df = fg.read()
        source = "Hopsworks feature store"
    except Exception as error:
        if verbose:
            print("  Hopsworks unavailable (%s) - using local CSV"
                  % type(error).__name__)
        df = pd.read_csv(config.FEATURES_CSV)
        source = "local CSV"

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # The local CSV can be AHEAD of the store. Step 08 writes the CSV
    # before attempting its upload, so any rows whose upload failed
    # exist locally but not in Hopsworks. Predicting from stale
    # conditions would be worse than using a local file, so we use
    # whichever source is fresher.
    if source != "local CSV" and config.FEATURES_CSV.exists():
        local = pd.read_csv(config.FEATURES_CSV)
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        if local["timestamp"].max() > df["timestamp"].max():
            behind = (local["timestamp"].max() - df["timestamp"].max())
            if verbose:
                print("  local CSV is %.0f hours ahead of the store - using it"
                      % (behind.total_seconds() / 3600))
            df = local.sort_values("timestamp").reset_index(drop=True)
            source = "local CSV (ahead of store)"

    return df, source


def load_local_delta(day, verbose=True):
    """Load the delta model and its feature list from disk."""
    model = joblib.load(config.MODELS_DIR / ("delta_model_day%d.pkl" % day))
    with open(config.MODELS_DIR / "feature_cols_delta.json") as f:
        feature_names = json.load(f)

    metrics = {}
    path = config.MODELS_DIR / "metrics_delta.json"
    if path.exists():
        saved = json.loads(path.read_text()).get("day%d" % day, {})
        winner = saved.get("winner")
        scores = next((s for s in saved.get("all_scores", [])
                       if s["model"] == winner), {})
        metrics = {"r2": saved.get("r2", 0),
                   "rmse": scores.get("rmse", 0),
                   "mae": scores.get("mae", 0),
                   "baseline_r2": saved.get("baseline_r2", 0)}
    if verbose:
        print("  day+%d: local delta model (R2 %.4f)"
              % (day, metrics.get("r2", float("nan"))))
    return model, feature_names, metrics, "local delta file", True


def load_model(day, verbose=True):
    """
    Load one model, newest registry version first.

    Returns (model, feature_names, metrics, description).

    Always takes the HIGHEST version number rather than a fixed one, so
    a model retrained overnight is picked up on the next dashboard load
    with no code change.
    """
    try:
        import hopsworks
        project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
        registry = project.get_model_registry()

        candidates = registry.get_models("aqi_ghotki_day%d" % day)
        if not candidates:
            raise RuntimeError("no versions registered")

        newest = max(candidates, key=lambda m: m.version)
        folder = Path(newest.download())

        model = joblib.load(folder / "model.pkl")
        with open(folder / "feature_cols.json") as f:
            feature_names = json.load(f)

        # The model declares its own type. Fall back to the description
        # tag for models registered before model_type.json existed.
        type_path = folder / "model_type.json"
        if type_path.exists():
            is_delta = json.loads(type_path.read_text()).get("is_delta", False)
        else:
            is_delta = "[DELTA]" in (newest.description or "")

        metrics = newest.training_metrics or {}
        where = "registry v%d" % newest.version
        if verbose:
            print("  day+%d: %s, %s (R2 %.4f)"
                  % (day, where, "delta" if is_delta else "absolute",
                     metrics.get("r2", float("nan"))))
        return model, feature_names, metrics, where, is_delta

    except Exception as error:
        if verbose:
            print("  day+%d: registry unavailable (%s) - using local file"
                  % (day, type(error).__name__))

        # Offline we cannot read the registry's model_type.json, so fall
        # back to the known list of delta horizons.
        if day in DELTA_FALLBACK:
            return load_local_delta(day, verbose)

        model = joblib.load(config.MODELS_DIR / ("model_day%d.pkl" % day))
        with open(config.MODELS_DIR / "feature_cols.json") as f:
            feature_names = json.load(f)

        metrics = {}
        metrics_path = config.MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                saved = json.load(f).get("day%d" % day, {})
            winner = saved.get("winner")
            scores = next((s for s in saved.get("all_scores", [])
                           if s["model"] == winner), {})
            metrics = {
                "r2": saved.get("r2", 0),
                "rmse": scores.get("rmse", 0),
                "mae": scores.get("mae", 0),
                "baseline_r2": saved.get("baseline_r2", 0),
            }
        return model, feature_names, metrics, "local file", day in DELTA_FALLBACK


# ---------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------
def build_input(latest_row, feature_names, hours_ahead):
    """
    One input row for the model.

    Most names are read straight from the latest row. Three kinds are
    computed here instead, because they depend on how far ahead we are
    predicting rather than on the data:

        hours_ahead        - the lead time itself
        sqrt_hours_ahead   - error grows roughly with its square root
        X_x_hours          - X multiplied by hours_ahead

    The interaction terms are what stop the forecast being a flat line.
    Without them hours_ahead can only shift every prediction by a fixed
    amount; with them the model can learn that the current AQI matters
    less, and a rising trend matters more, the further out you go.

    Order matters absolutely. feature_cols.json was written by step 06
    in the order the model was trained on, and we follow it exactly - a
    wrong order does not error, it silently returns nonsense.
    """
    values = []
    for name in feature_names:
        if name == "hours_ahead":
            values.append(float(hours_ahead))
        elif name == "sqrt_hours_ahead":
            values.append(float(np.sqrt(hours_ahead)))
        elif name.endswith("_x_hours"):
            base = name[:-len("_x_hours")]
            values.append(float(latest_row.get(base, np.nan)) * hours_ahead)
        else:
            values.append(float(latest_row.get(name, np.nan)))
    return np.array(values, dtype=np.float32).reshape(1, -1)


def forecast(verbose=True):
    """Returns (predictions dataframe, info dictionary)."""
    df, data_source = load_features(verbose)

    # Newest row that has every feature the models need. A row with a
    # missing lag cannot be used as an input.
    numeric = df.select_dtypes(include=[np.number]).columns
    complete = df[df[numeric].notna().all(axis=1)]
    if complete.empty:
        raise RuntimeError("No complete rows available to predict from.")

    latest = complete.iloc[-1]
    now = latest["timestamp"]

    if verbose:
        print("\nData source: %s" % data_source)
        print("Predicting from: %s (current AQI %.0f)"
              % (now, latest["aqi"]))
        print("\nLoading models:")

    rows = []
    model_info = {}

    for day, (first_hour, last_hour) in HORIZONS.items():
        model, feature_names, metrics, where, is_delta = load_model(day, verbose)
        model_info["day%d" % day] = {
            "source": where,
            "type": "delta" if is_delta else "absolute",
            "r2": metrics.get("r2"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        }

        for hours_ahead in range(first_hour, last_hour + 1):
            X = build_input(latest, feature_names, hours_ahead)
            predicted = float(model.predict(X)[0])

            # A delta model returns the CHANGE from the current AQI,
            # so add the current value back to get an absolute forecast.
            if is_delta:
                predicted += float(latest["aqi"])

            predicted = float(np.clip(predicted, 0, 700))

            label, colour = aqi_category(predicted)
            rows.append({
                "timestamp": now + timedelta(hours=hours_ahead),
                "hours_ahead": hours_ahead,
                "day": day,
                "aqi": round(predicted),
                "category": label,
                "colour": colour,
            })

    predictions = pd.DataFrame(rows)

    info = {
        "predicted_from": now,
        "current_aqi": float(latest["aqi"]),
        "current_category": aqi_category(latest["aqi"])[0],
        "dominant_pollutant": latest.get("dominant_pollutant", "unknown"),
        "data_source": data_source,
        "models": model_info,
    }
    return predictions, info


def daily_summary(predictions):
    """
    Collapse the hourly forecast into one row per day.

    The daily average is taken from the hourly predictions rather than
    predicted separately, so the two can never disagree.
    """
    out = []
    for day in sorted(predictions["day"].unique()):
        block = predictions[predictions["day"] == day]
        mean_aqi = round(block["aqi"].mean())
        out.append({
            "day": day,
            "date": block["timestamp"].iloc[0].date(),
            "mean_aqi": mean_aqi,
            "min_aqi": int(block["aqi"].min()),
            "max_aqi": int(block["aqi"].max()),
            "category": aqi_category(mean_aqi)[0],
            "colour": aqi_category(mean_aqi)[1],
            "worst_hour": block.loc[block["aqi"].idxmax(), "timestamp"],
        })
    return pd.DataFrame(out)


def main():
    predictions, info = forecast()

    print("\n" + "=" * 58)
    print("FORECAST")
    print("=" * 58)
    print("From %s | current AQI %.0f (%s, driven by %s)"
          % (info["predicted_from"], info["current_aqi"],
             info["current_category"], info["dominant_pollutant"]))

    print("\nDaily summary:")
    summary = daily_summary(predictions)
    for _, row in summary.iterrows():
        print("  Day+%d  %s   avg %3d  (range %d-%d)  %s"
              % (row["day"], row["date"], row["mean_aqi"],
                 row["min_aqi"], row["max_aqi"], row["category"]))

    print("\nEvery 6 hours:")
    for _, row in predictions[predictions.hours_ahead % 6 == 0].iterrows():
        print("  +%2dh  %s  AQI %3d  %s"
              % (row["hours_ahead"],
                 row["timestamp"].strftime("%a %H:%M"),
                 row["aqi"], row["category"]))

    worst = predictions.loc[predictions["aqi"].idxmax()]
    if worst["aqi"] >= config.ALERT_AQI_THRESHOLD:
        print("\nALERT: AQI reaches %d (%s) at %s"
              % (worst["aqi"], worst["category"],
                 worst["timestamp"].strftime("%a %H:%M")))

    out_path = config.PROCESSED_DIR / "latest_forecast.csv"
    predictions.to_csv(out_path, index=False)
    print("\nSaved %s" % out_path)


if __name__ == "__main__":
    main()