"""
PIPELINE STEP 11 - Explain the models with SHAP

Answers "why did the model predict that number?" by splitting each
prediction into the contribution of every feature. The contributions
sum exactly to the prediction, so nothing is hand-waved.

Two kinds of output
-------------------
GLOBAL   - averaged over many predictions, which features matter most
LOCAL    - one specific forecast, why it came out as it did

Why this is worth having
------------------------
A model that cannot be explained is hard to trust and harder to debug.
If day_of_week turned out to be a top feature, that would be a sign
something is wrong. It also separates two things the dashboard could
otherwise conflate: the DOMINANT POLLUTANT is arithmetic (EPA says AQI
equals your worst pollutant, and that is PM2.5 today), while SHAP is
about the model's reasoning for a future value.

A note on the two model types
-----------------------------
Day +1 is XGBoost, where SHAP's TreeExplainer gives genuinely
informative structure. Days 2 and 3 are Ridge - for a linear model the
contribution is just coefficient x feature value, which is valid but
tells you less. Both are computed; the report can compare them.

Writes: notebooks/figures/09_shap_importance_day{1,2,3}.png
        notebooks/figures/10_shap_current_forecast.png
        data/shap_summary.json   (for the dashboard)

Run:  python pipeline/11_explain.py
      python pipeline/11_explain.py --local
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

import config

FIGURES = ROOT / "notebooks" / "figures"
SUMMARY_JSON = ROOT / "data" / "shap_summary.json"

# Enough rows to be representative without waiting minutes for
# KernelExplainer-style work. TreeExplainer is exact and fast, so this
# is really a limit on the Ridge models.
SAMPLE_ROWS = 2000
TOP_N = 12

plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "font.size": 9,
})

INK = "#1a1a1a"
ACCENT = "#8DC63F"
MUTED = "#c9c9c4"


# ---------------------------------------------------------------
# Loading
# ---------------------------------------------------------------
def load_features():
    """Same source logic as training: store first, CSV if that fails."""
    if "--local" not in sys.argv:
        try:
            import hopsworks
            print("Reading features from Hopsworks...")
            project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
            fg = project.get_feature_store().get_feature_group(
                config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
            df = fg.read()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as error:
            print("  Hopsworks failed (%s) - using the local CSV"
                  % type(error).__name__)

    print("Reading %s" % config.FEATURES_CSV.name)
    df = pd.read_csv(config.FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_model(day):
    """
    The saved model plus the feature list it expects.

    Day +1 uses the delta model, so its predictions are a CHANGE from
    the current AQI rather than a level. That matters for reading the
    local explanation: the base value is near zero, not near 100.
    """
    delta = day == 1
    model_path = config.MODELS_DIR / (
        "delta_model_day%d.pkl" % day if delta else "model_day%d.pkl" % day)
    cols_path = config.MODELS_DIR / (
        "feature_cols_delta.json" if delta else "feature_cols.json")

    if not model_path.exists():
        print("  %s missing - run step 06 first" % model_path.name)
        return None, None, None

    with open(cols_path) as f:
        cols = json.load(f)
    return joblib.load(model_path), cols, delta


# ---------------------------------------------------------------
# Building model inputs
# ---------------------------------------------------------------
def build_matrix(df, cols, hours_ahead):
    """
    Rebuild the model's input exactly as steps 06 and 09 do.

    Most columns come straight from the table. Three kinds are computed:
    hours_ahead itself, its square root, and the interaction terms that
    multiply a feature by hours_ahead. Order follows feature_cols.json,
    because a wrong order does not error - it silently produces
    meaningless explanations.
    """
    out = {}
    for name in cols:
        if name == "hours_ahead":
            out[name] = np.full(len(df), float(hours_ahead))
        elif name == "sqrt_hours_ahead":
            out[name] = np.full(len(df), float(np.sqrt(hours_ahead)))
        elif name.endswith("_x_hours"):
            base = name[: -len("_x_hours")]
            out[name] = df[base].to_numpy(dtype=np.float32) * hours_ahead
        else:
            out[name] = df[name].to_numpy(dtype=np.float32)
    return pd.DataFrame(out, columns=cols)


def pretty(name):
    """Feature names a reader can follow without the codebase open."""
    lookup = {
        "aqi": "AQI now", "aqi_lag_1h": "AQI 1 h ago",
        "aqi_lag_3h": "AQI 3 h ago", "aqi_lag_6h": "AQI 6 h ago",
        "aqi_lag_12h": "AQI 12 h ago", "aqi_lag_24h": "AQI 24 h ago",
        "aqi_lag_48h": "AQI 48 h ago", "aqi_lag_72h": "AQI 72 h ago",
        "aqi_lag_168h": "AQI 1 week ago",
        "aqi_roll_mean_1d": "AQI, 1-day mean",
        "aqi_roll_mean_3d": "AQI, 3-day mean",
        "aqi_roll_mean_7d": "AQI, 7-day mean",
        "aqi_roll_std_1d": "AQI volatility, 1 day",
        "aqi_trend_24h": "24 h trend", "aqi_trend_72h": "72 h trend",
        "aqi_trend_6h": "6 h trend", "aqi_vs_7d": "vs weekly average",
        "aqi_range_1d": "daily AQI range",
        "hours_ahead": "hours ahead", "sqrt_hours_ahead": "hours ahead (sqrt)",
        "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2", "o3": "ozone",
        "co": "CO", "so2": "SO2", "nh3": "ammonia",
        "temperature": "temperature", "humidity": "humidity",
        "pressure": "pressure", "wind_speed": "wind speed",
        "wind_u": "wind, east-west", "wind_v": "wind, north-south",
        "ventilation": "ventilation index", "calm_flag": "calm air",
        "temp_range_24h": "24 h temperature range",
        "pressure_trend_24h": "24 h pressure trend",
        "precip_24h": "rain, last 24 h", "precip_72h": "rain, last 72 h",
        "season_winter": "winter", "season_summer": "summer",
        "season_monsoon": "monsoon", "season_post_monsoon": "post-monsoon",
    }
    if name.endswith("_x_hours"):
        return pretty(name[: -len("_x_hours")]) + " x lead time"
    return lookup.get(name, name.replace("_", " "))


# ---------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------
def global_importance(day, model, cols, delta, sample, hours_ahead):
    X = build_matrix(sample, cols, hours_ahead)

    if not hasattr(model, "named_steps"):
        # A bare tree model (the delta horizons). TreeExplainer is exact
        # and fast. Checking the object rather than the horizon means a
        # future change of algorithm does not silently break this.
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
    else:
        # Ridge behind a StandardScaler. Explaining model.predict makes
        # SHAP fall back to permutation sampling, which took 100 seconds
        # a horizon. LinearExplainer on the regressor is exact and
        # instant; the scaler is a fixed linear map, so we scale the
        # inputs ourselves and the contributions stay correct.
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        X_scaled = pd.DataFrame(scaler.transform(X), columns=cols)
        explainer = shap.LinearExplainer(ridge, X_scaled)
        values = explainer.shap_values(X_scaled)

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh([pretty(cols[i]) for i in order], mean_abs[order],
            color=ACCENT, edgecolor=INK, linewidth=.5)
    ax.set_xlabel("mean absolute SHAP value  (AQI points)")
    kind = ("delta model, predicts the change"
            if delta else "absolute model, predicts the level")
    ax.set_title("Day +%d: what drives the forecast\n%s, %d hours ahead"
                 % (day, kind, hours_ahead), loc="left")
    fig.tight_layout()
    path = FIGURES / ("09_shap_importance_day%d.png" % day)
    fig.savefig(path)
    plt.close(fig)
    print("  saved %s" % path.name)

    return {cols[i]: float(mean_abs[i]) for i in np.argsort(mean_abs)[::-1][:TOP_N]}


# ---------------------------------------------------------------
# One prediction, explained
# ---------------------------------------------------------------
def explain_current(day, model, cols, delta, latest_row, hours_ahead,
                    background):
    """
    Why is the next forecast the number it is?

    For the delta model the prediction is a change, so the base value is
    near zero and the bars read as "this pushed the AQI up/down from
    where it is now". For the absolute models the base is the average
    AQI across the background sample.

    The background matters. SHAP measures each contribution RELATIVE to
    a reference set, so passing the single row being explained makes
    every contribution exactly zero - the chart came out completely
    empty. It has to be a spread of typical rows.
    """
    X = build_matrix(latest_row, cols, hours_ahead)
    X_bg = build_matrix(background, cols, hours_ahead)

    if not hasattr(model, "named_steps"):
        # No background here. TreeExplainer derives its own baseline
        # from the tree structure, and passing one switches it to a mode
        # that rejects XGBoost's categorical splits.
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)[0]
        base = float(explainer.expected_value)
    else:
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        bg_scaled = pd.DataFrame(scaler.transform(X_bg), columns=cols)
        row_scaled = pd.DataFrame(scaler.transform(X), columns=cols)
        explainer = shap.LinearExplainer(ridge, bg_scaled)
        values = explainer.shap_values(row_scaled)[0]
        base = float(explainer.expected_value)

    prediction = base + values.sum()
    order = np.argsort(np.abs(values))[::-1][:TOP_N][::-1]

    labels = [pretty(cols[i]) for i in order]
    contrib = values[order]
    colours = [ACCENT if v > 0 else "#d86050" for v in contrib]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(labels, contrib, color=colours, edgecolor=INK, linewidth=.5)
    ax.axvline(0, color=INK, linewidth=.9)
    ax.set_xlabel("contribution to the prediction  (AQI points)")

    if delta:
        current = float(latest_row["aqi"].iloc[0])
        ax.set_title(
            "Why the model expects AQI %.0f in %d hours\n"
            "currently %.0f; the bars show the predicted change of %+.1f"
            % (current + prediction, hours_ahead, current, prediction),
            loc="left")
    else:
        ax.set_title(
            "Why the model expects AQI %.0f in %d hours\n"
            "baseline %.0f, adjusted by the features below"
            % (prediction, hours_ahead, base), loc="left")

    fig.tight_layout()
    path = FIGURES / ("10_shap_current_day%d.png" % day)
    fig.savefig(path)
    plt.close(fig)
    print("  saved %s" % path.name)

    return {
        "base": round(base, 2),
        "prediction": round(float(prediction), 2),
        "top": [{"feature": cols[i], "label": pretty(cols[i]),
                 "value": round(float(X.iloc[0, i]), 2),
                 "contribution": round(float(values[i]), 2)}
                for i in np.argsort(np.abs(values))[::-1][:TOP_N]],
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_features()
    print("  %d rows\n" % len(df))

    numeric = df.select_dtypes(include=[np.number]).columns
    complete = df[df[numeric].notna().all(axis=1)]
    if complete.empty:
        print("No complete rows to explain.")
        return

    sample = complete.tail(SAMPLE_ROWS)
    latest = complete.tail(1)
    print("Explaining from %s (AQI %.0f)\n"
          % (latest["timestamp"].iloc[0], latest["aqi"].iloc[0]))

    summary = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
               "explained_from": latest["timestamp"].iloc[0].isoformat(),
               "current_aqi": int(round(latest["aqi"].iloc[0])),
               "horizons": {}}

    # The middle hour of each block: representative of that model rather
    # than its easiest or hardest case.
    for day, hours_ahead in [(1, 12), (2, 36), (3, 60)]:
        print("Day +%d (%d hours ahead)" % (day, hours_ahead))
        model, cols, delta = load_model(day)
        if model is None:
            continue

        importance = global_importance(day, model, cols, delta,
                                       sample, hours_ahead)
        # A few hundred rows is enough of a reference and keeps the
        # tree explainer fast.
        local = explain_current(day, model, cols, delta, latest,
                                hours_ahead, sample.tail(300))

        summary["horizons"]["day%d" % day] = {
            "model": "XGBoost (delta)" if delta else "Ridge",
            "hours_ahead": hours_ahead,
            "global_importance": importance,
            "current_explanation": local,
        }
        print()

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=1)
    print("Saved %s" % SUMMARY_JSON.name)

    print("\n" + "=" * 58)
    print("TOP DRIVERS")
    print("=" * 58)
    for key, block in summary["horizons"].items():
        print("\n%s (%s):" % (key, block["model"]))
        for name, score in list(block["global_importance"].items())[:5]:
            print("  %-26s %6.2f" % (pretty(name), score))


if __name__ == "__main__":
    main()