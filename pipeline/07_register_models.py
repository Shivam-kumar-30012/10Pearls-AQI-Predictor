"""
PIPELINE STEP 07 - Register models in the Hopsworks Model Registry

Uploads each horizon's model along with everything needed to use it:
the model file, the feature list it expects, and its metrics.

Why the feature list travels with the model
-------------------------------------------
A model expects its inputs in exactly the order it was trained on. Give
it the same numbers in a different order and it does not error - it
returns nonsense, treating temperature as wind speed. Shipping
feature_cols.json alongside the model makes that impossible.

Absolute and delta models
-------------------------
Day+1 uses a DELTA model: it predicts the CHANGE from the current AQI
rather than the AQI itself, and the caller adds the current value back.
Days 2 and 3 use absolute models.

That distinction is recorded in the model's metadata rather than
hardcoded in the prediction script. A model should declare what it is -
otherwise every consumer has to keep its own list, and they drift apart
the first time a horizon switches type.

Run:  python pipeline/07_register_model.py
      python pipeline/07_register_model.py --list
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

config.require_secrets("HOPSWORKS_API_KEY")

import hopsworks

# Which horizons use a delta model, and where their files live.
# Matches DELTA_MODELS in 09_predict.py - but that script now reads the
# type back from the registry, so this is the single source of truth.
DELTA_HORIZONS = {1}

# Marker written into the description so consumers can detect the type
DELTA_TAG = "[DELTA]"


def model_files(day):
    """Returns (model_path, feature_cols_path, metrics_path, is_delta)."""
    if day in DELTA_HORIZONS:
        return (config.MODELS_DIR / ("delta_model_day%d.pkl" % day),
                config.MODELS_DIR / "feature_cols_delta.json",
                config.MODELS_DIR / "metrics_delta.json",
                True)
    return (config.MODELS_DIR / ("model_day%d.pkl" % day),
            config.MODELS_DIR / "feature_cols.json",
            config.MODELS_DIR / "metrics.json",
            False)


def register_one(registry, day):
    model_path, cols_path, metrics_path, is_delta = model_files(day)

    for path in (model_path, cols_path, metrics_path):
        if not path.exists():
            print("  day+%d: %s missing - skipping" % (day, path.name))
            return None

    with open(cols_path) as f:
        feature_cols = json.load(f)
    saved = json.loads(metrics_path.read_text()).get("day%d" % day, {})

    winner = saved.get("winner", "unknown")
    scores = next((s for s in saved.get("all_scores", [])
                   if s["model"] == winner), {})

    metrics = {
        "r2": saved.get("r2", 0),
        "rmse": scores.get("rmse", 0),
        "mae": scores.get("mae", 0),
        "baseline_r2": saved.get("baseline_r2", 0),
        "improvement_over_baseline": saved.get("improvement", 0),
    }

    first_hour, last_hour = saved.get("hours", [0, 0])

    kind = "delta" if is_delta else "absolute"
    description = (
        "%sGhotki AQI forecast, day+%d (hours %d-%d ahead). "
        "Algorithm: %s. Prediction type: %s. Takes hours_ahead as a "
        "feature, so one model covers the whole 24-hour block. "
        "%sTrained on feature group %s v%d."
        % (DELTA_TAG + " " if is_delta else "",
           day, first_hour, last_hour, winner, kind,
           "Returns the CHANGE from current AQI - add the current value "
           "to the prediction. " if is_delta else "",
           config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION)
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(model_path, tmp / "model.pkl")
        with open(tmp / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f, indent=2)
        with open(tmp / "metrics.json", "w") as f:
            json.dump(saved, f, indent=2)
        # An explicit file as well as the description tag, so consumers
        # do not have to parse prose to work out the model type.
        with open(tmp / "model_type.json", "w") as f:
            json.dump({"type": kind, "is_delta": is_delta,
                       "algorithm": winner,
                       "hours": [first_hour, last_hour]}, f, indent=2)

        name = "aqi_ghotki_day%d" % day
        print("  %s: %s, %s, R2 %.4f, %d features"
              % (name, winner, kind, metrics["r2"], len(feature_cols)))

        model = registry.python.create_model(
            name=name,
            metrics=metrics,
            description=description,
        )
        model.save(str(tmp))

    print("    registered as version %s" % model.version)
    return model.version


def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
    registry = project.get_model_registry()
    print("  project: %s\n" % project.name)

    print("Registering models:")
    versions = {}
    for day in config.FORECAST_HORIZONS:
        v = register_one(registry, day)
        if v:
            versions["day%d" % day] = v

    if not versions:
        print("\nNothing registered. Run steps 06 and 06b first.")
        return

    print("\nRegistered:")
    for name, version in versions.items():
        print("  aqi_ghotki_%s  version %s" % (name, version))
    print("\n09_predict.py loads the newest version of each and reads")
    print("model_type.json to decide whether to add the current AQI back.")


def list_models():
    """Show what is currently in the registry."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
    registry = project.get_model_registry()

    for day in config.FORECAST_HORIZONS:
        name = "aqi_ghotki_day%d" % day
        try:
            models = registry.get_models(name)
        except Exception:
            models = []

        if not models:
            print("\n%s: not registered" % name)
            continue

        print("\n%s:" % name)
        for m in sorted(models, key=lambda x: x.version):
            metrics = m.training_metrics or {}
            kind = "delta" if DELTA_TAG in (m.description or "") else "absolute"
            print("  v%-3s  %-9s R2 %-8s RMSE %-8s MAE %s"
                  % (m.version, kind,
                     round(metrics.get("r2", 0), 4),
                     round(metrics.get("rmse", 0), 2),
                     round(metrics.get("mae", 0), 2)))


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_models()
    else:
        main()