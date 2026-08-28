"""
PIPELINE STEP 08 - Hourly collector

Runs every hour via GitHub Actions. Fetches the latest pollutant and
weather readings, builds features, and saves them to TWO places:

    1. data/processed/features.csv   (local backup, written first)
    2. Hopsworks feature group v3    (the source of truth)

Why two destinations
--------------------
Hopsworks' free tier is unreliable - reads time out, and the Spark job
that materialises writes is often busy. The local CSV means data
collection never stops, and the pipeline can still be demonstrated.

Each destination is checked INDEPENDENTLY against its own newest
timestamp, so a retry after a failed upload does not duplicate rows in
the CSV. Order matters: the CSV is written FIRST, so a failed upload
never loses data.

Why it fetches a RANGE rather than one hour
-------------------------------------------
Gaps happen: upstream outages, GitHub's best-effort scheduler, a failed
run. Asking for "everything since the last stored hour" is 1 hour on a
normal run and 300 after an outage - same code either way. OpenWeather
keeps its history, so nothing is lost by collecting late.

Run:  python pipeline/08_hourly_collector.py
      python pipeline/08_hourly_collector.py --dry-run
"""

import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline"))

import config

config.require_secrets("OPENWEATHER_API_KEY", "HOPSWORKS_API_KEY")

import hopsworks

# Reuse step 04's feature functions rather than copying them, so the
# live features can never drift from the training features.
import importlib
build_features = importlib.import_module("04_build_features")

POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

CONTEXT_HOURS = 200          # one week of lags (168h) plus margin
UPLOAD_CHUNK = 100           # rows per insert call; 500 caused materialisation failures
READ_RETRIES = 3
READ_WAIT = 20               # seconds, doubles each attempt


# ---------------------------------------------------------------
# Reading
# ---------------------------------------------------------------
def read_with_retries(label, func):
    """
    Run a read, retrying with a growing pause. Returns None if all
    attempts fail.

    Reads are worth retrying because their failures are usually
    transient timeouts. Writes are NOT handled here - see upload().
    """
    wait = READ_WAIT
    for attempt in range(1, READ_RETRIES + 1):
        try:
            return func()
        except Exception as error:
            print("  %s failed (attempt %d/%d): %s"
                  % (label, attempt, READ_RETRIES, type(error).__name__))
            if attempt == READ_RETRIES:
                return None
            print("  retrying in %ds..." % wait)
            time.sleep(wait)
            wait *= 2
    return None


def csv_latest_time():
    """Newest hour in the local CSV, or None if it does not exist."""
    if not config.FEATURES_CSV.exists():
        return None
    df = pd.read_csv(config.FEATURES_CSV, usecols=["timestamp"])
    return pd.to_datetime(df["timestamp"], utc=True).max()


def store_latest_time(feature_group):
    """
    Newest hour in Hopsworks, and whether the read succeeded.

    Returns (timestamp, True) on success, (None, False) if unreachable.
    The caller needs to know which: an empty store and an unreadable
    store need different handling.
    """
    def _read():
        df = feature_group.select(["timestamp"]).read()
        if df is None or len(df) == 0:
            return pd.NaT
        return pd.to_datetime(df["timestamp"], utc=True).max()

    result = read_with_retries("Hopsworks read", _read)
    if result is None:
        return None, False
    if pd.isna(result):
        return None, True          # readable but empty
    return result, True


# ---------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------
def fetch_pollutants(start, end):
    params = {
        "lat": config.CITY_LAT,
        "lon": config.CITY_LON,
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "appid": config.OPENWEATHER_API_KEY,
    }
    r = requests.get(POLLUTION_URL, params=params, timeout=60)
    r.raise_for_status()

    rows = []
    for reading in r.json().get("list", []):
        c = reading["components"]

        # OpenWeather uses -9999 to mean "sensor failed". Convert to
        # missing here so it never reaches a calculation.
        def clean(v):
            return None if v == -9999 else v

        rows.append({
            "timestamp": datetime.fromtimestamp(reading["dt"], tz=timezone.utc),
            "city": config.CITY_NAME,
            "pm2_5": clean(c.get("pm2_5")), "pm10": clean(c.get("pm10")),
            "no2": clean(c.get("no2")), "o3": clean(c.get("o3")),
            "co": clean(c.get("co")), "so2": clean(c.get("so2")),
            "nh3": clean(c.get("nh3")),
        })
    return pd.DataFrame(rows)


def fetch_weather(start, end):
    """
    Open-Meteo forecast endpoint with past_days. The archive endpoint
    used in step 02 lags several days behind, so it cannot supply the
    current hour.
    """
    days_back = max(1, (datetime.now(timezone.utc) - start).days + 1)
    days_back = min(days_back, 92)

    params = {
        "latitude": config.CITY_LAT,
        "longitude": config.CITY_LON,
        "hourly": ("temperature_2m,relative_humidity_2m,pressure_msl,"
                   "wind_speed_10m,wind_direction_10m,cloud_cover,"
                   "precipitation"),
        "past_days": days_back,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["pressure_msl"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_deg": hourly["wind_direction_10m"],
        "clouds": hourly["cloud_cover"],
        "precipitation": hourly["precipitation"],
    })
    return df[(df.timestamp >= start) & (df.timestamp <= end)]


# ---------------------------------------------------------------
# Writing
# ---------------------------------------------------------------
def append_to_csv(fresh, csv_latest):
    """
    Append only rows newer than what the CSV already holds, then
    de-duplicate on timestamp as a second line of defence.
    """
    if csv_latest is not None:
        fresh = fresh[fresh.timestamp > csv_latest]

    if fresh.empty:
        print("  CSV already up to date - nothing appended")
        return 0

    if not config.FEATURES_CSV.exists():
        # First run on a fresh machine (or a CI runner). Start the file
        # from what we just built rather than failing.
        fresh.to_csv(config.FEATURES_CSV, index=False)
        print("  created %s with %d rows"
              % (config.FEATURES_CSV.name, len(fresh)))
        return len(fresh)

    existing = pd.read_csv(config.FEATURES_CSV)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)

    fresh = fresh[[c for c in existing.columns if c in fresh.columns]]

    combined = (pd.concat([existing, fresh], ignore_index=True)
                  .drop_duplicates(subset="timestamp", keep="last")
                  .sort_values("timestamp")
                  .reset_index(drop=True))

    combined.to_csv(config.FEATURES_CSV, index=False)
    print("  appended %d rows -> %d total (newest %s)"
          % (len(fresh), len(combined), combined.timestamp.max()))
    return len(fresh)


def upload(feature_group, rows):
    """
    Insert rows into Hopsworks, in chunks, distinguishing a BUSY
    materialisation job from a real failure.

    Hopsworks writes rows to a buffer immediately, then runs a Spark
    job to move them into the queryable table. If that job is already
    running it emits:

        UserWarning: Materialization job is already running,
        aborting new execution.

    An earlier version treated that as a failure and retried three
    times, twenty seconds apart - which was doubly wrong. The rows had
    already arrived, and the job it was waiting for takes minutes, so
    every retry hit the same busy job and re-sent the same rows.

    Deferred materialisation is not data loss. Hopsworks processes the
    buffer when the current job finishes, so we report it plainly and
    move on.

    Returns True if every chunk was accepted.
    """
    chunks = [rows.iloc[i:i + UPLOAD_CHUNK]
              for i in range(0, len(rows), UPLOAD_CHUNK)]
    deferred = False

    for i, chunk in enumerate(chunks, 1):
        label = "chunk %d/%d (%d rows)" % (i, len(chunks), len(chunk))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                feature_group.insert(chunk,
                                     write_options={"wait_for_job": False})

            busy = any("already running" in str(w.message) for w in caught)
            if busy:
                deferred = True
                print("  %s accepted (materialisation deferred)" % label)
            else:
                print("  %s accepted" % label)

        except Exception as error:
            message = str(error)
            if "already running" in message:
                # Same condition, raised instead of warned
                deferred = True
                print("  %s accepted (materialisation deferred)" % label)
                continue
            print("  %s FAILED: %s" % (label, type(error).__name__))
            print("    %s" % message[:300])
            return False

    if deferred:
        print("\n  Rows are in Hopsworks. A materialisation job was already")
        print("  running, so they become queryable once it finishes.")
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    dry_run = "--dry-run" in sys.argv
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    csv_latest = csv_latest_time()
    print("Local CSV newest hour:  %s" % csv_latest)

    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(config.FEATURE_GROUP_NAME,
                              version=config.FEATURE_GROUP_VERSION)

    store_latest, store_ok = store_latest_time(fg)
    if store_ok:
        print("Hopsworks newest hour:  %s" % store_latest)
    else:
        print("Hopsworks unreadable - will still update the local CSV")

    # Fetch from whichever destination is furthest behind, so one run
    # can bring both up to date.
    candidates = [t for t in (csv_latest, store_latest) if t is not None]
    if not candidates:
        print("\nNeither destination has data. Run steps 01-05 first.")
        sys.exit(1)

    fetch_after = min(candidates)
    gap_hours = int((now - fetch_after).total_seconds() // 3600)
    print("\nFetching from %s (%d hours)" % (fetch_after, gap_hours))

    if gap_hours < 1:
        print("Both destinations are up to date.")
        return

    print("\nFetching pollutants...")
    poll = fetch_pollutants(fetch_after + timedelta(hours=1), now)
    print("  %d readings" % len(poll))

    print("Fetching weather...")
    weather = fetch_weather(fetch_after + timedelta(hours=1), now)
    print("  %d readings" % len(weather))

    if poll.empty:
        print("\nNo new pollutant data yet.")
        return

    new_raw = poll.merge(weather, on="timestamp", how="inner")
    print("  %d hours with both" % len(new_raw))
    if new_raw.empty:
        print("\nNothing to add.")
        return

    # ---- Context for the lag features ----
    #
    # Prefer the local CSV: it is instant, and it is never behind the
    # store because we write it before every upload attempt.
    #
    # On a GitHub Actions runner there IS no CSV - the runner is wiped
    # after each run and data/ is gitignored - so we fall back to the
    # feature store. That is the correct source there anyway; the CSV
    # was always a local convenience.
    if config.FEATURES_CSV.exists():
        history = pd.read_csv(config.FEATURES_CSV)
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
        history = history.sort_values("timestamp")
        print("\n  context source: local CSV (%d rows)" % len(history))
    else:
        if not store_ok:
            print("\nNo local CSV and Hopsworks is unreadable - cannot")
            print("build lag features. Nothing written.")
            sys.exit(1)
        print("\n  no local CSV - reading context from Hopsworks...")
        history = read_with_retries("context read", lambda: fg.read())
        if history is None:
            print("  context read failed. Nothing written.")
            sys.exit(1)
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
        history = history.sort_values("timestamp")
        print("  context source: Hopsworks (%d rows)" % len(history))

    raw_cols = (["timestamp", "city"]
                + build_features.POLLUTANTS + build_features.WEATHER)
    raw_cols = [c for c in raw_cols if c in history.columns]

    # Context must be the rows immediately BEFORE the fetched window,
    # not simply the newest rows in the file. Those differ whenever the
    # CSV is ahead of the store. Taking .tail() would grab rows from
    # after the window, leaving the first fetched hours with no history
    # behind them - their 24-hour AQI averages would be computed from
    # 2 or 3 readings instead of 24, giving values far too low. That is
    # a silent corruption: the numbers look plausible but are wrong.
    context = history[history["timestamp"] <= fetch_after].tail(CONTEXT_HOURS)
    print("  context: %d rows (%s -> %s)"
          % (len(context),
             context["timestamp"].min() if len(context) else "none",
             context["timestamp"].max() if len(context) else "none"))

    if len(context) < 168:
        print("  WARNING: only %d hours of context (168 needed for the"
              % len(context))
        print("  weekly lag). Rows without full history are dropped below.")

    combined = (pd.concat([context[raw_cols], new_raw], ignore_index=True)
                  .drop_duplicates(subset="timestamp", keep="last")
                  .sort_values("timestamp")
                  .reset_index(drop=True))

    print("\nBuilding features over %d rows..." % len(combined))
    df = build_features.build_hourly_grid(combined)
    df = build_features.add_aqi(df)
    df = build_features.add_time_features(df)
    df = build_features.add_lag_features(df)
    df = build_features.add_weather_features(df)

    if "boundary_layer_height" in df.columns:
        df = df.drop(columns=["boundary_layer_height"])
    df["city"] = df["city"].ffill().bfill()

    fresh = df[df.timestamp > fetch_after].dropna(subset=["aqi"]).copy()

    # A row whose weekly lag is missing was built without enough
    # history, so its AQI and rolling features cannot be trusted.
    if "aqi_lag_168h" in fresh.columns:
        before = len(fresh)
        fresh = fresh.dropna(subset=["aqi_lag_168h"])
        if before != len(fresh):
            print("  dropped %d rows with insufficient history"
                  % (before - len(fresh)))

    print("  %d new rows built" % len(fresh))
    if fresh.empty:
        print("\nNo complete new rows.")
        return

    print("  AQI range %.0f - %.0f | latest %.0f"
          % (fresh.aqi.min(), fresh.aqi.max(), fresh.aqi.iloc[-1]))

    if dry_run:
        print("\n--dry-run: writing nothing.")
        print(fresh[["timestamp", "aqi", "dominant_pollutant"]]
              .tail(10).to_string(index=False))
        return

    # ---- 1. Local CSV first, so nothing is lost if the upload fails ----
    print("\nUpdating local CSV...")
    append_to_csv(fresh, csv_latest)

    # ---- 2. Then Hopsworks, using ITS own last timestamp ----
    if not store_ok:
        print("\nSkipping upload - Hopsworks was unreadable this run.")
        print("Data is safe in the CSV. The next run will upload it.")
        return

    to_upload = (fresh if store_latest is None
                 else fresh[fresh.timestamp > store_latest])
    if to_upload.empty:
        print("\nHopsworks already up to date.")
        return

    stored_cols = [c for c in history.columns if c in to_upload.columns]
    to_upload = to_upload[stored_cols].copy()
    if "dominant_pollutant" in to_upload.columns:
        to_upload["dominant_pollutant"] = (
            to_upload["dominant_pollutant"].fillna("unknown"))

    # Match the store's column types exactly.
    #
    # hour/day/month come out as int32 when the context was read from
    # Hopsworks, but int64 when it came from the CSV - pandas infers
    # differently from each source. The feature group was created from
    # CSV data, so it expects bigint, and an upload was rejected on the
    # runner with "expected type: 'bigint', derived from input: 'int'".
    #
    # Forcing the widest numeric types makes the upload identical
    # whichever source the context came from.
    for col in to_upload.columns:
        if col in ("timestamp", "dominant_pollutant", "city"):
            continue
        if pd.api.types.is_integer_dtype(to_upload[col]):
            to_upload[col] = to_upload[col].astype("int64")
        elif pd.api.types.is_float_dtype(to_upload[col]):
            to_upload[col] = to_upload[col].astype("float64")

    print("\nUploading %d rows to Hopsworks..." % len(to_upload))
    if not upload(fg, to_upload):
        print("\nUpload failed. Data is in the CSV; the next run retries.")
        sys.exit(1)

    print("\nDone. timestamp is the primary key, so any re-sent hour")
    print("updates in place rather than duplicating.")


if __name__ == "__main__":
    main()