"""
Upload features to Hopsworks

Sends the feature table to the Hopsworks Feature Store as feature
group v3.

v2 used a broken AQI calculator that returned 500 for any value falling
between two breakpoints, corrupting 4.38% of rows. v2 is left in place
rather than deleted - being able to supersede bad data without erasing
the record is the main reason feature stores are versioned.

Every row is uploaded, including the ~1,900 with a NaN somewhere. The
feature group is a record of what was observed; each training run picks
its own usable rows, because a row that cannot be an INPUT may still be
needed as the ANSWER for an earlier row.


"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

config.require_secrets("HOPSWORKS_API_KEY")

import hopsworks


def main():
    print("Loading features...")
    df = pd.read_csv(config.FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print("  %d rows x %d columns" % (len(df), df.shape[1]))
    print("  range: %s -> %s"
          % (df.timestamp.min().date(), df.timestamp.max().date()))

    # Hopsworks column names must be lowercase and have no spaces.
    df.columns = [c.lower().replace(" ", "_").replace("-", "_")
                  for c in df.columns]

    # dominant_pollutant is text; keep it but make sure there are no
    # NaNs, which the store handles less predictably for strings.
    if "dominant_pollutant" in df.columns:
        df["dominant_pollutant"] = df["dominant_pollutant"].fillna("unknown")

    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    print("  project: %s" % project.name)

    print("\nCreating feature group %s v%d..."
          % (config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION))
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description=(
            "Hourly AQI features for Ghotki, Pakistan. AQI computed with "
            "EPA averaging windows (24h PM, 8h CO/O3) and truncation-based "
            "breakpoint lookup. Supersedes v2, which used an AQI "
            "implementation that returned 500 for between-breakpoint values."
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    print("Uploading (this takes a few minutes)...")
    # wait_for_job=False on purpose.
    #
    # Hopsworks writes the rows, then runs a Spark job to materialise
    # them. On the free tier that job's metrics reporter (Prometheus)
    # times out during shutdown and the job reports FAILED - even though
    # every row was written correctly. Waiting on it makes a successful
    # upload look like a failure.
    #
    # So we submit and verify by reading the row count back instead of
    # trusting the job status.
    fg.insert(df, write_options={"wait_for_job": False})

    print("\nUpload submitted: %d rows -> %s v%d"
          % (len(df), config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION))
    print("  v2 left untouched as a record of the earlier version.")
    print("\nGive it a minute, then verify with:")
    print("    python pipeline/05_upload_to_store.py --verify")


def verify():
    """Reads the feature group back and reports how many rows landed."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=config.HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(config.FEATURE_GROUP_NAME,
                              version=config.FEATURE_GROUP_VERSION)
    print("Reading (takes a minute)...")
    df = fg.read()
    print("\n%s v%d contains %d rows x %d columns"
          % (config.FEATURE_GROUP_NAME, config.FEATURE_GROUP_VERSION,
             len(df), df.shape[1]))
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        print("Range: %s -> %s" % (ts.min().date(), ts.max().date()))


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        main()