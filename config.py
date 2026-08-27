
# Central configuration for the Pearls AQI Predictor.

# Every script imports from here instead of hardcoding values, so a
# setting only ever needs changing in one place for keeping things good and structured lets gooooo. Secrets are read from
# the .env file (which is gitignored) rather than living in source code.

# Usage in any pipeline script:
#     from config import CITY_LAT, CITY_LON, OPENWEATHER_API_KEY, RAW_DIR

import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env from the project root

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")



# SECRETS

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


def require_secrets(*names):
    
    # Call at the top of any script that needs credentials, so it fails
    # immediately with a clear message rather than midway through a long and mid-way breaking
    
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise RuntimeError(
            "Missing in .env: %s\n"
            "Copy .env.example to .env and fill in your keys."
            % ", ".join(missing)
        )



# LOCATION

CITY_NAME = "Ghotki"
CITY_LAT = 28.0089
CITY_LON = 69.3159
TIMEZONE = "UTC"          # everything stays UTC end to end



# DATA COLLECTION

# OpenWeather's air pollution history begins 2020-11-27
BACKFILL_START_DATE = "2020-11-27"



# PATHS/ folders 


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)    # it will create the directories if it doesn't exit

# Raw (untouched API output)
POLLUTANTS_RAW_CSV = RAW_DIR / "pollutants_historical.csv"
WEATHER_RAW_CSV = RAW_DIR / "weather_historical.csv"
HOURLY_LIVE_CSV = RAW_DIR / "hourly_live.csv"

# Processed
MERGED_RAW_CSV = PROCESSED_DIR / "raw_merged.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"



# FEATURE STORE

# v3 of data that is uploaded on hopeswork.
FEATURE_GROUP_NAME = "aqi_ghotki_features"
FEATURE_GROUP_VERSION = 3
FEATURE_VIEW_NAME = "aqi_ghotki_view"
MODEL_REGISTRY_NAME = "aqi_ghotki_ridge"


# 
# MODELLING
# 

FORECAST_HORIZONS = [1, 2, 3]     # days ahead
TEST_SPLIT_RATIO = 0.8            # chronological, never shuffled
RANDOM_SEED = 42

# EPA averaging windows (hours) - required before AQI lookup
PM_AVERAGING_HOURS = 24           # PM2.5, PM10
GAS_AVERAGING_HOURS = 8           # CO, O3


#
# ALERTS
# 

ALERT_AQI_THRESHOLD = 150         # "Unhealthy" and above
HAZARDOUS_AQI_THRESHOLD = 300     # "Hazardous"


if __name__ == "__main__":    # just cheching by self runnign  the file
    print("Project root :", PROJECT_ROOT)
    print("City         : %s (%.4f, %.4f)" % (CITY_NAME, CITY_LAT, CITY_LON))
    print("Backfill from:", BACKFILL_START_DATE)
    print("Feature group: %s v%d" % (FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION))
    print()
    print("OpenWeather key loaded:", bool(OPENWEATHER_API_KEY))
    print("Hopsworks key loaded  :", bool(HOPSWORKS_API_KEY))