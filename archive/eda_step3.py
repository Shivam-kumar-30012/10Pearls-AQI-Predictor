"""
EDA Step 3: Clean the dataset

Fixes we identified:
1. Remove rows with -9999 (sensor gaps - only ~4 rows total)
2. Remove duplicate rows (boundary overlap in backfill - 6 rows)

Saves a new, clean CSV - we keep the original untouched as a backup.
"""

import pandas as pd

df = pd.read_csv("aqi_historical_dataset.csv")
print(f"Starting rows: {len(df)}")

# ---- Fix 1: Remove rows with -9999 in ANY numeric column ----
# We check pm10, no2, and o3 specifically since that's where we found them
bad_value_cols = ["pm10", "no2", "o3"]

for col in bad_value_cols:
    before = len(df)
    df = df[df[col] != -9999]
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with -9999 in '{col}'")

# ---- Fix 2: Remove duplicate rows ----
before = len(df)
df = df.drop_duplicates()
removed = before - len(df)
print(f"Removed {removed} duplicate rows")

print(f"\nFinal rows: {len(df)}")

# ---- Save as a NEW file - never overwrite raw data ----
df.to_csv("aqi_historical_dataset_clean.csv", index=False)
print("Saved to aqi_historical_dataset_clean.csv")