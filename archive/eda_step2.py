"""
EDA Step 2: Investigate the -9999 problem and duplicates

Before fixing anything, we need to understand:
- How many rows have this bad -9999 value?
- Is it only in the 'o3' column, or elsewhere too?
- What do the duplicate rows actually look like?
"""

import pandas as pd

df = pd.read_csv("aqi_historical_dataset.csv")

print("=" * 50)
print("CHECKING EVERY NUMERIC COLUMN FOR -9999")
print("=" * 50)

# Get only the numeric columns (skip timestamp/city which are text)
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    bad_count = (df[col] == -9999).sum() # .sum() counts the number of True values.
    if bad_count > 0:
        print(f"  {col}: {bad_count} rows with -9999")

print("\n" + "=" * 50)
print("WHAT PERCENTAGE OF DATA IS AFFECTED?")
print("=" * 50)
o3_bad = (df["o3"] == -9999).sum()
total = len(df)
print(f"o3 bad rows: {o3_bad} out of {total} ({o3_bad/total*100:.2f}%)")

print("\n" + "=" * 50)
print("SAMPLE OF BAD ROWS (first 5)")
print("=" * 50)
print(df[df["o3"] == -9999].head())

print("\n" + "=" * 50)
print("DUPLICATE ROWS - WHAT DO THEY LOOK LIKE?")
print("=" * 50)
duplicates = df[df.duplicated(keep=False)]
print(duplicates.sort_values("timestamp"))