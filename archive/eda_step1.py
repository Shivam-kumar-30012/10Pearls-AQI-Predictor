"""
EDA Step 1: Basic Overview

Before visualizing anything, we first check:
- How many rows/columns do we have?
- What data types are in each column?
- Are there any missing values?
- Are there any duplicate rows?
- What do the numbers generally look like (min/max/average)? or outlier
"""

import pandas as pd

# Load the CSV into a pandas DataFrame - think of this as
# an in-memory spreadsheet we can query and manipulate.
df = pd.read_csv("aqi_historical_dataset.csv")

print("=" * 50)
print("SHAPE (rows, columns)")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 50)
print(df.dtypes)

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("MISSING VALUES PER COLUMN")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY (numeric columns)")
print("=" * 50)
print(df.describe())