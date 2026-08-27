"""
EDA Step 5: Actual visualizations using our real data.

Creates 4 charts:
1. AQI trend over time (5 years)
2. Average AQI by hour of day
3. Average AQI by month
4. Boxplot to check for outliers across pollutants
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("aqi_from_hopsworks.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---------------------------------------------------------------
# CHART 1: AQI trend over time
# ---------------------------------------------------------------
# We resample to DAILY averages first - plotting all 48,654 hourly
# points directly would be an unreadable solid blob of color.
daily_avg = df.set_index("timestamp")["target_aqi"].resample("D").mean()

plt.figure(figsize=(14, 5))
plt.plot(daily_avg.index, daily_avg.values, linewidth=0.8)
plt.title("Ghotki: Daily Average AQI Over Time (2020-2026)")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.savefig("chart1_aqi_trend.png", dpi=100)
plt.close()
print("Saved chart1_aqi_trend.png")

# ---------------------------------------------------------------
# CHART 2: Average AQI by hour of day
# ---------------------------------------------------------------
hourly_avg = df.groupby("hour")["target_aqi"].mean()

plt.figure(figsize=(10, 5))
plt.bar(hourly_avg.index, hourly_avg.values, color="steelblue")
plt.title("Average AQI by Hour of Day")
plt.xlabel("Hour (24h)")
plt.ylabel("Average AQI")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("chart2_hourly_pattern.png", dpi=100)
plt.close()
print("Saved chart2_hourly_pattern.png")

# ---------------------------------------------------------------
# CHART 3: Average AQI by month
# ---------------------------------------------------------------
monthly_avg = df.groupby("month")["target_aqi"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

plt.figure(figsize=(10, 5))
plt.bar(range(1, 13), monthly_avg.values, color="darkorange")
plt.title("Average AQI by Month (Seasonal Pattern)")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(range(1, 13), month_names)
plt.tight_layout()
plt.savefig("chart3_monthly_pattern.png", dpi=100)
plt.close()
print("Saved chart3_monthly_pattern.png")

# ---------------------------------------------------------------
# CHART 4: Boxplot - checking for outliers across pollutants
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# CHART 4: Boxplots - checking for outliers across ALL raw sensor
# readings (pollutants + weather), grouped by similar scale so
# no column gets visually flattened by a much larger-scale one.
# ---------------------------------------------------------------

# Group 1: small-scale gas pollutants (roughly 0-150 range)
group1 = ["no2", "so2", "nh3", "pm2_5"]

# Group 2: larger-scale pollutants (roughly 0-900 range)
group2 = ["pm10", "co", "o3"]

# Note: temperature/humidity/pressure/wind/clouds are NOT included -
# this historical dataset doesn't have weather columns (only our
# newer hourly_dataset.csv does, collected going forward).

groups = {
    "chart4a_outliers_gases.png": group1,
    "chart4b_outliers_particulates_co.png": group2,
}

for filename, cols in groups.items():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[cols])
    plt.title(f"Outlier Check: {', '.join(cols)}")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved {filename}")

print("\nAll charts saved! Open the PNG files to view them.")