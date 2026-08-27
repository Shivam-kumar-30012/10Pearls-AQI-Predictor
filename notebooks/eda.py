"""
EDA - Exploratory Data Analysis

Explores the feature table built by pipeline step 04, and saves charts
to notebooks/figures/.

Replaces the four older eda_step*.py scripts. Two of those existed to
find and remove -9999 sensor values and duplicate rows; that cleaning
now happens inside the pipeline (step 01 converts sentinels to NaN,
step 03 de-duplicates), so this script is purely about understanding
the data rather than repairing it.

The point of running this BEFORE training is to check that the features
we built are actually justified by the data. If wind speed shows no
relationship with AQI, the ventilation feature is not earning its place.

Reads:  data/processed/features.csv
Writes: notebooks/figures/*.png

Run:    python notebooks/eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEATURES_CSV, FIGURES_DIR, CITY_NAME

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# EPA category bands, used to shade charts
BANDS = [
    (0, 50, "#00e400", "Good"),
    (50, 100, "#ffff00", "Moderate"),
    (100, 150, "#ff7e00", "Unhealthy (sensitive)"),
    (150, 200, "#ff0000", "Unhealthy"),
    (200, 300, "#8f3f97", "Very unhealthy"),
    (300, 700, "#7e0023", "Hazardous"),
]


def save(fig, name):
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  saved %s" % name)


def shade_bands(ax, xmax):
    for lo, hi, colour, _ in BANDS:
        ax.axhspan(lo, hi, color=colour, alpha=0.10, zorder=0)


# ---------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------
def summary(df):
    print("\n" + "=" * 60)
    print("OVERVIEW")
    print("=" * 60)
    print("Rows        : %d" % len(df))
    print("Columns     : %d" % df.shape[1])
    print("Date range  : %s -> %s"
          % (df.timestamp.min().date(), df.timestamp.max().date()))
    print("Years       : %.1f" % ((df.timestamp.max() - df.timestamp.min()).days / 365))

    print("\nAQI")
    print("  mean %.1f | median %.1f | std %.1f"
          % (df.aqi.mean(), df.aqi.median(), df.aqi.std()))
    print("  min %.0f | max %.0f" % (df.aqi.min(), df.aqi.max()))

    print("\nTime in each EPA category:")
    for lo, hi, _, label in BANDS:
        pct = ((df.aqi >= lo) & (df.aqi < hi)).mean() * 100
        bar = "#" * int(pct / 2)
        print("  %-22s %5.1f%%  %s" % (label, pct, bar))

    print("\nDominant pollutant:")
    for k, v in df.dominant_pollutant.value_counts().items():
        print("  %-8s %6d  (%.1f%%)" % (k, v, v / len(df) * 100))

    print("\nAutocorrelation of AQI (how much the past predicts the future):")
    for h in [1, 6, 12, 24, 48, 72]:
        print("  %3dh ahead : %.3f" % (h, df.aqi.corr(df.aqi.shift(-h))))


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------
def chart_timeline(df):
    daily = df.set_index("timestamp")["aqi"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    shade_bands(ax, len(daily))
    ax.plot(daily.index, daily.values, lw=0.7, color="#1a1a1a")
    ax.plot(daily.index, daily.rolling(30, min_periods=10).mean(),
            lw=2, color="#0066cc", label="30-day average")
    ax.set_title("%s: daily average AQI, %s to %s"
                 % (CITY_NAME, daily.index.min().year, daily.index.max().year))
    ax.set_ylabel("AQI")
    ax.set_xlabel("")
    ax.set_ylim(0, min(400, daily.max() * 1.1))
    ax.legend(loc="upper right")
    save(fig, "01_aqi_timeline.png")


def chart_daily_seasonal(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    h = df.groupby("hour")["aqi"].agg(["mean", "std"])
    axes[0].plot(h.index, h["mean"], marker="o", color="#0066cc")
    axes[0].fill_between(h.index, h["mean"] - h["std"], h["mean"] + h["std"],
                         alpha=0.15, color="#0066cc")
    axes[0].set_title("Average AQI by hour of day (UTC)")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("AQI")
    axes[0].set_xticks(range(0, 24, 3))

    m = df.groupby("month")["aqi"].agg(["mean", "std"])
    colours = ["#7e0023" if v > 150 else "#ff7e00" if v > 100 else "#00a000"
               for v in m["mean"]]
    axes[1].bar(m.index, m["mean"], yerr=m["std"], color=colours,
                capsize=3, error_kw={"alpha": 0.4})
    axes[1].set_title("Average AQI by month")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("AQI")
    axes[1].set_xticks(list(m.index))
    axes[1].set_xticklabels([MONTHS[i - 1] for i in m.index], rotation=45)

    save(fig, "02_daily_seasonal_pattern.png")


def chart_dominant(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    counts = df.dominant_pollutant.value_counts()
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=["#0066cc", "#cc6600", "#00a000", "#999999"],
                startangle=90)
    axes[0].set_title("Which pollutant drives the AQI")

    by_month = (df.groupby(["month", "dominant_pollutant"]).size()
                  .unstack(fill_value=0))
    by_month = by_month.div(by_month.sum(axis=1), axis=0) * 100
    by_month.plot(kind="bar", stacked=True, ax=axes[1], width=0.85,
                  color=["#0066cc", "#cc6600", "#00a000", "#999999"])
    axes[1].set_title("Dominant pollutant by month (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("% of hours")
    axes[1].set_xticklabels([MONTHS[m - 1] for m in by_month.index], rotation=45)
    axes[1].legend(title="", fontsize=8)

    save(fig, "03_dominant_pollutant.png")


def chart_weather(df):
    """
    The justification check for our weather features. If these panels
    are flat, features like ventilation are not pulling their weight.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    pairs = [
        ("wind_speed", "Wind speed (m/s)", axes[0][0]),
        ("humidity", "Humidity (%)", axes[0][1]),
        ("temperature", "Temperature (C)", axes[0][2]),
        ("ventilation", "Ventilation index", axes[1][0]),
        ("temp_range_24h", "24h temperature range (C)", axes[1][1]),
        ("pressure", "Pressure (hPa)", axes[1][2]),
    ]

    for col, label, ax in pairs:
        d = df[[col, "aqi"]].dropna()
        if len(d) < 100:
            ax.set_visible(False)
            continue
        # Bin into deciles and plot the mean AQI per bin: far clearer
        # than a scatter of 48,000 points.
        bins = pd.qcut(d[col], 10, duplicates="drop")
        g = d.groupby(bins, observed=True)["aqi"].agg(["mean", "count"])
        centres = [iv.mid for iv in g.index]
        ax.plot(centres, g["mean"], marker="o", color="#0066cc")
        r = d[col].corr(d["aqi"])
        ax.set_title("%s   (r = %+.2f)" % (label, r), fontsize=10)
        ax.set_ylabel("mean AQI")

    fig.suptitle("AQI against weather conditions", fontsize=13)
    save(fig, "04_weather_relationships.png")


def chart_wind_rose(df):
    d = df[["wind_deg", "wind_speed", "aqi"]].dropna()
    sectors = np.arange(0, 361, 22.5)
    d = d.assign(sector=pd.cut(d.wind_deg, sectors, right=False))
    g = d.groupby("sector", observed=True)["aqi"].mean()

    theta = np.deg2rad([iv.left + 11.25 for iv in g.index])
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bars = ax.bar(theta, g.values, width=np.deg2rad(20), alpha=0.8)
    for v, bar in zip(g.values, bars):
        bar.set_facecolor(plt.cm.YlOrRd(
            (v - g.min()) / max(1e-9, g.max() - g.min())))
    ax.set_title("Mean AQI by wind direction\n(where the dirty air comes from)",
                 pad=20)
    save(fig, "05_wind_direction.png")


def chart_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(df.aqi.dropna(), bins=80, color="#0066cc", alpha=0.85)
    for lo, hi, colour, _ in BANDS:
        axes[0].axvspan(lo, hi, color=colour, alpha=0.12, zorder=0)
    axes[0].axvline(df.aqi.mean(), color="black", ls="--", lw=1.2,
                    label="mean %.0f" % df.aqi.mean())
    axes[0].set_title("Distribution of hourly AQI")
    axes[0].set_xlabel("AQI")
    axes[0].set_ylabel("hours")
    axes[0].set_xlim(0, min(400, df.aqi.max()))
    axes[0].legend()

    pollutants = ["pm2_5", "pm10", "no2", "o3", "so2", "nh3"]
    data = [df[c].dropna() for c in pollutants]
    try:
        axes[1].boxplot(data, tick_labels=pollutants, showfliers=False)
    except TypeError:   # matplotlib < 3.9
        axes[1].boxplot(data, labels=pollutants, showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_title("Pollutant concentrations (log scale, outliers hidden)")
    axes[1].set_ylabel("ug/m3")

    save(fig, "06_distributions.png")


def chart_correlations(df):
    cols = ["aqi", "aqi_lag_1h", "aqi_lag_24h", "aqi_lag_72h",
            "aqi_roll_mean_1d", "aqi_roll_mean_7d", "aqi_trend_24h",
            "pm2_5", "pm10", "o3", "co", "no2",
            "temperature", "humidity", "wind_speed", "ventilation",
            "pressure", "precip_24h", "temp_range_24h"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > 0.35:
                ax.text(j, i, "%.2f" % v, ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > 0.7 else "black")
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Feature correlations")
    save(fig, "07_correlations.png")


def chart_persistence(df):
    """
    How far ahead is AQI predictable at all? This sets the realistic
    expectation for the models trained in step 06.
    """
    hours = list(range(1, 73))
    corrs = [df.aqi.corr(df.aqi.shift(-h)) for h in hours]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hours, corrs, color="#0066cc", lw=2)
    ax.fill_between(hours, 0, corrs, alpha=0.15, color="#0066cc")
    for h, label in [(24, "day+1"), (48, "day+2"), (72, "day+3")]:
        ax.axvline(h, color="grey", ls=":", lw=1)
        ax.text(h, 1.02, label, ha="center", fontsize=9, color="grey")
    ax.set_title("How well does today's AQI predict later hours?")
    ax.set_xlabel("hours ahead")
    ax.set_ylabel("correlation")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 72)
    save(fig, "08_predictability.png")


def main():
    print("Loading features...")
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("  %d rows x %d columns" % (len(df), df.shape[1]))

    summary(df)

    print("\nBuilding charts...")
    chart_timeline(df)
    chart_daily_seasonal(df)
    chart_dominant(df)
    chart_weather(df)
    chart_wind_rose(df)
    chart_distribution(df)
    chart_correlations(df)
    chart_persistence(df)

    print("\nAll charts saved to %s" % FIGURES_DIR)


if __name__ == "__main__":
    main()