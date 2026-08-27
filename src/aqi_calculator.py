
# EPA AQI Calculator - CORRECTED VERSION


# Reference: US EPA, "Technical Assistance Document for the Reporting of
# Daily Air Quality - the Air Quality Index (AQI)".


import math
import pandas as pd

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
PM25_DECIMALS = 1

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]
PM10_DECIMALS = 0

CO_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]
CO_DECIMALS = 1

O3_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]
O3_DECIMALS = 0

CO_UGM3_TO_PPM = 1145.0
O3_UGM3_TO_PPB = 1.96


def truncate(value, decimals):
    # Truncates toward zero per EPA convention. This closes the gaps.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def calculate_sub_index(concentration, breakpoints, decimals):
    
# EPA linear interpolation:
#     I = ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo

# Below scale  -> 0
# Within scale -> interpolate
# Above scale  -> extrapolate along final bracket slope (not clamp)
    
    if concentration is None or (isinstance(concentration, float)
                                 and math.isnan(concentration)):
        return float("nan")

    c = truncate(concentration, decimals)

    if c < breakpoints[0][0]:
        return 0.0

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            slope = (i_high - i_low) / (c_high - c_low)
            return round(slope * (c - c_low) + i_low)

    c_low, c_high, i_low, i_high = breakpoints[-1]
    slope = (i_high - i_low) / (c_high - c_low)
    return round(slope * (c - c_low) + i_low)


def calculate_aqi(pm25=None, pm10=None, co=None, o3=None,
                  co_in_ugm3=True, o3_in_ugm3=True,
                  return_dominant=False):
    
# Overall AQI = max of available sub-indices (dominant pollutant rule).
# Missing pollutants are skipped, not treated as zero.
    
    subs = {}

    if pm25 is not None:
        subs["pm2_5"] = calculate_sub_index(pm25, PM25_BREAKPOINTS, PM25_DECIMALS)
    if pm10 is not None:
        subs["pm10"] = calculate_sub_index(pm10, PM10_BREAKPOINTS, PM10_DECIMALS)
    if co is not None:
        co_ppm = co / CO_UGM3_TO_PPM if co_in_ugm3 else co
        subs["co"] = calculate_sub_index(co_ppm, CO_BREAKPOINTS, CO_DECIMALS)
    if o3 is not None:
        o3_ppb = o3 / O3_UGM3_TO_PPB if o3_in_ugm3 else o3
        subs["o3"] = calculate_sub_index(o3_ppb, O3_BREAKPOINTS, O3_DECIMALS)

    valid = {k: v for k, v in subs.items() if not math.isnan(v)}
    if not valid:
        return (float("nan"), None) if return_dominant else float("nan")

    dominant = max(valid, key=valid.get)
    overall = valid[dominant]

    return (overall, dominant) if return_dominant else overall


def compute_aqi_series(df, pm25_col="pm2_5", pm10_col="pm10",
                       co_col="co", o3_col="o3",
                       timestamp_col="timestamp"):
    
    # Adds columns: aqi_correct, dominant_pollutant
    # Assumes hourly, chronologically sorted data.
    
    df = df.sort_values(timestamp_col).reset_index(drop=True).copy()

    df["_pm25_24h"] = df[pm25_col].rolling(24, min_periods=18).mean()
    df["_pm10_24h"] = df[pm10_col].rolling(24, min_periods=18).mean()
    df["_co_8h"] = df[co_col].rolling(8, min_periods=6).mean()
    df["_o3_8h"] = df[o3_col].rolling(8, min_periods=6).mean()

    results = df.apply(
        lambda r: calculate_aqi(
            pm25=r["_pm25_24h"], pm10=r["_pm10_24h"],
            co=r["_co_8h"], o3=r["_o3_8h"],
            return_dominant=True,
        ),
        axis=1, #row by row.
    )

    df["aqi_correct"] = [r[0] for r in results]
    df["dominant_pollutant"] = [r[1] for r in results]

    return df.drop(columns=["_pm25_24h", "_pm10_24h", "_co_8h", "_o3_8h"])


AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 10000, "Hazardous", "#7e0023"),
]


def aqi_category(aqi):
    """Returns (label, hex_colour) for dashboard display and alerts."""
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return ("Unknown", "#888888")

    # Round to a whole number first. The categories are defined on
    # integers (51-100, 101-150), so a float like 100.4 falls between
    # brackets and would drop through to the final fallback - the same
    # gap bug that corrupted the AQI calculation itself.
    aqi = round(aqi)

    for low, high, label, colour in AQI_CATEGORIES:
        if low <= aqi <= high:
            return (label, colour)
    return ("Hazardous", "#7e0023")


if __name__ == "__main__":
    print("=== Values that previously returned 500 ===")
    for v in [12.05, 35.45, 55.45, 150.45, 12.0, 12.1]:
        print("  pm2.5=%8.2f  %s" % (v, calculate_sub_index(v, PM25_BREAKPOINTS, 1)))

    print()
    for v in [54.5, 154.5, 254.5, 604.0, 800.0, 1262.51]:
        print("  pm10 =%8.2f  %s" % (v, calculate_sub_index(v, PM10_BREAKPOINTS, 0)))

    print()
    for ug in [107.0, 137.0, 168.0, 240.33]:
        print("  o3   =%8.2f ug/m3 (%6.2f ppb) -> %s"
              % (ug, ug / 1.96, calculate_sub_index(ug / 1.96, O3_BREAKPOINTS, 0)))

    print("\n Full AQI ")
    print("  pm25=12.05, clean air :", calculate_aqi(12.05, 10, 100, 20))
    print("  Ghotki dust storm     :", calculate_aqi(80, 1262.51, 460, 150,
                                                     return_dominant=True))
    print("  Typical Ghotki reading:", calculate_aqi(34.72, 113.19, 180.07, 69.22,
                                                     return_dominant=True))