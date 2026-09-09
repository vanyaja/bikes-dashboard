"""
open_meteo.py - fetch daily weather from Open-Meteo.

Open-Meteo (https://open-meteo.com) is free for non-commercial use and needs no
API key. This module has two functions, both returning the same tidy shape:

    from open_meteo import open_meteo, open_meteo_history

    open_meteo("London", 5)                              # next few days (forecast)
    open_meteo_history("London", "2026-01-01", "2026-01-07")  # a past date range

Both return a pandas DataFrame with one row per day and the columns:

    date, day_of_week, temp, humidity, precip, windspeed, cloudcover,
    solarenergy, visibility

which line up with the model's predictors. Temperature is in degrees Celsius,
wind in km/h, precipitation in mm, humidity and cloud cover in percent,
solarenergy in MJ/m^2 (matches the training data's `solarenergy`), and
visibility in km (matches the training data's `visibility`).

The forecast reaches about 7 days ahead. For any date in the past (for example
the first week of January 2026) use open_meteo_history, which reads Open-Meteo's
historical archive. Open-Meteo's archive has no visibility record, so
`visibility` comes back as NaN for historical dates; callers should fill it
with a sensible fallback (e.g. the training data's average) if their model needs it.
"""

import requests
import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Forecast API daily field  ->  the column name our model uses
DAILY_FIELDS = {
    "temperature_2m_mean": "temp",
    "relative_humidity_2m_mean": "humidity",
    "precipitation_sum": "precip",
    "wind_speed_10m_mean": "windspeed",
    "cloud_cover_mean": "cloudcover",
    "shortwave_radiation_sum": "solarenergy",  # MJ/m^2, same unit as the training data
}

# The archive API has no daily means for most fields, so we pull these hourly
# fields and aggregate them ourselves (mean for most, sum for precipitation).
# Visibility is not in the historical archive at all - it comes back as NaN.
HOURLY_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "wind_speed_10m": "windspeed",
    "cloud_cover": "cloudcover",
    "visibility": "visibility",
}

# Forecast visibility is only available hourly; we average it to a daily figure.
FORECAST_HOURLY_FIELDS = {
    "visibility": "visibility",
}


def geocode(location):
    """Turn a place name into (latitude, longitude, label) using Open-Meteo."""
    resp = requests.get(
        GEOCODE_URL,
        params={"name": location, "count": 1, "language": "en"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Open-Meteo could not find a location called {location!r}.")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("country")] if p)
    return top["latitude"], top["longitude"], label


def open_meteo(location="London", days_to_forecast=5):
    """Return a daily weather forecast for a location as a tidy DataFrame.

    Args:
        location (str): a place name, e.g. "London" or "Paris".
        days_to_forecast (int): number of days ahead, from 1 to 7
            (Open-Meteo's forecast does not go beyond 7 days).

    Returns:
        pandas.DataFrame with columns date, day_of_week, temp, humidity,
        precip, windspeed, cloudcover, solarenergy, visibility. The resolved
        place name is stored in df.attrs["location"].
    """
    days_to_forecast = int(days_to_forecast)
    if not 1 <= days_to_forecast <= 7:
        raise ValueError("days_to_forecast must be between 1 and 7 (Open-Meteo's forecast limit).")

    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_FIELDS),
        "hourly": ",".join(FORECAST_HOURLY_FIELDS),
        "forecast_days": days_to_forecast,
        "timezone": "auto",
        "wind_speed_unit": "kmh",   # matches the training data units
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload["daily"]
    hourly = payload["hourly"]

    df = pd.DataFrame({col: daily[field] for field, col in DAILY_FIELDS.items()})
    df.insert(0, "date", pd.to_datetime(daily["time"]))
    df.insert(1, "day_of_week", df["date"].dt.strftime("%a"))

    vis = pd.DataFrame({"visibility_m": hourly["visibility"], "date": pd.to_datetime(hourly["time"]).normalize()})
    daily_vis = vis.groupby("date")["visibility_m"].mean() / 1000  # m -> km, matches training data
    df["visibility"] = df["date"].map(daily_vis).round(1)

    df.attrs["location"] = label
    return df


def open_meteo_history(location, start_date, end_date):
    """Return daily weather for a past date range from Open-Meteo's archive.

    Use this for dates the forecast cannot reach, such as the first week of
    January 2026. The archive has no daily means for most fields, so we pull
    the hourly values and aggregate them to one row per day here. The archive
    has no visibility record at all, so that column comes back as NaN.

    Args:
        location (str): a place name, e.g. "London".
        start_date (str): first day, "YYYY-MM-DD".
        end_date (str): last day, "YYYY-MM-DD".

    Returns:
        pandas.DataFrame with the same columns as open_meteo(): date,
        day_of_week, temp, humidity, precip, windspeed, cloudcover,
        solarenergy, visibility.
    """
    lat, lon, label = geocode(location)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_FIELDS),
        "daily": "shortwave_radiation_sum",
        "timezone": "auto",
        "wind_speed_unit": "kmh",   # matches the training data units
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]

    hf = pd.DataFrame({col: hourly[field] for field, col in HOURLY_FIELDS.items()})
    hf["date"] = pd.to_datetime(hourly["time"]).normalize()
    hf["visibility"] = pd.to_numeric(hf["visibility"], errors="coerce") / 1000  # m -> km

    # Aggregate hours to days: mean for levels, sum for precipitation
    daily = hf.groupby("date").agg(
        temp=("temp", "mean"),
        humidity=("humidity", "mean"),
        precip=("precip", "sum"),
        windspeed=("windspeed", "mean"),
        cloudcover=("cloudcover", "mean"),
        visibility=("visibility", "mean"),
    ).reset_index()

    solar = pd.DataFrame(payload["daily"])
    solar["time"] = pd.to_datetime(solar["time"])
    daily["solarenergy"] = daily["date"].map(solar.set_index("time")["shortwave_radiation_sum"])

    daily.insert(1, "day_of_week", daily["date"].dt.strftime("%a"))
    daily.attrs["location"] = label
    return daily


if __name__ == "__main__":
    # Quick manual checks (need internet)
    print("Forecast (next 5 days):")
    print(open_meteo("London", 5).to_string(index=False))
    print("\nHistory (first week of January 2026):")
    print(open_meteo_history("London", "2026-01-01", "2026-01-07").to_string(index=False))
