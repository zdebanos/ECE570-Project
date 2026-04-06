import csv
import os
import re

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

LOCATIONS = [
    {"lat": "47.3456389", "lon": "19.0415875", "name": "Szigetszentmiklos", "tilt": 25},
    {"lat": "47.4058408", "lon": "19.2638489", "name": "Vecses",            "tilt": 27.5},
    {"lat": "47.4910539", "lon": "19.3426831", "name": "Pecel",             "tilt": 30},
    {"lat": "47.6312217", "lon": "19.1288833", "name": "Dunakezsi",         "tilt": 32.5},
]

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "global_tilted_irradiance",
]

BUDAPEST = {"lat": 47.4291, "lon": 19.1822}

EPOCH    = pd.Timestamp("1970-01-01", tz="UTC")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "dataset-openmeteo", "saved")


def build_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_location(client: openmeteo_requests.Client, loc: dict) -> None:
    name = loc["name"]
    print(f"Fetching {name} (lat={loc['lat']}, lon={loc['lon']}, tilt={loc['tilt']}°) …")
    params = {
        "latitude":  float(loc["lat"]),
        "longitude": float(loc["lon"]),
        "hourly":    HOURLY_VARIABLES,
        "tilt":      loc["tilt"],
        "timezone":  "UTC",
        "start_date": START_DATE,
        "end_date":   END_DATE,
    }
    responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    response  = responses[0]

    hourly = response.Hourly()
    timestamps = pd.date_range(
        start=pd.Timestamp(hourly.Time(),    unit="s", tz="UTC"),
        end=  pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"time": timestamps}
    for i, var in enumerate(HOURLY_VARIABLES):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)

    # Drop Feb 29
    df = df[~((df["time"].dt.month == 2) & (df["time"].dt.day == 29))]

    # Timestamps as minutes since Unix epoch
    minutes = ((df["time"] - EPOCH).dt.total_seconds() / 60).astype(np.int64).to_numpy()

    out_path = os.path.join(SAVE_DIR, f"{name}.npz")
    np.savez(
        out_path,
        timestamps=minutes,
        temperature_2m=df["temperature_2m"].to_numpy(dtype=np.float64),
        relative_humidity_2m=df["relative_humidity_2m"].to_numpy(dtype=np.float64),
        surface_pressure=df["surface_pressure"].to_numpy(dtype=np.float64),
        wind_speed_10m=df["wind_speed_10m"].to_numpy(dtype=np.float64),
        global_tilted_irradiance=df["global_tilted_irradiance"].to_numpy(dtype=np.float64),
    )
    print(f"  Saved {len(df)} rows → {out_path}")


def fetch_budapest_wind_chunks(client: openmeteo_requests.Client) -> None:
    bsrn_saved = os.path.join(os.path.dirname(__file__), "dataset-bsrn", "saved")
    dates_path = os.path.join(bsrn_saved, "dates.txt")

    with open(dates_path, newline="") as f:
        rows = list(csv.reader(f))

    lat, lon = BUDAPEST["lat"], BUDAPEST["lon"]

    for row in rows:
        chunk_file, start_str, end_str = row[0].strip(), row[1].strip(), row[2].strip()
        chunk_num = int(re.search(r"\d+", chunk_file).group())

        # Load the exact timestamps from the chunk so we match them precisely
        chunk_ts = np.load(os.path.join(bsrn_saved, chunk_file))["timestamps"]  # minutes since epoch
        chunk_times = pd.to_datetime(chunk_ts * 60, unit="s", utc=True)

        print(f"Fetching Budapest wind for chunk{chunk_num} ({start_str[:10]} → {end_str[:10]}) …")
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     ["wind_speed_10m"],
            "timezone":   "UTC",
            "start_date": start_str[:10],
            "end_date":   end_str[:10],
        }
        responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        hourly = responses[0].Hourly()
        timestamps = pd.date_range(
            start=pd.Timestamp(hourly.Time(),    unit="s", tz="UTC"),
            end=  pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({"time": timestamps, "wind_speed_10m": hourly.Variables(0).ValuesAsNumpy()})

        # Keep only rows whose timestamps appear in the chunk
        df = df[df["time"].isin(chunk_times)].reset_index(drop=True)

        minutes = ((df["time"] - EPOCH).dt.total_seconds() / 60).astype(np.int64).to_numpy()
        out_path = os.path.join(SAVE_DIR, f"Budapest{chunk_num}.npz")
        np.savez(out_path, timestamps=minutes, wind_speed_10m=df["wind_speed_10m"].to_numpy(dtype=np.float64))
        print(f"  Saved {len(df)} rows → {out_path}")


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    client = build_client()
    for loc in LOCATIONS:
        fetch_location(client, loc)
    fetch_budapest_wind_chunks(client)
    print("Done.")


if __name__ == "__main__":
    main()
