import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

START_DATE = "2024-08-01"
END_DATE   = "2024-08-30"
BUDAPEST   = {"lat": 47.4291, "lon": 19.1822}

def build_client():
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_cloud_cover(client):
    params = {
        "latitude":   BUDAPEST["lat"],
        "longitude":  BUDAPEST["lon"],
        "hourly":     ["rain"],
        "timezone":   "UTC",
        "start_date": START_DATE,
        "end_date":   END_DATE,
    }
    responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    hourly = responses[0].Hourly()
    timestamps = pd.date_range(
        start=pd.Timestamp(hourly.Time(),    unit="s", tz="UTC"),
        end=  pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    cloud_cover = hourly.Variables(0).ValuesAsNumpy()
    return timestamps, cloud_cover


if __name__ == "__main__":
    client = build_client()
    timestamps, cloud_cover = fetch_cloud_cover(client)

    print(f"Fetched {len(cloud_cover)} hourly values")
    print(f"Range: {timestamps[0]} → {timestamps[-1]}")
    print(f"Min: {cloud_cover.min():.1f}%  Max: {cloud_cover.max():.1f}%  Mean: {cloud_cover.mean():.1f}%")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(timestamps, cloud_cover, linewidth=0.5, color="#4A90D9", alpha=0.8)
    ax.set_title("Budapest — Hourly Cloud Cover Total", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cloud Cover [%]")
    #ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    plt.show()
