import os
import numpy as np
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

BUDAPEST   = {"lat": 47.4291, "lon": 19.1822}
START_DATE = "2020-01-02"
END_DATE   = "2020-02-29"

CHUNK_PATH = os.path.join(os.path.dirname(__file__), "dataset-bsrn", "saved", "chunk0.npz")


def build_client():
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_openmeteo_gti(client):
    params = {
        "latitude":   BUDAPEST["lat"],
        "longitude":  BUDAPEST["lon"],
        "hourly":     ["global_tilted_irradiance"],
        "tilt":       80,   # flat/horizontal — matches BSRN global_rad (SWD)
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
    gti = hourly.Variables(0).ValuesAsNumpy()
    return timestamps, gti


def load_bsrn_chunk():
    data = np.load(CHUNK_PATH)
    ts_minutes = data["timestamps"]
    global_rad = data["global_rad"]
    timestamps = pd.to_datetime(ts_minutes.astype(np.int64) * 60, unit="s", utc=True)
    return timestamps, global_rad


if __name__ == "__main__":
    client = build_client()

    om_ts,   om_gti    = fetch_openmeteo_gti(client)
    bsrn_ts, bsrn_gti = load_bsrn_chunk()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(om_ts,   om_gti,   linewidth=0.8, color="#E05C2A", alpha=0.85, label="OpenMeteo GTI (tilt=0°)")
    ax.plot(bsrn_ts, bsrn_gti, linewidth=0.8, color="#4A90D9", alpha=0.85, label="BSRN global_rad (SWD)")
    ax.set_title("Budapest — BSRN vs OpenMeteo global irradiance (chunk0)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Irradiance [W/m²]")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    plt.show()
