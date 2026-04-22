from matplotlib.pylab import plot
import numpy as np
import pandas as pd
import scipy
import torch
import rich
import sys
import openmeteo_requests
import requests_cache
import matplotlib.pyplot as plt
import pvlib

from retry_requests import retry
from typing import Tuple
from datetime import datetime, timezone
from importlib.resources import files
from solar_gru import SolarSeq2SeqGRU
from model_to_solar_plant import Model2SolarPlantMapper, Model2SolarPlantMapperCreator
from schedule_model.solar_milp_model import SolarMILPModel, SolarMilpModelParameters
from schedule_model.reference_data import reference_power_consumption

# Override the base Interface classes
class SimpleModelPowerMapper(Model2SolarPlantMapper):
    """A Simple GTI to Real Power mapper
    We suppose 12 AIKO Neostar 2S A-MAH54Mb 450W Solar panels with dimensions
    of 1757 x 1134 millimeters and a 22.6 % efficiency.

    Also converts W to kW.
    """
    def map(self, timestamps: np.ndarray, model_outputs: np.ndarray) -> np.ndarray:
        return model_outputs * 1757 * 1134 / 1e6 * 0.226 * 12 / 1000

class SolarPowerMapperCreator(Model2SolarPlantMapperCreator):
    def create_mapper(self, *args, **kwargs) -> Model2SolarPlantMapper:
        return SimpleModelPowerMapper()

def _ok_print(*args, **kwargs) -> None:
    text = " ".join(str(a) for a in args)
    rich.print(f"[green]{text}[/green]", **kwargs)

def _err_print(*args, **kwargs) -> None:
    text = " ".join(str(a) for a in args)
    rich.print(f"[red]{text}[/red]", **kwargs)

def _warn_print(*args, **kwargs) -> None:
    text = " ".join(str(a) for a in args)
    rich.print(f"[yellow]{text}[/yellow]", **kwargs)

_PAST_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "cloud_cover",
    "global_tilted_irradiance",
]

_FUTURE_VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "global_tilted_irradiance",
]

# Fixed 28-day February, it's consistent with the dataset
_DAYS_PER_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def _encode_timestamps(times: pd.DatetimeIndex,) -> dict:
    """Helper function.
    Compute cyclical time encodings from a DatetimeIndex.
    Applies sqrt(2) scaling to match the training encoding.

    Args:
        times: UTC DatetimeIndex to encode.

    Returns:
        Dictionary of numpy arrays (keys "h_cos", "h_sin", "d_cos", "d_sin").
    """
    sqrt2 = np.sqrt(2.0)
    hours = times.hour.to_numpy(dtype=np.float32)
    doy = np.array(
        [sum(_DAYS_PER_MONTH[1:t.month]) + (t.day - 1) for t in times],
        dtype=np.float32,
    )
    ret = {}
    ret["h_cos"] = (np.cos(hours / 24  * 2 * np.pi) * sqrt2).astype(np.float32)
    ret["h_sin"] = (np.sin(hours / 24  * 2 * np.pi) * sqrt2).astype(np.float32)
    ret["d_cos"] = (np.cos(doy   / 365 * 2 * np.pi) * sqrt2).astype(np.float32)
    ret["d_sin"] = (np.sin(doy   / 365 * 2 * np.pi) * sqrt2).astype(np.float32)
    return ret

def compute_time_encodings(anchor: datetime) -> Tuple[dict, dict]:
    """Compute cyclical time encodings for past and future windows
    from a datetime anchor.

    Args:
        anchor: Reference datetime (UTC).

    Returns:
        Tuple (past_enc, future_enc), each a dict with keys
        h_cos, h_sin, d_cos, d_sin as float32 arrays of length 96 and 48.
    """
    ts = pd.Timestamp(anchor)
    past_times   = pd.date_range(end=ts,   periods=96, freq="1h")
    future_times = pd.date_range(start=ts, periods=48, freq="1h")
    return _encode_timestamps(past_times), _encode_timestamps(future_times)

def _build_openmeteo_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)  # type: ignore[arg-type]

def fetch_weather_data(
    lat: float,
    lon: float,
    anchor: datetime,
) -> Tuple[dict, dict]:
    """Fetch past and future hourly weather data from OpenMeteo.

    Fetches 4 days (96 hours) of past data ending at anchor
    and 2 days (48 hours) of future data starting at anchor.
    GTI is fetched with a panel tilt of 30 degrees, so it matches
    the dataset the neural network was trained on.

    Args:
        lat:    Latitude in decimal degrees.
        lon:    Longitude in decimal degrees.
        anchor: Reference datetime (UTC).

    Returns:
        A tuple (past, future) of dicts mapping variable names from OpenMeteo.
          - past:   96 values per variable, except GTI with 97 values (one is needed
                    for the decoder), ending at anchor.
          - future: 48 values per variable, starting at anchor.
    """
    date = anchor.strftime("%Y-%m-%d")
    hour = anchor.hour
    fetch_start = (pd.Timestamp(anchor) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    fetch_end   = (pd.Timestamp(anchor) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    all_variables = list(dict.fromkeys(_PAST_VARIABLES + _FUTURE_VARIABLES))

    client = _build_openmeteo_client()
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     all_variables,
        "tilt":       30,
        "timezone":   "UTC",
        "start_date": fetch_start,
        "end_date":   fetch_end,
    }
    responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    hourly = responses[0].Hourly()
    assert hourly is not None, "OpenMeteo returned no hourly data"

    raw: dict = {}
    for i, var in enumerate(all_variables):
        raw[var] = hourly.Variables(i).ValuesAsNumpy()

    # Separate fetch for GTI: one extra hour into the past for GTI(i-1).
    # We actually need 5 days for the preceding day
    fetch_start_gti = (pd.Timestamp(date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    gti_responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params={
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ["global_tilted_irradiance"],
        "tilt":       30,
        "timezone":   "UTC",
        "start_date": fetch_start_gti,
        "end_date":   fetch_end,
    })
    gti_hourly = gti_responses[0].Hourly()
    assert gti_hourly is not None, "OpenMeteo returned no GTI data"
    # GTI fetch starts 5 days before date, so anchor is at index 5*24 + hour.
    gti_anchor_idx = 5 * 24 + hour
    gti_raw = gti_hourly.Variables(0).ValuesAsNumpy()

    anchor_idx = 4 * 24 + hour
    past = {var: raw[var][anchor_idx - 95 : anchor_idx + 1] for var in _PAST_VARIABLES}
    past["global_tilted_irradiance"] = gti_raw[gti_anchor_idx - 96 : gti_anchor_idx + 1]  # 97 values

    future = {var: raw[var][anchor_idx : anchor_idx + 48] for var in _FUTURE_VARIABLES}
    return past, future

def _norm(x: np.ndarray, mean, sigma) -> np.ndarray:
    """Helper function to normalize the data using the dataset means and sigmas.
    
    Args:
        x:     the array to be normalized
        mean:  the array mean
        sigma: the array sigma

    Returns:
        Normalized array.
    """
    return ((x - mean) / sigma).astype(np.float32)

def build_tensors(
    past: dict,
    future: dict,
    past_enc: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    future_enc: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    means: np.ndarray,
    sigmas: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build normalized encoder and decoder input tensors for SolarSeq2SeqGRU.

    Args:
        past:       Dict of past weather arrays (96 values each, GTI 97).
        future:     Dict of future weather arrays (48 values each).
        past_enc:   Cyclical time encodings for past (h_cos, h_sin, d_cos, d_sin).
        future_enc: Cyclical time encodings for future (h_cos, h_sin, d_cos, d_sin).

    Returns:
        Tuple (encoder_in, decoder_in):
          - encoder_in:  float32 tensor of shape (1, 96, 11).
          - decoder_in:  float32 tensor of shape (1, 48, 7).
    """
    # The package also has the dataset means and sigmas, load it
    h_cos_p = past_enc["h_cos"]
    h_sin_p = past_enc["h_sin"]
    d_cos_p = past_enc["d_cos"]
    d_sin_p = past_enc["d_sin"]
    h_cos_f = future_enc["h_cos"]
    h_sin_f = future_enc["h_sin"]
    d_cos_f = future_enc["d_cos"]
    d_sin_f = future_enc["d_sin"]

    # Encoder: h_cos, h_sin, d_cos, d_sin, GTI(i-1), temp, pressure, rel_hum, wind_speed, rain, cloud_cover
    encoder_np = np.vstack([
        h_cos_p,
        h_sin_p,
        d_cos_p,
        d_sin_p,
        _norm(past["global_tilted_irradiance"][0:-1], means[0], sigmas[0]), # the last element is kept for the decoder
        _norm(past["temperature_2m"],                 means[1], sigmas[1]),
        _norm(past["surface_pressure"],               means[2], sigmas[2]),
        _norm(past["relative_humidity_2m"],           means[3], sigmas[3]),
        _norm(past["wind_speed_10m"],                 means[4], sigmas[4]),
        _norm(past["rain"],                           means[5], sigmas[5]),
        _norm(past["cloud_cover"],                    means[6], sigmas[6])
    ]).T  # (96, 11)

    # Decoder: h_cos, h_sin, d_cos, d_sin, GTI(i-1), temp, cloud_cover
    # Leave GTI(i-1) with zeros, as it used only as a placeholder,
    # except the first one.
    gti_decoder = np.zeros(48, dtype=np.float32)
    gti_decoder[0] = _norm(past["global_tilted_irradiance"][-1:], means[0], sigmas[0])[0]

    decoder_np = np.vstack([
        h_cos_f,
        h_sin_f,
        d_cos_f,
        d_sin_f,
        gti_decoder,
        _norm(future["temperature_2m"], means[1], sigmas[1]),
        _norm(future["cloud_cover"],    means[6], sigmas[6]),
    ]).T  # (48, 7)

    # The interference is for one element only, but the framework needs it.
    encoder_in = torch.from_numpy(encoder_np).unsqueeze(0)  # (1, 96, 11)
    decoder_in = torch.from_numpy(decoder_np).unsqueeze(0)  # (1, 48, 7)

    return encoder_in, decoder_in

def postprocess_predictions(
    predictions: np.ndarray,
    anchor: datetime,
    lat: float,
    lon: float,
) -> np.ndarray:
    """Postprocess the predictions. Any negative values are set to zero
    and the sun's horizon is used to clip positive values during nighttime.

    Args:
        predictions: Raw output from the model (48 values).
        anchor:      Start datetime (UTC).
        lat:         Latitude in decimal degrees.
        lon:         Longitude in decimal degrees.

    Returns:
        Postprocessed predictions with a boolean array of daytime hours.
    """
    times = pd.date_range(start=pd.Timestamp(anchor), periods=48, freq="1h")
    solar_pos = pvlib.solarposition.get_solarposition(times, lat, lon)
    is_daytime = np.where(solar_pos["apparent_elevation"].to_numpy() >= 0, 1, 0)
    return np.where(predictions > 0, predictions, 0) * is_daytime

def do_predict(
    model: SolarSeq2SeqGRU,
    lat: float,
    lon: float,
    date: str,
    hour: int,
    means: np.ndarray,
    sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict the GTI from gathered weather data from OpenMeteo.

    Args:
        model:  Loaded SolarSeq2SeqGRU model.
        lat:    Latitude in decimal degrees.
        lon:    Longitude in decimal degrees.
        date:   Reference date in YYYY-MM-DD format.
        hour:   Reference hour in 0-23 range.
        means:  Mean for every variable that is passed to the network.
        sigmas: Sigma for every variable that is passed to the network.

    Returns:
        Predicted output and ground truth (according to OpenMeteo).
    """
    anchor = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]), hour, tzinfo=timezone.utc)
    past_meteodata, future_meteodata = fetch_weather_data(lat, lon, anchor)
    enc_past_ts, end_future_ts = compute_time_encodings(anchor)
    
    # Construct the torch tensors
    encoder_in, decoder_in = build_tensors(
        past_meteodata,
        future_meteodata,
        enc_past_ts,
        end_future_ts,
        means,
        sigmas)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forecast = model.forecast(encoder_in, decoder_in, device)

    # The index at 0 is the mean for GTI.
    return postprocess_predictions(forecast * sigmas[0] + means[0], anchor, lat, lon), \
           future_meteodata["global_tilted_irradiance"]

def _generate_xaxis_labels(time_steps: int, dt: float):
    times_str = []
    for i in range(time_steps):
        time_hours = i * dt
        h = int(time_hours) % 24
        mins = round((time_hours % 1) * 60)
        if mins == 60:
            mins = 0
            h = (h + 1) % 24
        times_str.append(f"{h}.{mins:02d}")
    return times_str

def plot_scheduled_data(
    result: scipy.optimize.result,
    ts: int,
    dt: float,
    params: SolarMilpModelParameters
) -> None:
    x = result.x
    P_grid = x[ts : 2 * ts]
    P_solar = x[2 * ts : 3 * ts]
    P_bat = x[3 * ts : 4 * ts]
    BC = x[5 * ts : 6 * ts]
    BC = np.hstack((params.initial_battery_capacity, BC))
    # Time labels for x-axis: "0.00", "1.00", etc.
    times_labels = _generate_xaxis_labels(ts, dt)

    fig, ax = plt.subplots()
    ax.step(range(ts), P_grid, label="Grid Power", where="post", linewidth=2.4, color='blue')
    ax.step(range(ts), P_solar, label="Solar Used", where="post", linewidth=2, color='orange', linestyle='--')
    ax.step(range(ts), P_bat, label="Battery Power", where="post", linewidth=1.7, color='green', linestyle=':')
    ax.step(range(ts), p_load, label="Household Power", where="post", linewidth=2, color='red', linestyle='-.')

    ax.set_title("Optimal Household Power Flows for 24h", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time of Day (h)", fontsize=12)
    ax.set_ylabel("Power / Energy (kW, kWh)", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticks(range(ts))
    ax.set_xticklabels(times_labels, rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rich.print("[cyan]Welcome to my ECE57000 Project![/cyan]")
    _warn_print("Note the model is trained on weather data that is close to Budapest!")
    _warn_print("Use weather data that are from a nearby location!")
    weights_path: str = str(files("solar_gru") / "solar_gru_weights.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model = SolarSeq2SeqGRU(weights=weights)
    _ok_print("Weights loaded and model initialized!")

    try:
        lat = float(input("Enter latitude in float (degrees): "))
        if lat < -90.0 or lat > 90.0:
            raise ValueError("Lat is ∈ [-90; 90] degrees!")
        lon = float(input("Enter longitude in float: "))
        if lon < -180.0 or lon > 180.0:
            raise ValueError("Lon is ∈ [-180, 180] degrees!")

        date = str(input("Enter a date (YYYY-MM-DD format): "))
        # If it throws an exception, there's smth wrong
        _ = int(date[0:4])
        _ = int(date[5:7])
        _ = int(date[8:10])
        if date[4] != '-' or date[7] != '-':
            raise ValueError("Incorrect date!")
        hour = int(input("Enter a hour (0-23): "))
    except ValueError as e:
        _err_print(f"Reason: {str(e)}")
        _err_print("Enter a valid value!")
        sys.exit(1)
    except:
        sys.exit(1)

    # Load the means and sigmas of the dataset variables
    means_sigmas = np.load(str(files("solar_gru") / "openmeteo_means_sigmas.npz"))
    means = means_sigmas["means"]
    sigmas = means_sigmas["sigmas"]
    prediction, ground_truth = \
        do_predict(model, lat, lon, date, hour, means, sigmas)
    # Map the predicted output to real power
    mapper_creator: Model2SolarPlantMapperCreator = SolarPowerMapperCreator()
    mapper = mapper_creator.create_mapper()
    solar_farm_power = mapper.map(np.zeros(len(prediction)), prediction)
    # Now we have everything ready, construct the milp model and plan optimal battery charging

    print()
    _ok_print("Model prediction successful, scheduling now!")

    """
    The inverter efficiency is 95%, as well the charging
    and discharging of the battery.
    We suppose 1 kWh of electricity costs 0.20 USD,
    we get paid 0.05 USD for every kWh.
    Charging power is set to capacity/5.
    """
    # the reference power is only for 24h, double it
    p_load = np.hstack((reference_power_consumption, reference_power_consumption))
    battery_capacity = 8 # kWh
    milp_scheduler_params = SolarMilpModelParameters(
        p_load=p_load,
        p_solaravail=solar_farm_power,
        eff_solar=0.95,
        eff_battery_chg=0.95,
        eff_battery_dis=0.95,
        grid_price_buy=0.20,
        grid_price_sell=0.05,
        p_grid_bound=(-24, 24),
        p_solar_bound=7,
        p_bat_bound=(-battery_capacity/5, battery_capacity/5),
        battery_capacity=battery_capacity,
        initial_battery_capacity=2
    )

    schedule_model = SolarMILPModel(time_steps=48, dt=1, params=milp_scheduler_params)
    result: scipy.optimize.OptimizeResult = schedule_model.solve()
    if result.success:
        _ok_print("Feasible solution found!")
        ts = schedule_model.time_steps
        dt = schedule_model.dt
        plot_scheduled_data(result, ts, dt, milp_scheduler_params)
    else:
        _err_print("Solver was unable to find a feasible solution.")
