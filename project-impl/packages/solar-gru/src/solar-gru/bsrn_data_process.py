import csv
import glob
import os
import numpy as np
import pandas as pd
import pvlib
from typing import List, Tuple

# Budapest, Hungary Solar measurement station coordinates.
LAT = 47.4291
LON = 19.1822

CONSECUTIVE_DAYS = 4
DUMMY = 11235813

class SolarFileData:
    def __init__(self, timestamps, direct_rad, diffuse_rad, global_rad, temperature, humidity, pressure):
        self.timestamps  = timestamps
        self.direct_rad  = direct_rad
        self.diffuse_rad = diffuse_rad
        self.global_rad  = global_rad   # SWD [W/m²], column 3 (index 2)
        self.temperature = temperature  # T2 [°C], column 27 (index 26)
        self.humidity    = humidity     # RH [%],  column 28 (index 27)
        self.pressure    = pressure     # PoPoPoPo [hPa], column 29 (index 28)

class RadiosondeFileData:
    def __init__(self, timestamps, altitude, pressure, temperature, dew_point, wind_speed):
        self.timestamps  = timestamps
        self.altitude    = altitude
        self.pressure    = pressure
        self.temperature = temperature
        self.dew_point   = dew_point
        self.wind_speed  = wind_speed

def date_format_to_year_and_minutes(s: str):
    # The format is yyyy-MM-ddTmm:ss
    MM   = int(s[5:7])
    dd   = int(s[8:10])
    yyyy = int(s[0:4])
    hh   = int(s[11:13])
    mm   = int(s[14:16])
    return (MM, dd, yyyy, hh, mm)

def _days_before_month(yyyy: int, MM: int) -> int:
    days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (yyyy % 4 == 0 and yyyy % 100 != 0) or (yyyy % 400 == 0):
        days[2] = 29
    return sum(days[1:MM])

def _minutes_to_date_str(total_minutes: int) -> str:
    total_days, rem = divmod(int(total_minutes), 24 * 60)
    hh, mm = divmod(rem, 60)

    def _days_since_year0(y, m, d):
        total = y * 365 + y // 4 - y // 100 + y // 400
        total += _days_before_month(y, m) + (d - 1)
        return total

    abs_days = total_days + _days_since_year0(1970, 1, 1)

    yyyy = int(abs_days / 365.2425)
    while _days_since_year0(yyyy + 1, 1, 1) <= abs_days:
        yyyy += 1
    while _days_since_year0(yyyy, 1, 1) > abs_days:
        yyyy -= 1

    MM = 1
    while MM < 12 and _days_since_year0(yyyy, MM + 1, 1) <= abs_days:
        MM += 1

    dd = abs_days - _days_since_year0(yyyy, MM, 1) + 1
    return f"{yyyy:04d}-{MM:02d}-{dd:02d}T{hh:02d}:{mm:02d}"

def _to_total_minutes(s: str) -> int:
    MM, dd, yyyy, hh, mm = date_format_to_year_and_minutes(s)
    # Days since Unix epoch (1970-01-01). Uses proleptic Gregorian day count.
    def _days_since_epoch(y, m, d):
        total = y * 365 + y // 4 - y // 100 + y // 400
        total += _days_before_month(y, m) + (d - 1)
        return total
    epoch_days = _days_since_epoch(1970, 1, 1)
    return (_days_since_epoch(yyyy, MM, dd) - epoch_days) * 24 * 60 + hh * 60 + mm

def _parse_tab_file(filepath: str):
    """
    Yields parsed parts (list[str]) for each data row in a BSRN .tab file,
    skipping the /* ... */ comment block and the column header row.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        in_comment = False
        header_skipped = False
        for line in f:
            if line.startswith("/*"):
                in_comment = True
            if in_comment:
                if "*/" in line:
                    in_comment = False
                continue
            if not header_skipped:
                header_skipped = True
                continue
            yield line.rstrip("\n").split("\t")


def _parse_float(s: str) -> float:
    return float(s) if s != '' else DUMMY

def _load_radiation(radiation_path: str) -> SolarFileData:
    timestamps = []
    direct_values, diffuse_values, global_values = [], [], []
    temperature_values, humidity_values, pressure_values = [], [], []

    for parts in _parse_tab_file(radiation_path):
        if len(parts) < 29:
            continue
        timestamps.append(parts[0])
        global_values.append(_parse_float(parts[2]))
        direct_values.append(_parse_float(parts[6]))
        diffuse_values.append(_parse_float(parts[10]))
        temperature_values.append(_parse_float(parts[26]))
        humidity_values.append(_parse_float(parts[27]))
        pressure_values.append(_parse_float(parts[28]))

    direct_rad  = np.maximum(np.array(direct_values,  dtype=np.float64), 0)
    diffuse_rad = np.maximum(np.array(diffuse_values, dtype=np.float64), 0)
    global_rad  = np.maximum(np.array(global_values,  dtype=np.float64), 0)
    return SolarFileData(
        timestamps, direct_rad, diffuse_rad, global_rad,
        np.array(temperature_values, dtype=np.float64),
        np.array(humidity_values,    dtype=np.float64),
        np.array(pressure_values,    dtype=np.float64),
    )

def _load_radiosonde(radiosonde_path: str) -> RadiosondeFileData:
    timestamps = []
    altitude, pressure, temperature, dew_point, wind_speed = [], [], [], [], []

    for parts in _parse_tab_file(radiosonde_path):
        if len(parts) < 7:
            continue
        timestamps.append(parts[0])
        # altitude(1), pressure(2), temp(3), dew point(4), wind speed(6) — wind dir(5) skipped
        altitude.append(_parse_float(parts[1]))
        pressure.append(_parse_float(parts[2]))
        temperature.append(_parse_float(parts[3]))
        dew_point.append(_parse_float(parts[4]))
        wind_speed.append(_parse_float(parts[6]))

    return RadiosondeFileData(
        timestamps,
        np.array(altitude,     dtype=np.float64),
        np.array(pressure,     dtype=np.float64),
        np.array(temperature,  dtype=np.float64),
        np.array(dew_point,    dtype=np.float64),
        np.array(wind_speed,   dtype=np.float64),
    )

# Accumulates hourly arrays across chunks until a gap > 1 hour is detected.
# Tuple of 8 np.ndarray: (timestamps, direct, diffuse, global, temperature, humidity, pressure, zenith)
_pending_chunk = None

_SAVED_DIR = os.path.join(os.path.dirname(__file__), "dataset-bsrn", "saved")

def _write_chunk(arrays):
    """Write a finalized tuple of 8 hourly arrays to disk."""
    ts, direct, diffuse, global_r, temperature, humidity, pressure, zenith = arrays

    existing = glob.glob(os.path.join(_SAVED_DIR, "chunk*.npz"))
    used_ids = set()
    for f in existing:
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            used_ids.add(int(base[len("chunk"):]))
        except ValueError:
            pass
    chunk_id = 0
    while chunk_id in used_ids:
        chunk_id += 1

    filename = f"chunk{chunk_id}.npz"
    np.savez(
        os.path.join(_SAVED_DIR, filename),
        timestamps=ts,
        direct_rad=direct,
        diffuse_rad=diffuse,
        global_rad=global_r,
        temperature=temperature,
        humidity=humidity,
        pressure=pressure,
        zenith_angle=zenith,
    )

    start_date = _minutes_to_date_str(ts[0])
    end_date   = _minutes_to_date_str(ts[-1])
    with open(os.path.join(_SAVED_DIR, "dates.txt"), "a", newline="") as f:
        csv.writer(f).writerow([filename, start_date, end_date])

    print(f"Saved {filename}  [{start_date} → {end_date}]  len={len(ts)}")

def flush_pending_chunk():
    """Flush any accumulated pending chunk to disk. Call after all files are processed."""
    global _pending_chunk
    if _pending_chunk is not None:
        _write_chunk(_pending_chunk)
        _pending_chunk = None

def save_postprocess_chunk(radiation: SolarFileData):
    global _pending_chunk

    # Ignore retarded files.
    if len(radiation.timestamps) < 24 * 60 * CONSECUTIVE_DAYS:
        return

    # Align to the first on-the-hour timestamp (divisible by 60)
    n = len(radiation.timestamps)
    start = 0
    while start < n and radiation.timestamps[start] % 60 != 0:
        start += 1

    # Build hourly arrays by averaging each 60-minute block
    hourly_timestamps   = []
    hourly_direct       = []
    hourly_diffuse      = []
    hourly_global       = []
    hourly_temperature  = []
    hourly_humidity     = []
    hourly_pressure     = []
    i = start
    while i + 60 <= n:
        date_str = _minutes_to_date_str(radiation.timestamps[i])
        if date_str[5:10] == "02-29":  # skip Feb 29
            i += 60
            continue
        hourly_timestamps.append(radiation.timestamps[i])
        hourly_direct.append(np.mean(radiation.direct_rad[i:i + 60]))
        hourly_diffuse.append(np.mean(radiation.diffuse_rad[i:i + 60]))
        hourly_global.append(np.mean(radiation.global_rad[i:i + 60]))
        hourly_temperature.append(np.mean(radiation.temperature[i:i + 60]))
        hourly_humidity.append(np.mean(radiation.humidity[i:i + 60]))
        hourly_pressure.append(np.mean(radiation.pressure[i:i + 60]))
        i += 60

    if len(hourly_timestamps) < 24 * CONSECUTIVE_DAYS:
        return

    new_ts = np.array(hourly_timestamps, dtype=np.int64)

    times = pd.to_datetime(new_ts.astype(np.int64) * 60, unit='s', utc=True)
    solar_pos = pvlib.solarposition.get_solarposition(times, LAT, LON)
    zenith = solar_pos['zenith'].to_numpy(dtype=np.float64)

    new_arr  = (
        new_ts,
        np.array(hourly_direct,      dtype=np.float64),
        np.array(hourly_diffuse,     dtype=np.float64),
        np.array(hourly_global,      dtype=np.float64),
        np.array(hourly_temperature, dtype=np.float64),
        np.array(hourly_humidity,    dtype=np.float64),
        np.array(hourly_pressure,    dtype=np.float64),
        zenith,
    )

    if _pending_chunk is not None:
        gap = new_ts[0] - _pending_chunk[0][-1]
        if gap == 60:
            # Consecutive — merge into pending
            _pending_chunk = tuple(np.concatenate([p, n]) for p, n in zip(_pending_chunk, new_arr))
            return
        else:
            # Gap too large — flush pending and start fresh
            _write_chunk(_pending_chunk)

    _pending_chunk = new_arr

def postprocess_file2(radiation: SolarFileData, radiosonde: RadiosondeFileData):
    # Ignore data that have less than CONSECUTIVE_DAYS
    if len(radiation.timestamps) < 24 * 60 * 4:
        return

    n = len(radiation.timestamps)
    segment_start = 0

    for i in range(n):
        if (radiation.direct_rad[i] == DUMMY or radiation.diffuse_rad[i] == DUMMY or radiation.global_rad[i] == DUMMY
                or radiation.temperature[i] == DUMMY or radiation.humidity[i] == DUMMY or radiation.pressure[i] == DUMMY):
            if i > segment_start:
                seg = SolarFileData(
                    radiation.timestamps[segment_start:i],
                    radiation.direct_rad[segment_start:i],
                    radiation.diffuse_rad[segment_start:i],
                    radiation.global_rad[segment_start:i],
                    radiation.temperature[segment_start:i],
                    radiation.humidity[segment_start:i],
                    radiation.pressure[segment_start:i],
                )
                save_postprocess_chunk(seg)
            # skip forward until all channels are valid again
            segment_start = i + 1
        elif i == n - 1:
            seg = SolarFileData(
                radiation.timestamps[segment_start:n],
                radiation.direct_rad[segment_start:n],
                radiation.diffuse_rad[segment_start:n],
                radiation.global_rad[segment_start:n],
                radiation.temperature[segment_start:n],
                radiation.humidity[segment_start:n],
                radiation.pressure[segment_start:n],
            )
            save_postprocess_chunk(seg)


def postprocess_file(radiation: SolarFileData, radiosonde: RadiosondeFileData):
    # Convert string timestamps to minutes since Unix epoch
    radiation.timestamps  = np.array([_to_total_minutes(t) for t in radiation.timestamps],  dtype=np.int64)
    radiosonde.timestamps = np.array([_to_total_minutes(t) for t in radiosonde.timestamps], dtype=np.int64)

    segment_start = 0
    n = len(radiation.timestamps)

    for i in range(1, n):
        if radiation.timestamps[i] - radiation.timestamps[i - 1] != 1:
            seg = SolarFileData(
                radiation.timestamps[segment_start:i],
                radiation.direct_rad[segment_start:i],
                radiation.diffuse_rad[segment_start:i],
                radiation.global_rad[segment_start:i],
                radiation.temperature[segment_start:i],
                radiation.humidity[segment_start:i],
                radiation.pressure[segment_start:i],
            )
            postprocess_file2(seg, radiosonde)
            segment_start = i

    # End of timestamps: flush remaining segment
    seg = SolarFileData(
        radiation.timestamps[segment_start:n],
        radiation.direct_rad[segment_start:n],
        radiation.diffuse_rad[segment_start:n],
        radiation.global_rad[segment_start:n],
        radiation.temperature[segment_start:n],
        radiation.humidity[segment_start:n],
        radiation.pressure[segment_start:n],
    )
    postprocess_file2(seg, radiosonde)


def load_bsrn_files(radiation_path: str, radiosonde_path: str):
    radiation  = _load_radiation(radiation_path)
    radiosonde = _load_radiosonde(radiosonde_path)
    postprocess_file(radiation, radiosonde)
    return radiation, radiosonde

def load_bsrn_year(year: str):
    data_dir = os.path.join(os.path.dirname(__file__), "dataset-bsrn")
    all_files = [
        (os.path.join(data_dir, "radiation",  f"BUD_radiation_{year}-{m:02d}.tab"),
         os.path.join(data_dir, "radiosonde", f"BUD_radiosonde_{year}-{m:02d}.tab")) for m in range(1, 13)
    ]
    all_files = [f for f in all_files if os.path.isfile(f[0])]
    all_files.sort()
    for rad_path, rs_path in all_files:
        radiation, radiosonde = load_bsrn_files(rad_path, rs_path)
    print(f"Succesfully loaded year {year}.")

if __name__ == "__main__":
    saved_dir = os.path.join(os.path.dirname(__file__), "dataset-bsrn", "saved")
    os.makedirs(saved_dir, exist_ok=True)
    for f in glob.glob(os.path.join(saved_dir, "*")):
        os.remove(f)

    for y in ["2020", "2021", "2022", "2023", "2024", "2025"]:
        load_bsrn_year(y)

    flush_pending_chunk()
