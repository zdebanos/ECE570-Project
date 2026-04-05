import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from bsrn_data_process import _minutes_to_date_str

def check(chunk_idx):
    path_to_file = os.path.join(os.path.dirname(__file__), "dataset-bsrn", "saved", f"chunk{chunk_idx}.npz")
    if not os.path.isfile(path_to_file):
        raise ValueError("Invalid file")

    data = np.load(path_to_file)
    timestamps  = data["timestamps"]
    direct_rad  = data["direct_rad"]
    diffuse_rad = data["diffuse_rad"]
    zenith      = data["zenith_angle"]
    global_rad  = data["global_rad"]

    radiation_total = diffuse_rad + np.maximum(np.cos(np.radians(zenith)), 0) * direct_rad

    start_date = _minutes_to_date_str(timestamps[0])
    end_date   = _minutes_to_date_str(timestamps[-1])

    fig, ax = plt.subplots(figsize=(14, 5))

    #ax.plot(direct_rad,  color="#E05C2A", linewidth=0.9, label="Direct radiation (DIR)")
    #ax.plot(diffuse_rad, color="#2A7BE0", linewidth=0.9, label="Diffuse radiation (DIF)")
    ax.plot(radiation_total, color="#E05C2A", linewidth=0.9, label="My calc")
    ax.plot(global_rad, color="#2A7BE0", linewidth=0.9, label="Global")

    ax.set_title(f"Chunk {chunk_idx} — Hourly radiation  [{start_date} → {end_date}]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour index", fontsize=10)
    ax.set_ylabel("Irradiance [W/m²]", fontsize=10)

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    ax.minorticks_on()

    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_xlim(0, len(direct_rad) - 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    saved_dir = os.path.join(os.path.dirname(__file__), "dataset-bsrn", "saved")
    chunks = sorted(
        int(os.path.splitext(f)[0][len("chunk"):])
        for f in os.listdir(saved_dir)
        if f.startswith("chunk") and f.endswith(".npz")
    )
    for idx in chunks:
        check(idx)
