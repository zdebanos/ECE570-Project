import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def check(name: str) -> None:
    path_to_file = os.path.join(os.path.dirname(__file__), "dataset-openmeteo", "saved", f"{name}.npz")
    if not os.path.isfile(path_to_file):
        raise ValueError(f"File not found: {path_to_file}")

    data = np.load(path_to_file)
    gti = data["global_tilted_irradiance"]
    temp = data["temperature_2m"]
    hum  = data["relative_humidity_2m"]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(gti, color="#E05C2A", linewidth=0.9, label="Global tilted irradiance (GTI)")

    ax.set_title(f"{name} — Hourly solar radiation (GTI)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hour index", fontsize=10)
    ax.set_ylabel("Irradiance [W/m²]", fontsize=10)

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    ax.minorticks_on()

    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_xlim(0, len(gti) - 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    saved_dir = os.path.join(os.path.dirname(__file__), "dataset-openmeteo", "saved")
    names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(saved_dir)
        if f.endswith(".npz")
    )
    for name in names:
        if name != "Budapest":
            check(name)
