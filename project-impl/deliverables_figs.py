import torch
import numpy as np 
from importlib.resources import files
from solar_scheduling import do_predict
from solar_gru.solar_seq2seq_gru import SolarSeq2SeqGRU
import matplotlib.pyplot as plt

if __name__ == "__main__":
    weights_path: str = str(files("solar_gru") / "solar_gru_weights.pth")
    weights = torch.load(weights_path, map_location="cpu")
    model = SolarSeq2SeqGRU(weights=weights)
    # Load the means and sigmas of the dataset variables
    means_sigmas = np.load(str(files("solar_gru") / "openmeteo_means_sigmas.npz"))
    means = means_sigmas["means"]
    sigmas = means_sigmas["sigmas"]
    lat = 47.49698
    lon = 19.06005

    days = []
    months = []
    dates = []
    new = list(range(16, 28+1))   # february
    days.extend(new)
    months.extend([2] * len(new))
    new = list(range(1, 31+1))    # march
    days.extend(new)
    months.extend([3] * len(new))
    new = list(range(1, 21+1))    # april
    days.extend(new)
    months.extend([4] * len(new))
    
    for i in range(len(days)):
        dates.append(f"2026-{months[i]:02d}-{days[i]:02d}")

    mae_arr = np.zeros(len(dates) * 24)

    idx = 0
    for d in dates:
        for h in range(1, 24):
            print(f"Inferencing for {d} at hour {h}.")
            prediction, ground_truth = do_predict(model, lat, lon, d, h, means, sigmas)
            mae_arr[idx] = 1/48 * np.sum(np.abs(prediction - ground_truth))
            idx += 1

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(mae_arr, bins=30, color="#f4a7a3", edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.axvline(mae_arr.mean(), color="#e67e22", linewidth=1.5,
               linestyle="--", label=f"Mean: {mae_arr.mean():.1f} W/m²")
    ax.set_title("MAE Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("MAE (W/m²)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gti_errors.png", dpi=300)
