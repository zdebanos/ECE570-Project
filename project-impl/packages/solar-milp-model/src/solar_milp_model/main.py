import numpy as np
import matplotlib.pyplot as plt
import scipy

from solar_milp_model.model import SolarMILPModel
from solar_milp_model.data import power_consumption, solar_summer

def generate_xaxis_labels(time_steps: int, dt: float):
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

def main():
    battery_capacity = 10 # kWh
    model = SolarMILPModel(
        time_steps=24,
        dt=1,
        p_load=power_consumption,
        p_solaravail=solar_summer,
        eff_solar=0.95,
        eff_battery_chg=0.95,
        eff_battery_dis=0.95,
        grid_price_buy=0.20,
        grid_price_sell=0.05,
        p_grid_bound=(-38, 38),
        p_solar_bound=7,
        p_bat_bound=(-battery_capacity/2, battery_capacity/2),
        battery_capacity=battery_capacity,
        initial_battery_capacity=2
    )
    result: scipy.optimize.OptimizeResult = model.solve()
    if result.success:
        print(f"Energy cost without solar: {np.sum(solar_summer)} * "
              f"{model.grid_price_buy:.2f} = {np.sum(solar_summer) * model.grid_price_buy:.2f} $")
        print(f"Energy cost with optimization: {result.fun:.2f} $")
        ts = model.time_steps
        dt = model.dt
        x = result.x
        P_grid = x[ts : 2 * ts]
        P_solar = x[2 * ts : 3 * ts]
        P_bat = x[3 * ts : 4 * ts]
        BC = x[5 * ts : 6 * ts]
        BC = np.hstack((model.initial_battery_capacity, BC))

        # Time labels for x-axis: "0.00", "1.00", ... or "0.00", "0.30", ... for 30-min steps
        times_labels = generate_xaxis_labels(ts, dt)

        fig, ax = plt.subplots()
        # Use different line styles and thicknesses for clarity
        ax.step(range(ts), P_grid, label="Grid Power", where="post", linewidth=2.4, color='blue')
        ax.step(range(ts), P_solar, label="Solar Used", where="post", linewidth=2, color='orange', linestyle='--')
        ax.step(range(ts), P_bat, label="Battery Power", where="post", linewidth=1.7, color='green', linestyle=':')
        ax.step(range(ts-1), BC[0:ts-1], label="Battery Charge", where="post", linewidth=2, color='red', linestyle='-.')

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
    else:
        raise ValueError("Optimization failed.")

if __name__ == "__main__":
    main()