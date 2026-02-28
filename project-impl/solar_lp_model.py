import numpy as np
import scipy
import time

class SolarLPModel:
    def __init__(self,
                 time_steps: int,
                 dt: float,
                 p_load: np.ndarray,
                 p_solaravail: np.ndarray,
                 eff_solar: float,
                 eff_battery_chg: float,
                 eff_battery_dis: float,
                 grid_price_buy: float,
                 grid_price_sell: float,
                 p_grid_bound: tuple[float, float],
                 p_solar_bound: float,
                 p_bat_bound: tuple[float, float],
                 battery_capacity: float,
                 initial_battery_capacity: float):
        """
        Linear Programming model for optimal solar energy scheduling.

        Args:
            time_steps (int): Number of discrete time steps.
            dt (float): Duration of each time step (hours).
            p_load (np.ndarray): Household load at each time step (kW).
            p_solaravail (np.ndarray): Solar power available at each time step (kW).
            eff_solar (float): Efficiency of solar power conversion [0,1].
            eff_battery_chg (float): Battery charging efficiency [0,1].
            eff_battery_dis (float): Battery discharging efficiency [0,1].
            grid_price_buy (float): Price for buying electricity from the grid ($/kWh).
            grid_price_sell (float): Price received for selling electricity to grid ($/kWh).
            p_grid_bound (tuple): Min/max grid power at each timestep (kW).
            p_solar_max (float): Max allowed solar power usage (kW).
            p_solar_min (tuple): Min/max allowed solar power usage (kW).
            battery_capacity (float): Usable battery capacity (kWh).
        """

        if grid_price_buy <= 0 or grid_price_sell <= 0:
            raise ValueError("Grid prices must be positive.")
        if grid_price_sell > grid_price_buy:
            raise ValueError("Grid price sell must be less than grid price buy.")
        if eff_battery_chg > 1 or eff_battery_dis > 1:
            raise ValueError("Invalid efficiency values. Must be between 0 and 1.")
        if p_grid_bound[0] >= p_grid_bound[1]:
            raise ValueError("Grid bound must be a tuple of two positive values with the first value less than the second.")
        if p_solar_bound <= 0:
            raise ValueError("Solar bound must be positive.")
        if battery_capacity <= 0:
            raise ValueError("Battery capacity must be positive.")
        if initial_battery_capacity <= 0 or initial_battery_capacity > battery_capacity:
            raise ValueError("Initial battery capacity must be positive and less than battery capacity.")
        if p_bat_bound[0] >= p_bat_bound[1]:
            raise ValueError("Battery bound must be a tuple of two positive values with the first value less than the second.")
        if p_bat_bound[0] > 0:
            raise ValueError("Battery lower bound must be non-positive.")
        if p_bat_bound[1] < 0:
            raise ValueError("Battery upper bound must be non-negative.")
    
        self.time_steps = time_steps
        self.dt = dt
        self.p_load = p_load
        self.p_solaravail = p_solaravail
        self.eff_solar = eff_solar
        self.eff_battery_chg = eff_battery_chg
        self.eff_battery_dis = eff_battery_dis
        self.grid_price_buy = grid_price_buy
        self.grid_price_sell = grid_price_sell
        self.p_grid_bound = p_grid_bound
        self.p_solar_bound = p_solar_bound
        self.p_bat_bound = p_bat_bound
        self.battery_capacity = battery_capacity
        self.initial_battery_capacity = initial_battery_capacity

        self.p_load = self.p_load.reshape(-1, 1)
        self.p_solaravail = self.p_solaravail.reshape(-1, 1)

        if self.p_load.size != self.time_steps:
            raise ValueError("Load must be an array of size time_steps.")
        if self.p_solaravail.size != self.time_steps:
            raise ValueError("Solar availability must be an array of size time_steps.")
        if self.initial_battery_capacity < self.p_bat_bound[0]:
            raise ValueError("The author of this program is a lazy retard and forgot to implement the initial battery capacity check. Just use something slightly higher than the lower bound.")

    def _setup_variables(self):
        np.set_printoptions(threshold=np.inf, linewidth=400)
        """
        We define the optimization vector as follows:
        x = | EC      |
            | P_grid  |
            | P_solar |
            | P_bat   |
            | BC(2-n) |.
        The equations are as follows:
        - Inequality ones:
          PB P_grid       - EC      <= 0
          PS P_grid       - EC      <= 0
        - Equality ones:
          P_grid  + eta_solar P_solar + P_bat = P_load
          BC(i+1) + P_bat(i) T        - BC(i) = 0
        Bounds:
          lb = | None       | ub = | None         |
               | P_grid^MIN |      | P_grid^MAX   |
               | 0          |      | P_solaravail |
               | P_bat^MIN  |      | P_bat^MAX    |
               | 0.1 BC^MAX |      | 0.9 BC^MAX   |
        """
        ts = self.time_steps
        # Criterion Vector (c)
        self.c = np.zeros((5*ts, 1))
        self.c[0:ts] = 1

        # Nonequality Matrix (A_ub)
        self.aub = np.zeros((2*ts, 5*ts))
        # 1st eq
        self.aub[0:ts, 0:ts]    = -np.eye(ts)
        self.aub[0:ts, ts:2*ts] = self.grid_price_buy * np.eye(ts)
        # 2nd eq
        self.aub[ts:2*ts, 0:ts]    = -np.eye(ts)
        self.aub[ts:2*ts, ts:2*ts] = self.grid_price_sell * np.eye(ts)

        # Nonequality Vector (b_ub)
        self.bub = np.zeros((2 * ts, 1))
        
        # Equality Matrix (A_eq)
        self.aeq = np.zeros((ts, 5*ts))
        # 1st eq
        self.aeq[0:ts, ts:2*ts]   = np.eye(ts)
        self.aeq[0:ts, 2*ts:3*ts] = self.eff_solar * np.eye(ts)
        self.aeq[0:ts, 3*ts:4*ts] = np.eye(ts)
        # 2nd eq
        col_offset = 3*ts
        self.aeq_2 = np.zeros((ts, 5*ts))
        self.aeq_2[0:ts, col_offset:col_offset+ts] = self.dt * np.eye(ts)
        self.aeq_2[0:ts, col_offset+ts:col_offset+2*ts] = np.eye(ts) - np.eye(ts, k=-1)
        self.aeq = np.concatenate((self.aeq, self.aeq_2), axis=0)

        # Equality Vector (b_eq)
        self.beq = np.zeros((2 * ts, 1))
        self.beq[0:ts, 0] = self.p_load.flatten()
        self.beq[ts, 0] = self.initial_battery_capacity

        # Lower Bound Vector (lb)
        self.lb = 5*ts * [None]
        self.lb[ts:2*ts]   = ts * [self.p_grid_bound[0]]
        self.lb[2*ts:3*ts] = ts * [0]
        self.lb[3*ts:4*ts] = ts * [self.p_bat_bound[0]]
        self.lb[4*ts:5*ts] = ts * [0.1 * self.battery_capacity]

        # Upper Bound Vector (ub)
        self.ub = 5*ts * [None]
        self.ub[ts:2*ts]   = ts * [self.p_grid_bound[1]]
        self.ub[2*ts:3*ts] = self.p_solaravail.reshape(1, -1).tolist()[0]
        self.ub[3*ts:4*ts] = ts * [self.p_bat_bound[1]]
        self.ub[4*ts:5*ts] = ts * [0.9 * self.battery_capacity]

    def solve(self) -> scipy.optimize.OptimizeResult:
        self._setup_variables()
        # Some circlejerk python list building bullshit
        bounds = list(zip(self.lb, self.ub))
        start_time = time.time()
        result = scipy.optimize.linprog(self.c, A_ub=self.aub, b_ub=self.bub, A_eq=self.aeq, b_eq=self.beq, \
                                        bounds=bounds, method='highs')
        print(f"Time taken: {time.time() - start_time} seconds")

        return result



