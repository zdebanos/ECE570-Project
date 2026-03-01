from platform import java_ver
import numpy as np
import scipy
import time
import warnings

class SolarMILPModel:
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
        Mixed Integer Linear Programming model for optimal solar energy scheduling.

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
        
        if np.max(self.p_solaravail) > self.p_solar_bound:
            warnings.warn("The maximum value in p_solaravail exceeds p_solar_bound.\n"
                          "Limiting p_solaravail to p_solar_bound.", stacklevel=2)
            self.p_solaravail = np.minimum(self.p_solaravail, self.p_solar_bound)

        # Print initial settings summary with values limited to 3 decimal places
        print("Model input parameters:")
        print(f"  Battery capacity: {battery_capacity:.3f} kWh")
        print(f"  Initial battery capacity: {self.initial_battery_capacity:.3f} kWh")
        print(f"  Grid price buy: {self.grid_price_buy:.3f} $/kWh")
        print(f"  Grid price sell: {self.grid_price_sell:.3f} $/kWh")
        print(f"  Battery charge efficiency: {100 * self.eff_battery_chg:.3f}%")
        print(f"  Battery discharge efficiency: {100 * self.eff_battery_dis:.3f}%")
        print(f"  Solar efficiency: {100 * self.eff_solar:.3f}%")
        print(f"  Grid power bounds: {self.p_grid_bound[0]:.3f} to {self.p_grid_bound[1]:.3f} kW")
        print(f"  Battery power bounds: {self.p_bat_bound[0]:.3f} to {self.p_bat_bound[1]:.3f} kW")
        print(f"  Max Solar usage: {self.p_solar_bound:.3f} kW")
        print("------------------------------------------------------------")

    @warnings.deprecated("Use _setup_variables_milp instead. Kept for reference.")
    def _setup_variables(self):
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
    
    def _setup_variables_milp(self):
        """
        We define the optimization vector as follows:
        x = | EC      |
            | P_grid  |
            | P_solar |
            | P_bat   |
            | P_charg |
            | BC(2-n) |
            | CH(2-n) | <------------- binary variables
        The equations are as follows:
        - Inequality ones:
          PB P_grid       - EC       <= 0               // Buying from the grid
          PS P_grid       - EC       <= 0               // Selling to the grid
          1/eta_dis P_bat            <= P_charg         // Discharging the battery (big M)
          eta_chg P_bat              <= P_charg         // Charging the battery (big M)
          P_charg         + M CH     <= 1/eta_dis P_bat // Charging the battery (big M)
          P_charg         + M (1-CH) <= eta_chg P_bat   // Charging the battery (big M)
        - Equality ones:
          P_grid  + eta_solar P_solar + P_bat = P_load // Demand is satisfied
          BC(i+1) + P_charg(i) T      - BC(i) = 0      // Battery charge state
        Bounds:
          lb = | None       | ub = | None         |
               | P_grid^MIN |      | P_grid^MAX   |
               | 0          |      | P_solaravail |
               | P_bat^MIN  |      | P_bat^MAX    |
               | None       |      | None         |
               | 0.1 BC^MAX |      | 0.9 BC^MAX   |
               | None (bin) |      | None (bin)   |
        """
        bigM = 1e7
        ts = self.time_steps
        self.c = np.zeros(7*ts)
        self.c[0:ts] = 1

        # Nonequality Matrix
        self.aub = np.zeros((6*ts, 7*ts))
        # 1st eq
        self.aub[0:ts, 0:ts]    = -np.eye(ts)
        self.aub[0:ts, ts:2*ts] = self.grid_price_buy * np.eye(ts)
        # 2nd eq
        self.aub[ts:2*ts, 0:ts]    = -np.eye(ts)
        self.aub[ts:2*ts, ts:2*ts] = self.grid_price_sell * np.eye(ts)
        # 3rd eq
        self.aub[2*ts:3*ts, 3*ts:4*ts] = 1/self.eff_battery_dis * np.eye(ts)
        self.aub[2*ts:3*ts, 4*ts:5*ts] = -np.eye(ts)
        # 4th eq
        self.aub[3*ts:4*ts, 3*ts:4*ts] = self.eff_battery_chg * np.eye(ts)
        self.aub[3*ts:4*ts, 4*ts:5*ts] = -np.eye(ts)
        # 5th eq
        self.aub[4*ts:5*ts, 3*ts:4*ts] = -1/self.eff_battery_dis * np.eye(ts)
        self.aub[4*ts:5*ts, 4*ts:5*ts] = np.eye(ts)
        self.aub[4*ts:5*ts, 6*ts:7*ts] = -bigM * np.eye(ts)
        # 6th eq
        self.aub[5*ts:6*ts, 3*ts:4*ts] = -self.eff_battery_chg * np.eye(ts)
        self.aub[5*ts:6*ts, 4*ts:5*ts] = np.eye(ts)
        self.aub[5*ts:6*ts, 6*ts:7*ts] = bigM * np.eye(ts)

        # Nonequality Vector
        self.bub = np.zeros(6*ts)
        # Handle the last equation with bigM
        self.bub[5*ts:6*ts] = bigM
        
        # Equality Matrix
        self.aeq = np.zeros((ts, 7*ts))
        # 1st eq
        self.aeq[0:ts, ts:2*ts]   = np.eye(ts)
        self.aeq[0:ts, 2*ts:3*ts] = self.eff_solar * np.eye(ts)
        self.aeq[0:ts, 3*ts:4*ts] = np.eye(ts)
        # 2nd eq
        col_offset = 4*ts
        self.aeq_2 = np.zeros((ts, 7*ts))
        self.aeq_2[0:ts, col_offset:col_offset+ts] = self.dt * np.eye(ts)
        self.aeq_2[0:ts, col_offset+ts:col_offset+2*ts] = np.eye(ts) - np.eye(ts, k=-1)
        self.aeq = np.concatenate((self.aeq, self.aeq_2), axis=0)

        # Equality Vector
        self.beq = np.zeros(2 * ts)
        self.beq[0:ts] = self.p_load.flatten()
        self.beq[ts] = self.initial_battery_capacity

        # Lower Bounds Vector
        self.lb = np.ones(7*ts) * -np.inf
        self.lb[ts:2*ts]   = self.p_grid_bound[0]
        self.lb[2*ts:3*ts] = 0
        self.lb[3*ts:4*ts] = self.p_bat_bound[0]
        self.lb[5*ts:6*ts] = 0.1 * self.battery_capacity
        self.lb[6*ts:7*ts] = 0 # integer binary

        # Upper Bounds Vector
        self.ub = np.ones(7*ts) * np.inf
        self.ub[ts:2*ts]   = self.p_grid_bound[1]
        self.ub[2*ts:3*ts] = self.p_solaravail.flatten()
        self.ub[3*ts:4*ts] = self.p_bat_bound[1]
        self.ub[5*ts:6*ts] = 0.9 * self.battery_capacity
        self.ub[6*ts:7*ts] = 1 # integer binary

        self.bounds = scipy.optimize.Bounds(self.lb, self.ub)

        # LinearConstraint expects lb/ub 1D (length = number of rows of A)
        self.eq_constr  = scipy.optimize.LinearConstraint(self.aeq, ub=self.beq, lb=self.beq)
        self.neq_constr = scipy.optimize.LinearConstraint(self.aub, ub=self.bub)

    def solve(self):
        ts = self.time_steps
        print("Running MILP Solver...")
        self._setup_variables_milp()
        integrality = np.zeros(7*ts)
        integrality[6*ts:7*ts] = 1   # integer variables
        t1 = time.time()
        res = scipy.optimize.milp(self.c, bounds=self.bounds, \
                                  constraints=[self.eq_constr, self.neq_constr], \
                                  integrality=integrality)
        print(f"MILP Solver Finished in {1000 * (time.time() - t1):.2f} ms.")
        return res
