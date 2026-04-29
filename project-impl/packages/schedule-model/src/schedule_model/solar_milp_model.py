import numpy as np
import scipy
import time
import warnings

from schedule_model.schedule_model_base import ScheduleModel, ScheduleModelParameters
from typing import cast

class SolarMilpModelParameters(ScheduleModelParameters):
    def __init__(
        self,
        p_load: np.ndarray,
        p_solaravail: np.ndarray,
        grid_prices: np.ndarray,
        eff_solar: float,
        eff_battery_chg: float,
        eff_battery_dis: float,
        p_grid_bound: tuple[float, float],
        p_solar_bound: float,
        p_bat_bound: tuple[float, float],
        battery_capacity: float,
        initial_battery_capacity: float
    ):
        """Mixed Integer Linear Programming model parameters.

        Args:
            p_load (np.ndarray): Household load at each time step (kW).
            p_solaravail (np.ndarray): Solar power available at each time step (kW).
            grid_prices (np.ndarray): Grid energy price at each time step ($/kWh).
            eff_solar (float): Efficiency of solar power conversion [0,1].
            eff_battery_chg (float): Battery charging efficiency [0,1].
            eff_battery_dis (float): Battery discharging efficiency [0,1].
            p_grid_bound (tuple): Min/max grid power at each timestep (kW).
            p_solar_bound (float): Max solar inverter power at each timestep (kW).
            p_bat_bound (tuple): Allowed ranges of battery charges.
            battery_capacity (float): Usable battery capacity (kWh).
            initial_battery_capacity (float): Battery capacity at first timestamp.
        """
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

        self.p_load = p_load
        self.p_solaravail = p_solaravail
        self.grid_prices = grid_prices
        self.eff_solar = eff_solar
        self.eff_battery_chg = eff_battery_chg
        self.eff_battery_dis = eff_battery_dis
        self.p_grid_bound = p_grid_bound
        self.p_solar_bound = p_solar_bound
        self.p_bat_bound = p_bat_bound
        self.battery_capacity = battery_capacity
        self.initial_battery_capacity = initial_battery_capacity

class SolarMILPModel(ScheduleModel):
    def __init__(self,
                 time_steps: int,
                 dt: float,
                 params: ScheduleModelParameters):

        assert isinstance(params, SolarMilpModelParameters)
        super().__init__(time_steps, dt, params)
        self.params = params

        self.params.p_load = self.params.p_load.reshape(-1, 1)
        self.params.p_solaravail = self.params.p_solaravail.reshape(-1, 1)

        if self.params.p_load.size != self.time_steps:
            raise ValueError("Load must be an array of size time_steps.")
        if self.params.p_solaravail.size != self.time_steps:
            raise ValueError("Solar availability must be an array of size time_steps.")
        if self.params.initial_battery_capacity < self.params.p_bat_bound[0]:
            raise ValueError("The author of this program is a lazy retard and forgot to implement the initial battery capacity check. Just use something slightly higher than the lower bound.")

        if np.max(self.params.p_solaravail) > self.params.p_solar_bound:
            warnings.warn("The maximum value in p_solaravail exceeds p_solar_bound.\n"
                          "Limiting p_solaravail to p_solar_bound.", stacklevel=2)
            self.params.p_solaravail = np.minimum(self.params.p_solaravail, self.params.p_solar_bound)

        # Print initial settings summary with values limited to 3 decimal places
        print("Model input parameters:")
        print(f"  Battery capacity: {params.battery_capacity:.3f} kWh")
        print(f"  Initial battery capacity: {params.initial_battery_capacity:.3f} kWh")
        print(f"  Battery charge efficiency: {100 * params.eff_battery_chg:.3f}%")
        print(f"  Battery discharge efficiency: {100 * params.eff_battery_dis:.3f}%")
        print(f"  Solar efficiency: {100 * params.eff_solar:.3f}%")
        print(f"  Grid power bounds: {params.p_grid_bound[0]:.3f} to {params.p_grid_bound[1]:.3f} kW")
        print(f"  Battery power bounds: {params.p_bat_bound[0]:.3f} to {params.p_bat_bound[1]:.3f} kW")
        print(f"  Max Solar usage: {params.p_solar_bound:.3f} kW")
        print("------------------------------------------------------------")

    @warnings.deprecated("Use _setup_variables_milp instead. Kept for reference.")
    def _setup_variables(self):
        self.params = cast(SolarMilpModelParameters, self.params)
        ts = self.time_steps
        # Criterion Vector (c)
        self.c = np.zeros((5*ts, 1))
        self.c[0:ts] = 1

        # Nonequality Matrix (A_ub)
        self.aub = np.zeros((2*ts, 5*ts))
        # 1st eq
        self.aub[0:ts, 0:ts]    = -np.eye(ts)
        self.aub[0:ts, ts:2*ts] = self.params.grid_price_buy * np.eye(ts)
        # 2nd eq
        self.aub[ts:2*ts, 0:ts]    = -np.eye(ts)
        self.aub[ts:2*ts, ts:2*ts] = self.params.grid_price_sell * np.eye(ts)

        # Nonequality Vector (b_ub)
        self.bub = np.zeros((2 * ts, 1))

        # Equality Matrix (A_eq)
        self.aeq = np.zeros((ts, 5*ts))
        # 1st eq
        self.aeq[0:ts, ts:2*ts]   = np.eye(ts)
        self.aeq[0:ts, 2*ts:3*ts] = self.params.eff_solar * np.eye(ts)
        self.aeq[0:ts, 3*ts:4*ts] = np.eye(ts)
        # 2nd eq
        col_offset = 3*ts
        self.aeq_2 = np.zeros((ts, 5*ts))
        self.aeq_2[0:ts, col_offset:col_offset+ts] = self.dt * np.eye(ts)
        self.aeq_2[0:ts, col_offset+ts:col_offset+2*ts] = np.eye(ts) - np.eye(ts, k=-1)
        self.aeq = np.concatenate((self.aeq, self.aeq_2), axis=0)

        # Equality Vector (b_eq)
        self.beq = np.zeros((2 * ts, 1))
        self.beq[0:ts, 0] = self.params.p_load.flatten()
        self.beq[ts, 0] = self.params.initial_battery_capacity

        # Lower Bound Vector (lb)
        self.lb = 5*ts * [None]
        self.lb[ts:2*ts]   = ts * [self.params.p_grid_bound[0]]
        self.lb[2*ts:3*ts] = ts * [0]
        self.lb[3*ts:4*ts] = ts * [self.params.p_bat_bound[0]]
        self.lb[4*ts:5*ts] = ts * [0.1 * self.params.battery_capacity]

        # Upper Bound Vector (ub)
        self.ub = 5*ts * [None]
        self.ub[ts:2*ts]   = ts * [self.params.p_grid_bound[1]]
        self.ub[2*ts:3*ts] = self.params.p_solaravail.reshape(1, -1).tolist()[0]
        self.ub[3*ts:4*ts] = ts * [self.params.p_bat_bound[1]]
        self.ub[4*ts:5*ts] = ts * [0.9 * self.params.battery_capacity]

    def _setup_variables_milp(self):
        """
        Optimization vector (6*ts):
        x = [P_grid | P_solar | P_bat | P_charg | BC(2-n) | CH(2-n)]
              0:ts    ts:2ts   2ts:3ts  3ts:4ts   4ts:5ts   5ts:6ts
        CH is binary.

        Objective: min dt * dot(prices, P_grid)

        Inequality constraints (A_ub x <= b_ub):
          1/eta_dis P_bat - P_charg              <= 0   // discharge upper bound
          eta_chg   P_bat - P_charg              <= 0   // charge upper bound
          -1/eta_dis P_bat + P_charg - M*CH      <= 0   // big-M discharge mode
          -eta_chg   P_bat + P_charg + M*CH      <= M   // big-M charge mode

        Equality constraints (A_eq x == b_eq):
          P_grid + eta_solar P_solar + P_bat      = P_load  // demand is satisfied
          dt P_charg(i) + BC(i+1) - BC(i)         = 0       // battery charge equation

        Bounds:
          P_grid  in [P_grid^MIN,  P_grid^MAX]
          P_solar in [0,           P_solaravail]
          P_bat   in [P_bat^MIN,   P_bat^MAX]
          P_charg in [-inf,        +inf]
          BC      in [0.1 BC^MAX,  0.9 BC^MAX]
          CH      is binary
        """
        self.params = cast(SolarMilpModelParameters, self.params)
        bigM = 1e7
        ts = self.time_steps

        # Criterion: min dt * dot(prices, P_grid)
        self.c = np.zeros(6*ts)
        self.c[0:ts] = self.params.grid_prices * self.dt

        # Nonequality Matrix (4*ts rows, 6*ts cols)
        self.aub = np.zeros((4*ts, 6*ts))
        # 1st eq: 1/eta_dis P_bat - P_charg <= 0
        self.aub[0:ts, 2*ts:3*ts] = 1/self.params.eff_battery_dis * np.eye(ts)
        self.aub[0:ts, 3*ts:4*ts] = -np.eye(ts)
        # 2nd eq: eta_chg P_bat - P_charg <= 0
        self.aub[ts:2*ts, 2*ts:3*ts] = self.params.eff_battery_chg * np.eye(ts)
        self.aub[ts:2*ts, 3*ts:4*ts] = -np.eye(ts)
        # 3rd eq: -1/eta_dis P_bat + P_charg - M*CH <= 0
        self.aub[2*ts:3*ts, 2*ts:3*ts] = -1/self.params.eff_battery_dis * np.eye(ts)
        self.aub[2*ts:3*ts, 3*ts:4*ts] = np.eye(ts)
        self.aub[2*ts:3*ts, 5*ts:6*ts] = -bigM * np.eye(ts)
        # 4th eq: -eta_chg P_bat + P_charg + M*CH <= M
        self.aub[3*ts:4*ts, 2*ts:3*ts] = -self.params.eff_battery_chg * np.eye(ts)
        self.aub[3*ts:4*ts, 3*ts:4*ts] = np.eye(ts)
        self.aub[3*ts:4*ts, 5*ts:6*ts] = bigM * np.eye(ts)

        # Nonequality Vector
        self.bub = np.zeros(4*ts)
        self.bub[3*ts:4*ts] = bigM

        # Equality Matrix (2*ts rows, 6*ts cols)
        self.aeq = np.zeros((ts, 6*ts))
        # 1st eq: power balance
        self.aeq[0:ts, 0:ts]      = np.eye(ts)
        self.aeq[0:ts, ts:2*ts]   = self.params.eff_solar * np.eye(ts)
        self.aeq[0:ts, 2*ts:3*ts] = np.eye(ts)
        # 2nd eq: SoC update
        self.aeq_2 = np.zeros((ts, 6*ts))
        self.aeq_2[0:ts, 3*ts:4*ts] = self.dt * np.eye(ts)
        self.aeq_2[0:ts, 4*ts:5*ts] = np.eye(ts) - np.eye(ts, k=-1)
        self.aeq = np.concatenate((self.aeq, self.aeq_2), axis=0)

        # Equality Vector
        self.beq = np.zeros(2 * ts)
        self.beq[0:ts] = self.params.p_load.flatten()
        self.beq[ts] = self.params.initial_battery_capacity

        # Lower Bounds Vector
        self.lb = np.ones(6*ts) * -np.inf
        self.lb[0:ts]      = self.params.p_grid_bound[0]
        self.lb[ts:2*ts]   = 0
        self.lb[2*ts:3*ts] = self.params.p_bat_bound[0]
        self.lb[4*ts:5*ts] = 0.1 * self.params.battery_capacity
        self.lb[5*ts:6*ts] = 0  # integer binary

        # Upper Bounds Vector
        self.ub = np.ones(6*ts) * np.inf
        self.ub[0:ts]      = self.params.p_grid_bound[1]
        self.ub[ts:2*ts]   = self.params.p_solaravail.flatten()
        self.ub[2*ts:3*ts] = self.params.p_bat_bound[1]
        self.ub[4*ts:5*ts] = 0.9 * self.params.battery_capacity
        self.ub[5*ts:6*ts] = 1  # integer binary

        self.bounds = scipy.optimize.Bounds(self.lb, self.ub)

        # LinearConstraint expects lb/ub 1D (length = number of rows of A)
        self.eq_constr  = scipy.optimize.LinearConstraint(self.aeq, ub=self.beq, lb=self.beq)
        self.neq_constr = scipy.optimize.LinearConstraint(self.aub, ub=self.bub)

    def solve(self):
        ts = self.time_steps
        print("Running MILP Solver...")
        self._setup_variables_milp()
        integrality = np.zeros(6*ts)
        integrality[5*ts:6*ts] = 1   # integer variables
        t1 = time.time()
        res = scipy.optimize.milp(self.c, bounds=self.bounds, \
                                  constraints=[self.eq_constr, self.neq_constr], \
                                  integrality=integrality)
        print(f"MILP Solver Finished in {1000 * (time.time() - t1):.2f} ms.")
        return res
