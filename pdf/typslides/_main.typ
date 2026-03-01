#import "@preview/typslides:1.3.2": *
#import "@preview/equate:0.1.0": equate

// Project configuration
#show: typslides.with(
  ratio: "4-3",
  theme: "bluey",
  font: "New Computer Modern",
  font-size: 13pt,
  link-style: "color",
  show-progress: true,
)

#set math.equation(numbering: "(1)")
#show: equate.with(number-mode: "line")

// The front slide is the first slide of your presentation
//#front-slide(
//  title: "ECE 57000 Progress Check 1",
//  subtitle: [Solar Energy Forecasting Using a Neural Network and Energy Scheduling
//             Using Linear Programming],
//  authors: "Stepan Pressl",
//)

// Custom outline
// We dont need this cause im not presenting this.
//#table-of-contents()

//// Title slides create new sections
//#title-slide[
  //This is a _Title slide_
//]

//// A simple slide
//#slide[
  //- This is a simple `slide` with no title.
  //- #stress("Bold and coloured") text by using `#stress(text)`.
  //- Sample link: #link("typst.app").
    //- Link styling using `link-style`: `"color"`, `"underline"`, `"both"`
  //- Font selection using `font: "Fira Sans"`, `size: 21pt`.

  //#framed[This text has been written using `#framed(text)`. The background color of the box is customisable.]

  //#framed(title: "Frame with title")[This text has been written using `#framed(title:"Frame with title")[text]`.]
//]

//// Focus slide
//#focus-slide[
  //This is an auto-resized _focus slide_.
//]

//// Blank slide
//#blank-slide[
  //- This is a `#blank-slide`.

  //- Available #stress[themes]#footnote[Use them as *color* functions! e.g., `#reddy("your text")`]:

  //#framed(back-color: white)[
    //#bluey("bluey"), #reddy("reddy"), #greeny("greeny"), #yelly("yelly"), #purply("purply"), #dusky("dusky"), darky.
  //]

  //```typst
  //#show: typslides.with(
    //ratio: "16-9",
    //theme: "bluey",
    //...
  //)
  //```

  //- Or just use *your own theme color*:
    //- `theme: rgb("30500B")`
//]

//#slide(title: "ECE 57000 Project Timeline", outlined: true)[
//  - Progress Check 1 (see next slides): \
//    1. Problem Formulation and Solution Approach.
//    2. Household scheduling using a linear programming model.
//  - Progress Check 2 (TBD): \
//    1. Solar Power Data collection and scraping.
//    2. Neural Network architecture and initial training.
//  - Final Report (TBD): \
//    - Overall integration of the model with the scheduling system.
//]

// Slide with title
#slide(title: "Problem Statement & Goal", outlined: true)[
  - Chosen track is 2. The name of the project is Solar Energy Forecasting Using a Neural Network and Energy Scheduling Using Linear Programming
  - #bluey[*DISCLAIMER:*] The project is not about the MPPT algorithm, which is utilized in all solar inverters.
  - Suppose we have a household with a solar plant and an electricity storage (battery).
  - The goal of the application is following: the user specifies the 
    expected energy consumption at given times for the following time intervals (e.g. two days). Our product will forecast
    solar power in the following days using a recurrent neural network,
    based on current weather forecast, historical solar data,
    and current date.
    Afterwards, the application will schedule electricity grid
    consumption and battery charging plans which minimize the
    electricity bill the user has to pay. Also, the application will find all time slots with unused solar energy. In this problem,
    we will suppose flat electricity prices, so I only train
    one neural network for the solar output forecast (also majority of households use flat
    prices). We generally assume the electricity produced
    by the solar plant can be sold to the distributor for
    a flat price as well.
  - Let's mention several heuristics to manage solar energy (already implemented in many smart homes):
    - Time-of-use charging: charge the battery during off-peak hours and discharge it during peak
                   hours when the grid electricity price is high.
    - Priority-based consumption: prioritze critical loads, and if there's energy
                                  surplus, use it to charge the battery or any other
                                  load (such as water heater).
    - As mentioned in @sigenergy, solar inverters
      with AI already exist. However, we aim for a more
      modular, open and custom solution.
]


#slide(title: "Problem Statement & Goal + Methodology Overview", outlined: true)[
  - The problem with aforementioned strategies - not generalizable. For example,
    electricity prices may even be negative (in case of energy surplus). The
    solution is to develop a linear programming (LP) model and
    then find the optimal schedule at respective $i$ time steps.
  - Other approaches to the problem are:
    - Mixed-integer linear programming (MILP): necessary for complex binary "turn on/off" or integer constraints, in this project
                  we assume everything is continuous. Definitely usable in future work;
    - Pure Reinforcement learning using NN: much faster to solve the prediction problem, however the problem does not require frequent energy schedules. Every 15 minutes is enough. Also does not guarantee optimal solution. In case of a different model (more powerful solar panels, higher capacity battery), the model needs to be retrained (in former case, we modify the constants only);
    - Genetic algorithms: does not guarantee optimal solution,
      but will probably be much faster
      than MILP if the problem has too many variables. Also does not require retraining.
  - #bluey[*Remaining Issues for Deployment:*] Solar energy production and electricity prices forecasting. This will be the topic of Progress Check 2 and final report with a recurrent neural network.
  - For Progress Check 1, we will focus on the LP model (our sandbox for overall testing).
  - Project will be implemented in Python. For visualization (scheduled energy production and forecasted solar power production), I plan to program a webpage.
]

#slide(title: "LP Modelling", outlined: true)[
  We need to formulate the problem with the following variables (suppose timesteps $i in {1, ..., N}$ sampled
   in the "zero-order hold" manner, length of the time step is $T$ [h]). Also suppose "sanity-check"
   bounds (e.g. we cannot pull infinite amount of energy from a battery):
  - $P_upright("grid")^((i)) in [P_upright("grid")^("MIN"), P_upright("grid")^("MAX")]$ [$upright("kW")$]: grid electricity consumption at time step $i$, negative in case the household sells energy to the grid, values are all _optimization variables_.
  - $P_upright("solaravail")^((i)) in RR^(+)_0$ [$upright("kW")$]: solar power production at time step $i$, values are all _given constants_ (in the final application predicted from the NN).
  - $P_upright("solar")^((i)) in [0, P_upright("solaravail")^((i))]$ [$upright("kW")$]:
    used solar power in the household (converted by the solar inverter) at time step $i$, and the
    values are all _optimization variables_.
    We must also take into account the maximum power the solar
    inverter can produce. However, to reduce the number of
    variables, we can clip predicted NN data.
  - $P_upright("bat")^((i)) in [P_upright("bat")^("MIN"), P_upright("bat")^("MAX")]$ [$upright("kW")$]: supplied power to the battery at time step $i$, values are all _optimization variables_. Positive value at time $i$ means the battery discharges to supply electricity.
  - $B C^((i)) in [0.1 dot B C^("MAX"), 0.9 dot B C^("MAX")]$:
    helper _optimization variables_ which denote current battery
    charge, the charge is bounded in such a way
    the battery is not fully discharged neither charged,
    as we want to mitigate battery degradation.
  - $P_upright("load")^((i)) in RR^(+)_0$ [$upright("kW")$]: specified household
    electricity demand at time step $i$, value are all _given constants_.
  - $eta_upright("solar")$: efficiency of DC solar power
    conversion to the AC household grid. This is a _constant_.
  - $P B$ [$"$"/upright("kWh")$]: the price for which we buy 1 kWh of electricity. This is a _constant_.
  - $P S$ [$"$"/upright("kWh")$]: the price for which we sell 1 kWh of electricity. This is a _constant_.
  The problem assumes the battery can be charged either from
  the grid or the solar panel inverter. We also assume $P S < P B$, since usually the household sells for less than it buys. This makes the following function convex (needed
  for the LP solvers):
  $
  E C^((i)) = max{ P B dot T dot P_upright("grid")^((i))
                   (upright("for") thick P_upright("grid")^((i)) >= 0), thick
                   P S dot T dot P_upright("grid")^((i))
                   (upright("for") thick P_upright("grid")^((i)) < 0)}
  $
  Let $E C^((i))$ be the energy cost at time step $i$
  (negative when electricity is sold, else otherwise).
]



#slide(title: "LP Modelling Cont. & Finished", outlined: true)[
  Finally, the optimization criterion and constraints:
  $
    min sum^(N)_(i=1) E C^((i))
  $ 
  $
    P_upright("load")^((i)) &= P_"grid"^((i)) +
      eta_"solar" dot P_"solar"^((i)) +
                      P_"bat"^((i))      && "Demand is satisfied." \
    P B dot P_"grid"^((i)) &<= E C^((i)), quad
    P S dot P_"grid"^((i)) <= E C^((i))  && "Energy cost is a max of two functions." \
    B C ^((i+1)) &= B C^((i)) - P_"bat"^((i)) dot T
                                         && "Battery charge state. We compute" B C^((n+1)) "as well." \
    P_"grid"^("MIN") &<= P_"grid"^((i)) <= P_"grid"^("MAX"), quad
    0 <= P_"solar"^((i)) <= P_"solaravail"^((i))
                                         && "Sanity checks." \
    P_"bat"^"MIN" &<= P_"bat"^((i)) <= P_"bat"^("MAX"), quad
    0.1 dot B C^"MAX" <= B C^((i)) <= 0.9 dot B C^"MAX"
                                         && "Sanity checks." \
    B C^((1)) &= B C^("INIT") && "Initial battery charge."
  $
  #cols(columns: (3.5fr, 1.5fr), gutter: 1em)[
    #figure(
      image("./figures/SolarSchematic.png", width: 110%)
    )
  ][
    Schematic of the modelled system. The unidirectional
    energy flow is represented by a diode. A typical setup
    with an AC coupled battery is considered.
    Setups with a DC coupled battery connected directly
    to the inverter
    are commonly used as well. Charging from the AC grid
    is possible, as the inverter converts AC back to
    charge the DC coupled battery.
  ]
]

#slide(title: "Implementation in Python", outlined: true)[
  We can use the `scipy.optimize.linprog` function to solve the LP model
  (a wrapper of the HiGHS solver). The documentation states the problem needs to be formualted as:
  $
  min &c^T x \
  "s.t.": A_"ub" x &<= b_"ub" \
  A_"eq" x &= b_"eq" \
  l b <= x &<= u b \
  "The optimization vector is formulated as: " x^T &=  mat(E C^T, P_"grid"^T, P_"solar"^T, P_"bat"^T, B C^T)
  $
  The code below implements all the needed matrices and vectors.
  ```python
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
  ```
]

#slide(title: "Python LP Code", outlined: true)[
  ```python
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

  # ================= The solution ===================
  bounds = list(zip(self.lb, self.ub))
  result = scipy.optimize.linprog(self.c, A_ub=self.aub, b_ub=self.bub, A_eq=self.aeq, b_eq=self.beq, \
                                  bounds=bounds, method='highs')
  # The resulting vector is stored in result.x and the success is stored in result.success.
  ```
]

#slide(title: "Results", outlined: true)[
  I have downloaded daily power consumption data from @tou-charging and
  manually created a `numpy` array with size 24 (one for each hour).
  #figure(
    image("./figures/LOAD.png", width: 80%),
    caption: "Household power consumption for 24h. It can be seen the consumption peaks in the morning and evening."
  )
]

#slide(title: "Results Cont.", outlined: true)[
  I have also downloaded PV production curves from @pv-production-curves.
  The summer day curve is shown in the figures below, as well as the results
  of the solver for two cases. The cases are described in the captions.
  The initial battery charge is 2 kWh. The battery capacity is 10 kWh,
  but we won't allow it to discharge below 1 kWh neither charge above 9 kWh.
  The solar inverter efficiency is 0.95. It must be mentioned that
  the two datasets are taken from different sources and are used for
  demonstration purposes only. Time step is 1 hour.

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    figure(
      image("./figures/FIG1.png", width: 105%),
      caption: [Buying price is 0.20 USD/kWh, selling price is 0.05 USD/kWh.
                We can see the battery is charged in the afternoon,
                so it can be discharged in the evening.]
    ),
    figure(
      image("./figures/FIG2.png", width: 105%),
      caption: [Buying and selling prices are both 0.20 USD/kWh.
                We can see some energy is sold to the grid in the afternoon,
                when there's energy surplus.]
    )
  )
]

#let bib = bibliography("bibliography.bib")
#bibliography-slide(bib)
