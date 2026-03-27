import numpy as np

power_consumption = np.array([
    0.7, 0.6, 0.5, 0.6, 0.65, 1.2, 1.2, 1.0, 1.2, 1.4, 1.6, 1.8, # 1am - noon
    2.0, 2.2, 3.2, 3.5, 4.0, 4.0, 3.5, 2.5, 2.0, 1.5, 1.0, 0.7  # 1pm - midnight
])

# Representative Solar Summer Generation (kW) for each hour
solar_summer = np.array([
    0.  , 0.  , 0.  , 0.  , 0. ,  0   , 0.25  , 0.35  , 0.5  , 0.5 , 1.0 ,
    0.8 , 3.0 , 4.0  , 3.8  , 3.1  , 2.4 , 2.0 , 1.3  , 0.5  , 0.1  , 0.  , 0., 0.0
])

# Representative Solar Winter Generation (kW) for each hour
solar_winter = 0.3 * solar_summer