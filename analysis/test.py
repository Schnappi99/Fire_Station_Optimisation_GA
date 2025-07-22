import numpy as np
import pandas as pd


eff =  pd.read_csv("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layouts.csv")
fitness = eff['fitness']/ np.sum(_incident_freq)