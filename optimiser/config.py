import numpy as np
from pathlib import Path
import os

'''
data folder
'''

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


'''
genetic algorithm parameters
'''

config = {
    "num_stations": 40,  # the number of stations
    "generations": 500,  # the number of generations
    "sol_per_pop": 100,   # population size for each generation
    "num_parents_mating": 20,  # number of parents to select for mating
    "parent_selection_type": "tournament",  #  parent select strategy  sss = steady-state selection
    "crossover_type": "uniform",         #"single_point",  # crossover strategy
    "mutation_type": "random",         # mutation strategy
    "crossover_probability": 0.8,      # crossover possibility
    "mutation_probability": 0.12,      # mutation possibility
    "keep_elitism": 2,     # keep the best individuals for the next generation
    "keep_parents": 0,     # keep parents for the next generation
    "stop_criteria": ["saturate_200"],  # stop criteria
    "random_seed": 0,
    "log_dir": "log",     # log Directory
    "n_random_layouts": 1000,    # random baseline experiment number (Compared to GA)
}





n_random_layouts = 1000