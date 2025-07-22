import numpy as np
from pathlib import Path
import os

'''
data folder
'''

DATA_DIR = Path(__file__).parent.parent / "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


'''
genetic algorithm parameters
'''

config = {
    "num_stations": 40,  # the number of stations
    "generations": 500,  # the number of generations
    "sol_per_pop": 50,   # population size for each generation
    "num_parents_mating": 20,  # number of parents to select for mating
    "parent_selection_type": "sss",  #  parent select strategy  sss = steady-state selection
    "crossover_type": "single_point",  # crossover strategy
    "mutation_type": "random",         # mutation strategy
    "crossover_probability": 0.9,      # crossover possibility
    "mutation_probability": 0.08,      # mutation possibility
    "keep_elitism": 2,     # keep the best individuals for the next generation
    "keep_parents": 4,     # keep parents for the next generation
    "stop_criteria": ["saturate_50"],  # stop criteria 
    "random_seed": 0
}

# ---------- random  baseline ----------
n_random_layouts = 1000