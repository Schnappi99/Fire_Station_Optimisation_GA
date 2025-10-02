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

config_old = {
    "num_stations": 40,  # the number of stations
    "generations": 500,  # the number of generations
    "sol_per_pop": 100,   # population size for each generation
    "num_parents_mating": 20,  # number of parents to select for mating
    "parent_selection_type": "sss",  #  parent select strategy  sss = steady-state selection
    "crossover_type": "uniform",         #"single_point",  # crossover strategy
    "mutation_type": "random",         # mutation strategy
    "crossover_probability": 0.8,      # crossover possibility
    "mutation_probability": 0.20,      # mutation possibility
    "keep_elitism": 2,     # keep the best individuals for the next generation
    "keep_parents": 0,     # keep parents for the next generation
    "stop_criteria": ["saturate_200"],  # stop criteria
    "random_seed": 0,
    "log_dir": "log",     # log Directory
    "n_random_layouts": 1000,    # random baseline experiment number (Compared to GA)
}

config = {
    "generations": 1000,
    "sol_per_pop": 200,
    "num_parents_mating": 50,
    "num_stations": 40,
    "parent_selection_type": "tournament",
    "K_tournament": 3,   # tournament_size
    "crossover_type": "uniform",
    "crossover_probability": 0.8,
    "mutation_probability": 0.2,
    "keep_elitism": 2,
    "keep_parents": 2,
    "stop_criteria": ["saturate_100"],
    "random_seed": 0,
    "log_dir": "log",  # log Directory
    "n_random_layouts": 1000,  # random baseline experiment number (Compared to GA)
    # parameters in init_pop
    "gene_space_top_pct": 0.2,           # top 20%
    "n_single_swap_seeds": 100,          #
    "seed_uniform_mix_ratio": 0.3,       #
    "seed_alpha": 1.0,                   #
    "n_random_layouts": 1000,    # random baseline experiment number (Compared to GA)
}



n_random_layouts = 1000