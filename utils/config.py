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
    "generations": 800,
    "sol_per_pop": 200,  # 200
    "num_parents_mating": 50,
    "num_stations": 40,
    "parent_selection_type": "tournament",
    "K_tournament": 3,   # tournament_size
    "crossover_type": "uniform",       # "single_point" "two_points" "uniform" "scattered"
    "crossover_probability": 0.8,
    "mutation_probability": 0.2,    # 0.2
    "keep_elitism": 2,
    "keep_parents": 2,
    "random_seed": 0,
    "log_dir": "log",  # log Directory
    "n_random_layouts": 1000,  # random baseline experiment number (Compared to GA)
    # parameters in init_pop
    # "gene_space_top_pct": "30%",
    "gene_space_top_pct": [0.40, 0.30, 0.20, 0.10],
    "stop_criteria": ["saturate_300", "saturate_500"],
    # "method_mode": ["mixed", "single_swap", "random"],
    "method_mode": ["mixed"],
    "min_station_spacing": 3000,
    "n_single_swap_seeds": 100,          #
    "seed_uniform_mix_ratio": 0.3,       #
    "seed_alpha": 1.0,                   #
}

