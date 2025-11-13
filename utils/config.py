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
    "generations": 800,               # Maximum generations
    "sol_per_pop": 200,               # Number of individuals per population
    "num_parents_mating": 50,         # Number of parents selected for mating in each generation
    "num_stations": 40,               # Number of stations (genes) per solution

    #  Selection & Crossover settings
    "parent_selection_type": "tournament",   # Parent selection strategy ("tournament" / "rank" / "roulette")
    "K_tournament": 3,                       # Tournament size (number of individuals competing)
    "crossover_type": "uniform",             # Crossover method ("single_point", "two_points", "uniform", "scattered")
    "crossover_probability": 0.8,            # Probability that crossover occurs between two parents

    #  Mutation settings
    "mutation_probability": 0.2,             # Probability that a gene mutates
    "keep_elitism": 2,                       # Number of best solutions preserved each generation (elitism)
    "keep_parents": 2,                       # Number of parent solutions carried over to next generation
    "random_seed": 0,                        # Random seed for reproducibility

    # Logging and baseline experiment
    "log_dir": "log",                        # Directory for saving run logs
    "n_random_layouts": 1000,                # Number of random layouts for baseline comparison (outside GA)

    # gene_space_top_pct defines the proportion of high-demand cells used when sampling initial solutions
    "gene_space_top_pct": [0.40, 0.30, 0.20, 0.10],   # Top X% of demand cells considered (used in sweep runs)

    #  Initial population parameters
    "method_mode": ["balanced_init", "local_init", "random_init"],                # Initialisation mode: "mixed", "single_swap", or "random"
    "n_single_swap_seeds": 100,  # Number of single-swap neighbours generated from baseline
    "seed_uniform_mix_ratio": 0.3,  # Ratio of purely random seeds mixed with demand-weighted ones
    "seed_alpha": 1.0,                                                # Demand-weighting factor (1.0 = full demand-based sampling)

    #  Stopping criteria
    "stop_criteria": ["saturate_300", "saturate_500"],  # Stop when no improvement for 300/500 generations

}

