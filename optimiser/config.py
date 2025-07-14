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
grid parameters
'''
rows, cols = 107, 71      # depends on xy_all.npy
cell_width = 500            # metre


'''
genetic algorithm parameters
'''

num_generations     = 500
sol_per_pop         = 50
num_parents_mating  = 20

parent_selection_type = "sss"     
crossover_type        = "single_point"
mutation_type         = "random"
crossover_probability = 0.9
mutation_probability  = 0.08
keep_elitism          = 2
keep_parents          = 4          
stop_criteria         = ["saturate_50"]  
random_seed           = 0

# ---------- random  baseline ----------
n_random_layouts = 1000