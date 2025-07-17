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
    "num_stations": 40,  # 要选址的消防站数量
    "generations": 500,  # 遗传算法的最大迭代次数
    "sol_per_pop": 50,   # 每一代的个体数（population size）
    "num_parents_mating": 20,  # 每代中用于交叉的父代个体数
    "parent_selection_type": "sss",  # 父代选择方式：sss = steady-state selection（稳定选择）
    "crossover_type": "single_point",  # 单点交叉
    "mutation_type": "random",         # 随机突变
    "crossover_probability": 0.9,      # 交叉发生的概率
    "mutation_probability": 0.08,      # 突变发生的概率
    "keep_elitism": 2,     # 每代保留表现最好的个体数量
    "keep_parents": 4,     # 每代保留用于下一代的父母个体数量
    "stop_criteria": ["saturate_50"],  # 停止标准：适应度50代未提升则终止
    "random_seed": 0
}

# ---------- random  baseline ----------
n_random_layouts = 1000