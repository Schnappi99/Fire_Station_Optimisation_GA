from optimiser.data_loader import load_data
from optimiser.ga_optimiser import run_optimisation
from optimiser.config import config
import numpy as np

if __name__ == "__main__":
    data = load_data()
    gene_space = np.arange(len(data["xy_all"]))

    best_solution, best_fitness, best_eff_pct = run_optimisation(
        data_dict=data,
        gene_space=gene_space,
        config=config,
        verbose=True
    )

    print("\n? ??????:", best_solution)
    print("? ??????:", best_fitness)
    print("? ??????:", f"{best_eff_pct:.2%}")