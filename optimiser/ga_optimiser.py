from pathlib import Path
import numpy as np
import pygad
from scipy.spatial import cKDTree
from joblib import load
from config import *
import pandas as pd


# 为 pygad fitness 函数设置全局变量
_xy_all = None
_incident_xy = None
_incident_freq = None
_incident_grid_idx = None
_features = None
_rf_model = None
_total_incidents = None

def fitness_function(ga_instance, solution, solution_idx):
    station_xy = _xy_all[solution]
    distances, _ = cKDTree(station_xy).query(_incident_xy, k=1)

    dist_df = pd.DataFrame({'grid_idx': _incident_grid_idx, 'distance': distances})
    mean_dist_per_grid = dist_df.groupby('grid_idx')['distance'].mean()

    mean_dist_full = pd.Series(np.nan, index=np.arange(_xy_all.shape[0]), dtype=float)
    mean_dist_full.update(mean_dist_per_grid)
    mean_dist_full = mean_dist_full.fillna(0)

    feature_names = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']

    X = pd.DataFrame(
        np.column_stack([mean_dist_full.values.reshape(-1, 1), _features]),
        columns=feature_names
    )

    efficiency = _rf_model.predict(X)
    fitness = np.sum(efficiency * _incident_freq)
    return float(fitness)


def run_optimisation(data_dict, gene_space, config, verbose=True, plot=True):
    global _xy_all, _incident_xy, _incident_freq, _incident_grid_idx, _features, _rf_model, _total_incidents
    _xy_all = data_dict["xy_all"]
    _incident_xy = data_dict["incident_xy"]
    _incident_freq = data_dict["incident_freq"]
    _incident_grid_idx = data_dict["incident_grid_idx"]
    _features = data_dict["features"]
    _rf_model = data_dict["rf_model"]
    _total_incidents = data_dict["total_incidents"]

    n_station = config["num_stations"]

    if verbose:
        print(f"Optimising with {n_station} stations from {len(gene_space)} feasible locations...")

    ga = pygad.GA(
        num_generations=config["generations"],
        sol_per_pop=config["sol_per_pop"],
        num_parents_mating=config["num_parents_mating"],
        num_genes=n_station,
        gene_type=int,
        gene_space=gene_space,
        fitness_func=fitness_function,
        parent_selection_type=config["parent_selection_type"],
        crossover_type=config["crossover_type"],
        crossover_probability=config["crossover_probability"],
        mutation_type=config["mutation_type"],
        mutation_probability=config["mutation_probability"],
        keep_elitism=config["keep_elitism"],
        keep_parents=config["keep_parents"],
        stop_criteria=config["stop_criteria"],
        random_seed=config["random_seed"],
        allow_duplicate_genes=False
    )

    ga.run()

    best_solution = ga.best_solutions[ga.best_solution_generation]
    best_fitness = ga.best_solutions_fitness[-1]
    best_eff_pct = best_fitness / _total_incidents

    if plot:
        try:
            ga.plot_fitness(title="Fitness Over Generations")
        except Exception as e:
            print(" Failed to plot fitness:", e)

    if verbose:
        print("--------------------------------------------------")
        print("GA finished running!")
        print(f"Best fitness: {best_fitness:,.0f} / {int(_total_incidents)} incidents")
        print(f"Efficiency: {best_eff_pct:.2%}")
        print(f"Best layout indices: {best_solution.tolist()}")

    return best_solution, float(best_fitness), float(best_eff_pct)