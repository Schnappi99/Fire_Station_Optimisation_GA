from pathlib import Path
import numpy as np
import pygad
from scipy.spatial import cKDTree
from joblib import load
from optimiser.config import *
import pandas as pd
import time
import matplotlib.pyplot as plt
import utils.network_tools as tools

# 为 pygad fitness 函数设置全局变量
_xy_all = None
_incident_xy = None
_incident_freq = None
_incident_grid_idx = None
_features = None
_rf_model = None
_total_incidents = None


def on_start(ga):
    global t, log, best_layout, best_fitness
    t = time.time()
    log = []
    best_fitness = 0
    best_layout = []


def on_generation(ga):
    """
      This function is for recording the time cost of each generation and plotting fitness curve.
      """
    # global log
    # print("Generation {}: time cost: {:.1f}; fitness:{:.0f}".format(ga.generations_completed, time.time() - t, ga.best_solutions_fitness[-1]))
    # log.append([time.time() - t, ga.best_solutions_fitness[-1]])

    # The following part is for Steady-State version, which has too many generations.
    global log
    if ga.generations_completed % 10 == 0:
        print("Generation {} | Elapsed: {:.1f}s | Best fitness: {:.0f}".format(
            ga.generations_completed,
            time.time() - t,
            ga.best_solution_fitness))
        log.append([time.time() - t, ga.best_solution_fitness])


def on_stop(ga, last_fit):
    """
    This function is for saving logs when stopped (visual + save as CSV)
    """
    _log = np.array(log, dtype='float64')
    pd.DataFrame(_log, columns=["Time", "Fitness"]).to_csv("log/log.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(_log[:, 0], _log[:, 1], marker='o', color='orange')
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Best Fitness")
    plt.title("GA Fitness Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("log/fitness_curve.png")
    plt.show()


def fitness_function(ga_instance, solution, solution_idx):
    station_xy = _xy_all[solution]
    station_coors = tools._transform_coords(station_xy)
    event_coors = tools._transform_coords(_incident_xy)
    min_time = tools.get_osrm_time(event_coors, station_coors)


    time_df = pd.DataFrame({'grid_idx': _incident_grid_idx, 'driving_time': min_time})
    mean_time_per_grid = time_df.groupby('grid_idx')['driving_time'].mean()

    mean_dist_full = pd.Series(np.nan, index=np.arange(_xy_all.shape[0]), dtype=float)
    mean_dist_full.update(mean_time_per_grid)
    mean_dist_full = mean_dist_full.fillna(0)

    feature_names = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']

    X = pd.DataFrame(
        np.column_stack([mean_dist_full.values.reshape(-1, 1), _features]),
        columns=feature_names
    )

    efficiency = _rf_model.predict(X)
    # fitness = np.sum(efficiency * _incident_freq)
    fitness = np.sum(efficiency * _incident_freq) / np.sum(_incident_freq)
    print(fitness)
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
        on_start=on_start,
        on_generation=on_generation,
        on_stop=on_stop,
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