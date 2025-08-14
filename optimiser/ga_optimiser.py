from pathlib import Path
import numpy as np
import geopandas as gpd
import pygad
from fiona.features import length
from scipy.spatial import cKDTree
from joblib import load

from optimiser.config import *
from optimiser.data_loader import load_data

import pandas as pd
import time
import matplotlib.pyplot as plt
import utils.osrm_utils as tools
from shapely.geometry import Point


# set global variables
_xy_all = None
_time_matrix = None
_incident_freq = None
_partial_features = None
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


def count_stations_per_buffer(xy_all, solution, buffer_m=10000):
    """
    Calculate the number of stations within a buffer area for each grid cell.

    Parameters:
        xy_all: (N, 2) numpy array, coordinates of all grid centroids (EPSG:27700)
        station_gdf: GeoDataFrame, columns = ['grid', 'geometry'], CRS = EPSG:27700
        buffer_m: Buffer radius in meters

    Returns:
            pd.Series, index = grid, values = station_num
    """

    # Create fire station GeoDataFrame
    station = gpd.GeoDataFrame(
        {'grid': solution},
        geometry=[Point(xy) for xy in xy_all[solution]],
        crs="EPSG:27700"
    )
    station_gdf = station.to_crs(27700)

    # Create grid centroid GeoDataFrame
    cent = gpd.GeoDataFrame(
        {'grid': np.arange(len(xy_all), dtype=int)},
        geometry=[Point(xy) for xy in xy_all],
        crs="EPSG:27700"
    )

    # Create buffer polygons (add 'grid_buf' column for spatial join)
    buffer_for_join = cent.copy()
    buffer_for_join['geometry'] = buffer_for_join.geometry.buffer(buffer_m)
    buffer_for_join = buffer_for_join.rename(columns={'grid': 'grid_buf'})

    # Spatial join (left = buffer polygons, right = station points)
    joined = gpd.sjoin(buffer_for_join, station_gdf, how='left', predicate='contains')

    # Count number of stations within each buffer
    station_num = (joined.groupby('grid_buf')['index_right']
                   .count()
                   .reindex(buffer_for_join['grid_buf'], fill_value=0)
                   .astype(int)
                   .rename('station_num'))

    station_num.index.name = 'grid'
    return station_num


def fitness_function(ga_instance, solution, solution_idx):
    # Select nearest_station_travel_time from the time matrix
    selected_times = _time_matrix[:, solution]
    nearest_times = selected_times.min(axis=1)

    # Calculate the number of the station in the buffer, radius=10000
    station_num = count_stations_per_buffer(_xy_all, solution, buffer_m=10000)

    # Combine features
    feature_names = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']

    X = pd.DataFrame(
        np.column_stack([nearest_times, _partial_features, station_num]),
        columns=feature_names
    )

    # Predict the fire service efficiency
    efficiency = _rf_model.predict(X)

    # Calculate the fitness
    fitness = np.sum(efficiency * _incident_freq) / np.sum(_incident_freq)
    print(fitness)

    return float(fitness)


def run_optimisation(data_dict, gene_space, config, verbose=True, plot=True):
    # Load data
    data = load_data()
    _xy_all = data_dict["xy_all"]
    _incident_freq = data_dict["incident_freq"]
    _time_matrix = data_dict["time_matrix"]
    _partial_features = data_dict["partial_features"]
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