from pathlib import Path
import numpy as np
import pygad
from scipy.spatial import cKDTree
from joblib import load
from config import *
import pandas as pd


def load_data():

    global xy_all, incident_xy, incident_freq, features, rf_model, TOTAL_INCIDENTS, incident_grid_idx

    xy_all = np.load(DATA_DIR / "xy_all.npy")
    incident_xy = np.load(DATA_DIR / "incident_xy.npy")
    incident_freq = np.load(DATA_DIR / "incident_freq.npy")
    features = np.load(DATA_DIR / "Five_features.npy", allow_pickle=True)

    rf_model = load(DATA_DIR / "rf_model.joblib")

    incident_grid_idx = np.load(DATA_DIR / "incident_grid_idx.npy")

    TOTAL_INCIDENTS = np.sum(incident_freq)



def fitness_function(ga_instance, solution, solution_idx):

    """

    Calculate the fitness of a given solution (fire station layout).
    """
    # current solution is a list of indices representing fire station locations
    station_xy = xy_all[solution]  # shape (n_station, 2)

    # calculate the nearest station travel time (change to the OSRM API in the future)
    distances, _ = cKDTree(station_xy).query(incident_xy, k=1)  # shape (n_incidents,)

    # calculate the mean distance to the nearest station for each grid cell
    dist_df = pd.DataFrame({'grid_idx': incident_grid_idx, 'distance': distances})
    mean_dist_per_grid = dist_df.groupby('grid_idx')['distance'].mean()

    # mean_dist_per_grid is a Series indexed by grid_idx, with mean distances
    mean_dist_full = pd.Series(np.nan, index=np.arange(xy_all.shape[0]), dtype=float)
    mean_dist_full.update(mean_dist_per_grid)
    mean_dist_full = mean_dist_full.fillna(0)  

    feature_names = ['nearest_station_travel_time','neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland','station_count']

    X = pd.DataFrame(
        np.column_stack([mean_dist_full.values.reshape(-1, 1), features]),
        columns=feature_names
    )

    # predict the fire service efficiency
    efficiency = rf_model.predict(X)  # (N,)

    # calculate the fitness score
    fitness = np.sum(efficiency * incident_freq)  # incident_freq shape = (N,)

    return float(fitness)


def optimisation(n_station: int,
                 rows: int,
                 cols: int,
                 gene_space: np.ndarray,
                 verbose: bool = True):
    """
    n_station : number of fire stations to place
    rows/cols : grid dimensions (rows x cols)
    gene_space: numpy array of feasible indices for each station
    """
    if verbose:
        print(f"Rows: {rows}  Cols: {cols}  Stations: {n_station}")
        print(f"Feasible cells: {len(gene_space)}")

    ga = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        num_genes=n_station,
        gene_type=int,
        gene_space=gene_space,
        fitness_func=fitness_function,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        crossover_probability=crossover_probability,
        mutation_type=mutation_type,
        mutation_probability=mutation_probability,
        keep_elitism=keep_elitism,
        stop_criteria=stop_criteria,
        random_seed=random_seed,
        allow_duplicate_genes=False
    )

    ga.run()

    best_solution   = ga.best_solutions[ga.best_solution_generation]
    best_fitness    = ga.best_solutions_fitness[-1]
    best_eff_pct    = best_fitness / TOTAL_INCIDENTS   

    if verbose:
        print("--------------------------------------------------")
        print("GA finished running!")
        print(f"Best fitness  = {best_fitness:,.0f}  of total incidents")
        print(f"Overall efficiency = {best_eff_pct:.3%}")
        print(f"Best layout indices: {best_solution.tolist()}")

    return best_solution, float(best_fitness), float(best_eff_pct)


if __name__ == "__main__":
    
    load_data()
    n_station = 40  
    gene_space = np.arange(rows * cols)  

    # Optimisation
    best_solution, best_fitness, best_eff_pct = optimisation(
    n_station=n_station,
    rows=rows,
    cols=cols,
    gene_space=gene_space,
    verbose=True
)

    # Print results
    print("Best layout (indices):", best_solution)
    print("Best fitness:", best_fitness)
    print("Overall efficiency:", f"{best_eff_pct:.2%}")