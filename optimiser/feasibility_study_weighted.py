import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import percentileofscore

from GA_algorithm import ga_optimiser
from GA_algorithm.data_loader import load_data

from config import DATA_DIR
from config import n_random_layouts
from config import config

from tqdm import tqdm
from typing import Optional
from scipy.spatial import cKDTree



# set global variables
_xy_all = None
_time_matrix = None
_incident_freq = None
_partial_features = None
_rf_model = None
_total_incidents = None


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

def demand_weighted_sample(
    candidate_cells: np.ndarray,
    incident_frequencies: np.ndarray,
    n_stations: int,
    epsilon: float = 1e-6,
    alpha: float = 1.0,
    uniform_mix_ratio: float = 0.0,
    xy_all: Optional[np.ndarray] = None,
    min_dist: Optional[float] = None         # Minimum spacing in meters; None means no constraint
) -> np.ndarray:
    """
    Randomly sample station locations from a set of feasible candidates,
    with probabilities weighted by historical incident frequencies.
    Optionally enforces a minimum distance between selected stations.
    """

    #  Original logic (without spacing constraint)
    if min_dist is None:
        freq_subset = incident_frequencies[candidate_cells].astype(float)
        weights = np.maximum(freq_subset, epsilon) ** alpha

        if uniform_mix_ratio > 0.0:
            lam = 1.0 - float(uniform_mix_ratio)
            uniform_weights = np.ones_like(weights, dtype=float)
            weights = lam * weights + (1.0 - lam) * uniform_weights

        probs = weights / weights.sum()
        chosen_cells = np.random.choice(candidate_cells, size=n_stations, replace=False, p=probs)
        return np.sort(chosen_cells)

    # Logic with minimum spacing constraint
    if xy_all is None:
        raise ValueError("xy_all must be provided when min_dist is set.")

    remaining = candidate_cells.copy()
    chosen = []

    while len(chosen) < n_stations and len(remaining) > 0:
        # Compute weights
        freq_subset = incident_frequencies[remaining].astype(float)
        weights = np.maximum(freq_subset, epsilon) ** alpha

        if uniform_mix_ratio > 0.0:
            lam = 1.0 - float(uniform_mix_ratio)
            uniform_weights = np.ones_like(weights, dtype=float)
            weights = lam * weights + (1.0 - lam) * uniform_weights

        probs = weights / weights.sum()

        # Sample one location
        chosen_idx_in_remaining = np.random.choice(len(remaining), p=probs)
        chosen_cell = remaining[chosen_idx_in_remaining]
        chosen.append(chosen_cell)

        # Remove all candidates within min_dist of the chosen point
        chosen_coord = xy_all[chosen_cell]
        coords_remaining = xy_all[remaining]
        dists = np.linalg.norm(coords_remaining - chosen_coord, axis=1)

        mask_keep = dists >= min_dist
        mask_keep[chosen_idx_in_remaining] = False  #  Also remove the chosen point itself
        remaining = remaining[mask_keep]

    # If not enough stations selected, randomly fill the remainder
    if len(chosen) < n_stations:
        missing = n_stations - len(chosen)
        extras = np.random.choice(candidate_cells, size=missing, replace=False)
        chosen.extend(extras)

    return np.sort(np.array(chosen))

def evaluate_demand_weighted_layouts(
    n_layouts: int,                # Number of layouts to sample
    n_stations: int,               # Number of stations per layout
    candidate_cells: np.ndarray,   # Feasible candidate locations (global indices)
    incident_freq: np.ndarray,     # Historical incident frequencies (length = N total grid cells)
    epsilon: float = 1e-6,         # Minimum weight
    alpha: float = 1.0,            # Weight scaling factor
    min_dist: Optional[float] = None,                  # Minimum spacing in meters
) -> pd.DataFrame:
    """
    Generate multiple demand-weighted random layouts and evaluate their fitness.

    Returns:  DataFrame with columns:
        - "layout": list of selected station cell indices
        - "fitness": computed fitness value for each layout
    """
    results = []
    print(f"Evaluating {n_layouts} demand-weighted random layouts with {n_stations} stations...")

    for _ in tqdm(range(n_layouts)):
        # Sample according to demand weights
        layout_indices = demand_weighted_sample(candidate_cells=feasible_cells,
                                                incident_frequencies=incident_freq,
                                                n_stations=n_stations,
                                                epsilon=1e-6, alpha=1.0,
                                                uniform_mix_ratio=0.0,
                                                xy_all=_xy_all,
                                                min_dist=3500)  # Apply minimum spacing constraint

        # Compute fitness
        fitness_val =  fitness_function(None, layout_indices, 0)
        results.append((layout_indices, fitness_val))

    return pd.DataFrame(results, columns=["layout", "fitness"])


if __name__ == "__main__":

    # Load data
    data = load_data()
    _xy_all = data["xy_all"]
    _time_matrix = data["time_matrix"]
    _incident_freq = data["incident_freq"]
    _partial_features = data["partial_features"]
    _rf_model = data["rf_model"]
    _total_incidents = data["total_incidents"]

    # The baseline efficiency
    current_layout_idx = np.load(DATA_DIR / "current_layout_idx.npy")
    baseline_fitness = fitness_function(None, current_layout_idx, 0)
    print(baseline_fitness)

    # set parameters: number of station is import from config; n_random_layouts is import from config
    n_station = config["num_stations"]
    # find out the cells with incident_freq > 0
    feasible_idx = np.where(_incident_freq > 0)[0]
    feasible_cells = feasible_idx

    # Sampling 1000 demand weighted random Layout
    df_demand_weighted = evaluate_demand_weighted_layouts(
        n_layouts=1000,
        n_stations=n_station,
        candidate_cells=feasible_cells,
        incident_freq=_incident_freq,
        epsilon=1e-6,
        alpha=1.0,
        min_dist=3000)

    print(df_demand_weighted.head())
    # Average fitness
    print("Average fitness:", df_demand_weighted["fitness"].mean())

    # Save results to CSV
    out_path = DATA_DIR.parents[0] / "analysis" / "demand_weighted_layouts_3.csv"
    df_demand_weighted.to_csv(out_path, index=False)
    print(f"Saved random layout results to: {out_path}")

    # Plot the fitness distribution of random layouts   # current layout fitness
    plt.figure(figsize=(10, 6))
    plt.hist(df_demand_weighted["fitness"], bins=30, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(baseline_fitness, color='red', linestyle='--', linewidth=2, label='Current layout efficiency')
    plt.title("Efficiency Distribution of 1000 Random Layouts")
    plt.xlabel("Efficiency")
    plt.ylabel("Number of Studies")
    plt.xlim(left=0)  # ensure x-axis starts from 0
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # calculate the percentile of the current layout fitness
    percentile = percentileofscore(df_demand_weighted["fitness"], baseline_fitness)
    print(f"Current layout fitness: {baseline_fitness}")
    print(f"Percentile in random layouts: {percentile:.2f}%")

    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_fitness_histogram.png", dpi=300)