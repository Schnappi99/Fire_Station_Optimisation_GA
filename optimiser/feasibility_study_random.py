import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import percentileofscore

from optimiser import ga_optimiser
from optimiser.data_loader import load_data

from config import DATA_DIR
from config import n_random_layouts
from config import config

from tqdm import tqdm


# Set global variables
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

def evaluate_random_layouts(n_samples: int, n_station: int, feasible_cells: np.ndarray) -> pd.DataFrame:
    """
    Randomly sample several layouts, calculate their fitness, and compare them with optimization results.
    """
    results = []

    print(f"Evaluating {n_samples} random layouts with {n_station} stations...")

    for _ in tqdm(range(n_samples)):
        random_layout = np.random.choice(feasible_cells, size=n_station, replace=False)
        fitness = fitness_function(None, random_layout, 0)  # Ignore GA parameters
        results.append((random_layout, fitness))

    # Save as DataFrame
    df = pd.DataFrame(results, columns=["layout", "fitness"])
    return df


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
    baseline_fitness = fitness_function(None, current_layout_idx,0)
    print(baseline_fitness)

    # set parameters: number of station is import from config; n_random_layouts is import from config
    n_station = config["num_stations"]
    feasible_cells = np.arange(_xy_all.shape[0])
    # Run a random layout evaluation
    df_random = evaluate_random_layouts(n_random_layouts, n_station, feasible_cells)

    # Save results to CSV
    out_path = DATA_DIR.parents[0] / "analysis" / "random_layouts_with_efficiency.csv"
    df_random.to_csv(out_path, index=False)
    print(f"Saved random layout results to: {out_path}")
    print("")

    # Plot the fitness distribution of random layouts   # current layout fitness
    plt.figure(figsize=(10, 6))
    plt.hist(df_random["fitness"], bins=30, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(baseline_fitness, color='red', linestyle='--', linewidth=2, label='Current layout efficiency')
    plt.title("Efficiency Distribution of 1000 Random Layouts")
    plt.xlabel("Efficiency")
    plt.ylabel("Number of Studies")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calculate the percentile of the current layout fitness
    percentile = percentileofscore(df_random["fitness"], baseline_fitness)
    print(f"Current layout fitness: {baseline_fitness}")
    print(f"Percentile in random layouts: {percentile:.2f}%")

