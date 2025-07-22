import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from optimiser import ga_optimiser
from optimiser.data_loader import load_data




def evaluate_random_layouts(n_samples: int, n_station: int, feasible_cells: np.ndarray) -> pd.DataFrame:
    """
    Randomly sample several layouts, calculate their fitness, and compare them with optimization results.
    """
    results = []

    print(f"Evaluating {n_samples} random layouts with {n_station} stations...")

    for _ in tqdm(range(n_samples)):
        random_layout = np.random.choice(feasible_cells, size=n_station, replace=False)
        fitness = ga_optimiser.fitness_function(None, random_layout, 0)  # Ignore GA parameters
        results.append((random_layout, fitness))

    # Save as DataFrame
    df = pd.DataFrame(results, columns=["layout", "fitness"])
    return df


if __name__ == "__main__":

    # load data
    data = load_data()
    ga_optimiser._xy_all = data["xy_all"]
    ga_optimiser._incident_xy = data["incident_xy"]
    ga_optimiser._incident_freq = data["incident_freq"]
    ga_optimiser._incident_grid_idx = data["incident_grid_idx"]
    ga_optimiser._features = data["features"]
    ga_optimiser._rf_model = data["rf_model"]
    ga_optimiser._total_incidents = data["total_incidents"]

    # set parameters
    n_samples = 1000
    n_station = 40
    feasible_cells = np.arange(ga_optimiser._xy_all.shape[0])  # 所有 cell 都可行，也可读取 mask

    # Run a random layout evaluation
    df_random = evaluate_random_layouts(n_samples, n_station, feasible_cells)

    # Save results to CSV
    out_path = "/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layouts.csv"
    df_random.to_csv(out_path, index=False)
    print(f"Saved random layout results to: {out_path}")

    # Plot the fitness distribution of random layouts
    plt.figure(figsize=(10, 6))
    plt.hist(df_random["fitness"], bins=30, color="skyblue", edgecolor="black", alpha=0.8)
    plt.title("Fitness Distribution of 1000 Random Layouts")
    plt.xlabel("Fitness")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layout_hist.png")
    plt.show()

    # calculate the percentile of the current layout fitness
    current_fitness =  49.981  # current layout fitness
    percentile = percentileofscore(df_random["fitness"], current_fitness)
    print(f"Current layout fitness: {current_fitness}")
    print(f"Percentile in random layouts: {percentile:.2f}%")

    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_fitness_histogram.png", dpi=300)