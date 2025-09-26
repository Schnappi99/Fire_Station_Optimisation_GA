import numpy as np
from joblib import load
#from optimiser.ga_optimiser import GAOptimiser, make_single_swap_seeds_weighted
from optimiser.GA_algorithm import GAOptimiser, make_single_swap_seeds_weighted
from optimiser.data_loader import load_data
from optimiser.data_loader import load_data
from optimiser.config import config, DATA_DIR

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

out_dir = Path("outputs_none/run_latest")
out_dir.mkdir(parents=True, exist_ok=True)

def save_layout_map(xy_all: np.ndarray, candidate_xy: np.ndarray, best_solution: np.ndarray, out_path="outputs_none/optimised_layout_map.png"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    # all candidates (if candidate_xy == xy_all's subset, draw it; otherwise skip)
    if candidate_xy is not None:
        ax.scatter(candidate_xy[:,0], candidate_xy[:,1], s=8, marker="x", alpha=0.3, label="Candidates")

    # all central points
    ax.scatter(xy_all[:,0], xy_all[:,1], s=2, alpha=0.2, label="Grid centroids")

    # selected grid 
    sel_xy = candidate_xy[best_solution] if candidate_xy is not None else xy_all[best_solution]
    ax.scatter(sel_xy[:,0], sel_xy[:,1], s=60, marker="*", label="Selected stations")

    ax.set_title("Optimised Fire Station Locations")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved map to {out_path}")

# Load data
data_dict = load_data()
_xy_all = data_dict["xy_all"]
_incident_freq = data_dict["incident_freq"]
_A_matrix = data_dict["A_matrix"]
_time_matrix = data_dict["time_matrix"]
_partial_features = data_dict["partial_features"]
_rf_model = data_dict["rf_model"]
_total_incidents = data_dict["total_incidents"]

n_station = config["num_stations"]

total_incidents = float(_incident_freq.sum())
start_layout = np.load("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/data/current_layout_idx.npy")

# Gene space = indices of candidate station locations (columns of time_matrix).
# If you already pruned feasible cells, pass their indices here.
candidate_indices = np.arange(_time_matrix.shape[1])  # all candidates

# ---- build domains ----
incident_freq_arr = _incident_freq.ravel().astype(float)
pos_idx = np.flatnonzero(incident_freq_arr > 0).astype(int)


# top-p% set for seeding convenience (not the gene_space)
k = int(config["num_stations"])
p = float(config.get("gene_space_top_pct", 0.10))
if pos_idx.size == 0:
    raise ValueError("No cells with incident_freq > 0.")
# vals : incident>0
vals = incident_freq_arr[pos_idx]
# target: The target scale of the Top set = max(k, ceil(p * number of required grids)), ensuring that at least ≥ k.
target = max(k, int(np.ceil(p * pos_idx.size)))
target = min(target, pos_idx.size)
# Get the top 10% (or more, at least k) grid index; the last row is sorted stably from high to low by frequency (just for easy observation).
part = np.argpartition(-vals, target-1)[:target]
top_cells = pos_idx[part]
top_cells = top_cells[np.argsort(-incident_freq_arr[top_cells])]

# wide gene_space: all incident>0 cells
gene_space = np.flatnonzero(_incident_freq.ravel() > 0).astype(int)

# ---- optimiser ----
opt = GAOptimiser(data=data_dict, config=config, gene_space=gene_space)

# ---- mixed/weighted seeding (optional). If you want GA to random-init → set initial_population=None
rng = np.random.default_rng(config.get("random_seed", None))
# Configurable seeding parameters (given the default value, can also be provided in config)
n_single_swap = int(config.get("n_single_swap_seeds", 60))
alpha = float(config.get("seed_alpha", 1.0))
mix = float(config.get("seed_uniform_mix_ratio", 0.1))

# generate init_pop
init_pop = make_single_swap_seeds_weighted(
    base_layout=start_layout,                # based on current layout to do the single swamp change
    incident_freq=incident_freq_arr,
    #feasible_indices=top10_cells,            # only choose the top 10% demand cells
    feasible_indices=gene_space,            # choose the gene space/ top p %
    n_single_swap_seeds=n_single_swap,
    pop_size=int(config["sol_per_pop"]),
    rng=rng,
    k=k,
    top_pct=p,                               # same as gene_space
    alpha=alpha,
    uniform_mix_ratio=mix
)

best_solution, best_incidents, best_pct, ga = opt.run_single(
    initial_population=None,    #  or init_pop
    start_layout=start_layout,
    plot=True,
    verbose=True,
)

# Use results
print("Selected candidate indices:", best_solution.tolist())
print(f"Expected efficiently served incidents: {best_incidents:,.0f} "
      f"({best_pct:.2%} of total {total_incidents:,.0f})")

# Evaluate
base_served, base_pct, _ = opt.evaluate_layout(start_layout)
best_served, best_pct, _ = opt.evaluate_layout(best_solution)

print(f"Baseline: {base_served:,.0f} ({base_pct:.2%})")
print(f"Optimised: {best_served:,.0f} ({best_pct:.2%})")
print(f"+{best_served - base_served:,.0f} incidents | Δ{(best_pct - base_pct):.2%}")

# save best solution
np.save(out_dir / "best_solution.npy", best_solution)
pd.Series(best_solution, name="candidate_index").to_csv(out_dir / "best_solution.csv", index=False)

# save summary
summary = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "num_stations": int(len(best_solution)),
    "best_incidents": float(best_incidents),
    "best_pct": float(best_pct),
    "total_incidents": float(total_incidents),
}
with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Save results 
# nearest_time / station_count / incident_freq / efficiency / expected_served
served, pct, detail = opt.evaluate_layout(best_solution)
detail.to_csv(out_dir / "cell_detail.csv", index=False)

# Save GA history best fitness curve
#   - "log/log.csv" is saved in _on_generation
#   - Save PyGAD's best fitness sequence
pd.Series(ga.best_solutions_fitness, name="best_fitness_each_gen").to_csv(
    out_dir / "best_fitness_each_gen.csv", index=False
)

print(f"Results saved to: {out_dir.resolve()}")

save_layout_map(_xy_all, candidate_xy=None, best_solution=best_solution, out_path="outputs/optimised_layout_map.png")