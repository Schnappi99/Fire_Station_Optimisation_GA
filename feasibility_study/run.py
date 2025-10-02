# feasibility_study/run.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from pathlib import Path

from utils.data_loader import load_data
from utils.config import DATA_DIR, n_random_layouts, config
from utils.evaluator import Evaluator

from feasibility_study.methods.demand_weighted_single_sample import demand_weighted_single_sample
from feasibility_study.methods.best_k import (
    demand_weighted_single_swap
)


def main():
    # Load data and build Evaluator
    data = load_data()
    ev = Evaluator(
        xy_all=data["xy_all"],
        time_matrix=data["time_matrix"],
        A_matrix=data["A_matrix"],
        partial_features=data["partial_features"],
        incident_freq=np.asarray(data["incident_freq"], dtype=float).ravel(),
        rf_model=data["rf_model"],
        total_incidents=float(data["total_incidents"]),
        feature_names=[
            "nearest_station_travel_time",
            "neighbour_frequency_per_month",
            "Agriculture - mainly crops",
            "Deciduous woodland",
            "station_count",
        ],
    )

    # baseline and feasibility study
    current_layout_idx = np.load(DATA_DIR / "current_layout_idx.npy")
    feasible_cells = np.where(ev.incident_freq > 0)[0]
    baseline_inc, _, _ = ev.evaluate_layout(current_layout_idx)

    # run the method
    # best_k
    df_runs = demand_weighted_single_swap(
        evaluator=ev,
        current_layout_idx=current_layout_idx,
        iterations=1000,
        candidate_cells=feasible_cells,
        MIN_DIST=3000.0,
        K_best=30,
        random_state=42,
        show_progress=True  # time_progress
    )

    # # demand_weighted_sample
    # df_runs = demand_weighted(
    #     evaluator=ev,
    #     current_layout_idx=current_layout_idx,
    #     iterations=1000,
    #     candidate_cells=feasible_cells,
    #     MIN_DIST=3000.0,
    #     alpha=1.0,
    #     uniform_mix_ratio=0.0,
    #     mutual_exclusion=True,
    #     accept_if_better=False,  # Set to True = Only better accept (greed)
    #     random_state=42,
    #     show_progress=True
    # )

    # Visualisation
    # out_dir = DATA_DIR.parents[0] / "results"/ "demand_weighted"
    out_dir = DATA_DIR.parents[0] / "results" / "best_k"
    out_dir.mkdir(parents=True, exist_ok=True)

    vals = df_runs["fitness"].dropna().to_numpy()
    baseline = float(df_runs.attrs.get("baseline_fitness", baseline_inc))

    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(baseline, color="red", linestyle="--", linewidth=2,
                label=f"Current layout efficiency: {baseline:.4f}")
    plt.title(f"Efficiency Distribution of {len(vals)} Single-Station Swaps (K={30})")
    plt.xlabel("Efficiency")
    plt.ylabel("Count")
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / "single_swap_random_walk_hist.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved histogram to: {out_png}")

    if vals.size > 0:
        pct = percentileofscore(vals, baseline)
        print(f"Current layout fitness: {baseline:.6f}")
        print(f"Percentile in single-swap runs: {pct:.2f}%")
        print(f"Average fitness over runs: {np.nanmean(vals):.6f}")
    else:
        print("No valid fitness recorded (all iterations were skipped).")

    out_csv = out_dir / "single_swap_runs.csv"
    df_runs.to_csv(out_csv, index=False)
    print(f"Saved runs to: {out_csv}")

if __name__ == "__main__":
    main()