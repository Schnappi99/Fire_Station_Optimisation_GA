# ga_run.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from optimiser.GAOptimiser import GAOptimiser
from utils.data_loader import load_data
from utils.config import config, DATA_DIR


def save_layout_map(xy_all, candidate_xy, best_solution, out_path="outputs/optimised_layout_map.png"):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if candidate_xy is not None:
        ax.scatter(candidate_xy[:, 0], candidate_xy[:, 1], s=8, marker="x", alpha=0.3, label="Candidates")
    ax.scatter(xy_all[:, 0], xy_all[:, 1], s=2, alpha=0.2, label="Grid centroids")
    sel_xy = (candidate_xy if candidate_xy is not None else xy_all)[best_solution]
    ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=60, marker="*", label="Selected stations")
    ax.set_title("Optimised Fire Station Locations")
    ax.set_aspect("equal"); ax.legend(); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved map to {out_path}")

if __name__ == "__main__":

    # setting the path
    out_dir = Path("outputs/run_latest"); out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    gene_space = np.flatnonzero(data['incident_freq'] > 0).astype(int)
    # build optimiser
    opt = GAOptimiser(data=data, config=config, gene_space=gene_space)
    start_layout = np.load(DATA_DIR / "current_layout_idx.npy")

    # Read from config (it could be "20%" or a numeric value like 0.2)
    top_raw = config.get("gene_space_top_pct")
    # Convert "x%" → 0.x
    if isinstance(top_raw, str) and top_raw.endswith("%"):
        top_pct = float(top_raw.strip("%")) / 100.0
    else:
        top_pct = float(top_raw)

    print(top_raw)

    init_pop = opt.build_initial_population(
        base_layout=start_layout,
        pop_size=int(opt.config["sol_per_pop"]),
        mode="mixed",
        # with spacing constraint
        # min_dist=float(config.get("min_station_spacing", 3000.0)),
        # enforce_spacing=True,
        n_single_swap_from_base=int(opt.config.get("n_single_swap_seeds", 60)),
        alpha=float(opt.config.get("seed_alpha", 1.0)),
        uniform_mix_ratio=float(opt.config.get("seed_uniform_mix_ratio", 0.1)),
        top_pct=top_pct,
    )

    # 6) run GA (single-swap mutation)
    best_solution, best_incidents, best_pct, ga = opt.run_single(
        start_layout=start_layout,
        initial_population=init_pop,
        plot=True,
        verbose=True,
    )

    # 7) evaluate & save
    base_served, base_pct, _ = opt.evaluate_layout(start_layout)
    best_served, best_pct2, detail = opt.evaluate_layout(best_solution)

    print(f"Baseline:  {base_served:,.0f} ({base_pct:.2%})")
    print(f"Optimised: {best_served:,.0f} ({best_pct2:.2%})")
    print(f"+{best_served - base_served:,.0f} incidents | {(best_pct2 - base_pct):.2%}")

    np.save(out_dir / "best_solution.npy", best_solution)
    pd.Series(best_solution, name="candidate_index").to_csv(out_dir / "best_solution.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gene_space_top_pct": top_pct,
        "num_stations": int(len(best_solution)),
        "best_incidents": float(best_served),
        "best_pct": float(best_pct2),
        "top_pct": top_pct,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    detail.to_csv(out_dir / "cell_detail.csv", index=False)

    # GA best per generation
    pd.Series(ga.best_solutions_fitness, name="best_fitness_each_gen").to_csv(
        out_dir / "best_fitness_each_gen.csv", index=False
    )

    # save_layout_map(xy_all, candidate_xy=None, best_solution=best_solution,
    #                 out_path="outputs/k40_pop200_gen300_top0.20_mut0.20_running_20251014-101109/optimised_layout_map.png")
    #
    # print(f"Results saved to: {out_dir.resolve()}")