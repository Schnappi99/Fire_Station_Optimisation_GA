# run_ga.py
import numpy as np
from joblib import load
from optimiser.ga_optimiser import GAOptimiser, make_single_swap_seeds
from optimiser.data_loader import load_data
from optimiser.config import config, DATA_DIR

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

out_dir = Path("outputs/run_latest")
out_dir.mkdir(parents=True, exist_ok=True)

def save_layout_map(xy_all: np.ndarray, candidate_xy: np.ndarray, best_solution: np.ndarray, out_path="outputs/optimised_layout_map.png"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    # 全部候选点（若 candidate_xy == xy_all 的子集，就画它；否则直接略过）
    if candidate_xy is not None:
        ax.scatter(candidate_xy[:,0], candidate_xy[:,1], s=8, marker="x", alpha=0.3, label="Candidates")

    # 全部格网质心（可选）
    ax.scatter(xy_all[:,0], xy_all[:,1], s=2, alpha=0.2, label="Grid centroids")

    # 选中的站点
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
# Run optimisation
# ------------------------------------------------
opt = GAOptimiser(data=data_dict, config=config, gene_space=candidate_indices)

rng = np.random.default_rng(config.get("random_seed", None))
init_pop = make_single_swap_seeds(start_layout, candidate_indices, 60, config["sol_per_pop"], rng)

best_solution, best_incidents, best_pct, ga = opt.run(
    initial_population=init_pop,
    plot=True,
    verbose=True,
)

# # best_solution, best_incidents, best_pct, ga = opt.run(plot=True, verbose=True)
# best_solution, best_incidents, best_pct, ga = opt.run(
#     plot=True,
#     verbose=True,
#     start_layout=start_layout,   # 以当前布局为起点
#     n_seeds=30,                  # 生成 30 个扰动种子
#     seed_replace_rate=0.2        # 每个种子替换 20% 的站点
# )

# 4) Use results
# ------------------------------------------------
print("Selected candidate indices:", best_solution.tolist())
print(f"Expected efficiently served incidents: {best_incidents:,.0f} "
      f"({best_pct:.2%} of total {total_incidents:,.0f})")

# Evaluate
base_served, base_pct, _ = opt.evaluate_layout(start_layout)
best_served, best_pct, _ = opt.evaluate_layout(best_solution)

print(f"Baseline: {base_served:,.0f} ({base_pct:.2%})")
print(f"Optimised: {best_served:,.0f} ({best_pct:.2%})")
print(f"+{best_served - base_served:,.0f} incidents | Δ{(best_pct - base_pct):.2%}")

# 1) 保存最优布局（索引）
np.save(out_dir / "best_solution.npy", best_solution)
pd.Series(best_solution, name="candidate_index").to_csv(out_dir / "best_solution.csv", index=False)

# 2) 保存关键指标
summary = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "num_stations": int(len(best_solution)),
    "best_incidents": float(best_incidents),
    "best_pct": float(best_pct),
    "total_incidents": float(total_incidents),
}
with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 3) 保存每格网明细（nearest_time / station_count / incident_freq / efficiency / expected_served）
served, pct, detail = opt.evaluate_layout(best_solution)
detail.to_csv(out_dir / "cell_detail.csv", index=False)

# 4) 保存 GA 历史最优曲线
#   - 你在 _on_generation 里已经把 "log/log.csv" 保存了（Time_s, Best_fitness_incidents）
#   - 这里再保存 PyGAD 自带的每代最佳适应度序列：
pd.Series(ga.best_solutions_fitness, name="best_fitness_each_gen").to_csv(
    out_dir / "best_fitness_each_gen.csv", index=False
)

print(f"Results saved to: {out_dir.resolve()}")

# 用法：
# 假设你的候选位置坐标就是某个数组 candidate_xy (C,2)，如果没有就用 xy_all 的对应行：
# candidate_xy = xy_all_candidates
save_layout_map(_xy_all, candidate_xy=None, best_solution=best_solution, out_path="outputs/optimised_layout_map.png")