import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import percentileofscore

from optimiser.evaluator import Evaluator
from optimiser import data_loader

from config import DATA_DIR
from config import n_random_layouts
from config import config
from tqdm import tqdm

def evaluate_demand_weighted_layouts_single(
    evaluator,
    current_layout_idx: np.ndarray,
    iterations: int,
    candidate_cells: np.ndarray,
    MIN_DIST: float = 3000.0,
    baseline_fitness: float | None = None,   # 新增参数
    epsilon: float = 1e-6,
    alpha: float = 1.0,
    uniform_mix_ratio: float = 0.0,
    random_state: int | None = None,
    mutual_exclusion: bool = True
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    # —— 统一从 evaluator 取全局数据 —— #
    xy_all = evaluator.xy_all
    incident_freq = evaluator.incident_freq_full

    # 候选集合上的 demand 权重
    w = np.maximum(incident_freq[candidate_cells].astype(float), epsilon) ** alpha
    if uniform_mix_ratio > 0.0:
        lam = 1.0 - float(uniform_mix_ratio)
        w = lam * w + (1.0 - lam) * 1.0
    base_prob = w / w.sum()

    layout = np.array(current_layout_idx, dtype=int).copy()
    n_stations = layout.size

    pos_in_cands = {c: i for i, c in enumerate(candidate_cells)}
    records = []

    for it in tqdm(range(1, iterations + 1), desc="Feasibility (random 1-swap)", unit="iter", mininterval=0.5):
        s = rng.integers(0, n_stations)
        old_cell = layout[s]

        allowed = np.ones(candidate_cells.shape[0], dtype=bool)

        # 互斥（除去当前站的旧位置）
        if mutual_exclusion:
            occ = set(layout.tolist()); occ.discard(old_cell)
            for oc in occ:
                j = pos_in_cands.get(oc)
                if j is not None:
                    allowed[j] = False

        # 距离约束
        if n_stations > 1:
            others = np.delete(layout, s)
            if others.size > 0:
                cand_xy = xy_all[candidate_cells]
                others_xy = xy_all[others]
                dmin = np.min(np.linalg.norm(cand_xy[:, None, :] - others_xy[None, :, :], axis=2), axis=1)
                allowed &= (dmin >= float(MIN_DIST))

        # 不允许原地
        j_old = pos_in_cands.get(old_cell)
        if j_old is not None:
            allowed[j_old] = False

        if not allowed.any():
            records.append({
                "iter": it, "fitness": np.nan, "moved_station": s,
                "old_cell": old_cell, "new_cell": None, "layout": layout.copy(),
                "note": "skip_no_feasible_target"
            })
            continue

        probs = np.zeros_like(base_prob)
        probs[allowed] = base_prob[allowed]
        probs /= probs.sum()

        new_cell = rng.choice(candidate_cells, p=probs)
        layout[s] = new_cell

        f = evaluator.evaluate(layout)
        records.append({
            "iter": it,
            "fitness": f,
            "moved_station": s,
            "old_cell": old_cell,
            "new_cell": new_cell,
            "layout": layout.copy()
        })

    df_runs = pd.DataFrame.from_records(records)
    df_runs.attrs["baseline_fitness"] = baseline_fitness
    return df_runs







if __name__ == "__main__":

    data = data_loader.load_data()
    xy_all = data["xy_all"]  # (N,2)
    time_matrix = data["time_matrix"]  # (N,N)
    incident_freq = data["incident_freq"]  # (N,)
    partial_features = data["partial_features"]  # (N,p)
    rf_model = data["rf_model"]

    # 构建一次 Evaluator（N×N 版本）
    evaluator = Evaluator.build_from_raw(
        xy_all=xy_all,
        time_matrix=time_matrix,
        incident_freq=incident_freq,
        partial_features=partial_features,
        rf_model=rf_model,
        radius_m=10_000.0,
        demand_idx=None  # 默认为 incident_freq>0
    )

    # Calculate the current efficiency
    current_layout_idx = np.load(DATA_DIR / "current_layout_idx.npy")
    baseline_fitness = evaluator.evaluate(current_layout_idx)

    feasible_cells = np.where(incident_freq > 0)[0]

    df_runs = evaluate_demand_weighted_layouts_single(
        evaluator=evaluator,
        current_layout_idx=current_layout_idx,
        iterations=n_random_layouts,
        candidate_cells=feasible_cells,
        MIN_DIST=3000,
        baseline_fitness=baseline_fitness,
        epsilon=1e-6,
        alpha=1.0,  # 需求权重幂指数
        uniform_mix_ratio=0.0,  # 与均匀分布混合比例
        random_state=42,
        mutual_exclusion=True
    )

    print(df_runs.head())
    print("Baseline:", df_runs.attrs["baseline_fitness"])


    # Histogram (x-axis from 0)
    plt.figure(figsize=(10, 6))
    vals = df_runs["fitness"].dropna().to_numpy()
    plt.hist(vals, bins=30, color="skyblue", edgecolor="black", alpha=0.8)

    plt.axvline(baseline_fitness, color="red", linestyle="--", linewidth=2,
                label=f"Current layout efficiency: {baseline_fitness:.4f}")
    plt.title(f"Efficiency Distribution of {n_random_layouts} Single-Station Swaps")
    plt.xlabel("Efficiency")
    plt.ylabel("Count")
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save BEFORE show (safer)
    out_png = DATA_DIR.parents[0] / "analysis" / "random_change_1.png"
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"Saved histogram to: {out_png}")

    # Percentile of baseline among runs
    if vals.size > 0:
        pct = percentileofscore(vals, baseline_fitness)
        print(f"Current layout fitness: {baseline_fitness:.6f}")
        print(f"Percentile in single-swap runs: {pct:.2f}%")
        print(f"Average fitness over runs: {np.nanmean(vals):.6f}")
    else:
        print("No valid fitness recorded (all iterations were skipped).")

    # Save runs
    out_csv = DATA_DIR.parents[0] / "analysis" / "random_change_1.csv"
    df_runs.to_csv(out_csv, index=False)
    print(f"Saved runs to: {out_csv}")















