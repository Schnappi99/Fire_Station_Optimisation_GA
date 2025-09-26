import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import percentileofscore

from GA_algorithm import GAOptimiser
from data_loader import load_data

from config import DATA_DIR
from config import n_random_layouts
from config import config

from tqdm import tqdm
from typing import Optional
from scipy.spatial import cKDTree
import time

# set global variables
_xy_all = None
_time_matrix = None
_incident_freq = None
_partial_features = None
_rf_model = None
_total_incidents = None
_neighbors = None
_A_time = None

_RADIUS_M = 10_000.0


def precompute_neighbors(xy_all: np.ndarray, radius: float = _RADIUS_M):
    """一次性计算每个格子的 radius 邻居索引"""
    tree = cKDTree(xy_all)
    neighbors = tree.query_ball_point(xy_all, r=radius)
    return neighbors  # list of lists

def station_count_from_layout(layout_cells: np.ndarray, neighbors, n_cells: int):
    """
    Calculate the number of sites within 10km of each grid based on the current layout.
    layout_cells: (K,) grid index of the current site
    neighbors: list[list[int]], pre-calculated neighborhood table
    """
    counts = np.zeros(n_cells, dtype=np.int16)
    for cell in layout_cells:
        counts[neighbors[cell]] += 1
    return counts

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
    # station_num = count_stations_per_buffer(_xy_all, solution, buffer_m=10000)

    # calculate the station number under each new layout
    station_num = station_count_from_layout(solution, _neighbors, _xy_all.shape[0])

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

def evaluate_demand_weighted_layouts_single(
    current_layout_idx: np.ndarray,      # start from your baseline layout
    iterations: int,                     # number of single-station swaps
    candidate_cells: np.ndarray,         # feasible cells (incident_freq > 0)
    incident_freq: np.ndarray,           # (N,)
    MIN_DIST: int,
    epsilon: float = 1e-6,
    alpha: float = 1.0,
    uniform_mix_ratio: float = 0.0,
    random_state: Optional[int] = None,
    mutual_exclusion: bool = True
) -> pd.DataFrame:
    """
    Single-station demand-weighted random walk:
    - Each iteration: move ONE station to a new feasible cell sampled by demand weights.
    - Enforces mutual exclusion and a fixed min_dist=3000m using global _xy_all.
    Returns DataFrame: iter, fitness, moved_station, old_cell, new_cell (NaN fitness if skipped).
    """
    rng = np.random.default_rng(random_state)

    # --- precompute demand probs over feasible set ---
    weights = np.maximum(incident_freq[candidate_cells].astype(float), epsilon) ** alpha
    if uniform_mix_ratio > 0.0:
        lam = 1.0 - float(uniform_mix_ratio)
        weights = lam * weights + (1.0 - lam) * 1.0
    base_prob = weights / weights.sum()

    layout = np.array(current_layout_idx, dtype=int).copy()
    n_stations = layout.size
    baseline_fitness = fitness_function(None, layout, 0)

    # map feasible cell -> its position in candidate_cells for fast masking
    pos_in_candidates = {c: i for i, c in enumerate(candidate_cells)}

    records = []
    for it in tqdm(range(1, iterations + 1), desc="Random walk", unit="iter", mininterval=1.0):
        s = rng.integers(0, n_stations)
        old_cell = layout[s]

        allowed = np.ones(candidate_cells.shape[0], dtype=bool)

        # mutual exclusion (except the moved station's own old_cell)
        if mutual_exclusion:
            occ = set(layout.tolist())
            occ.discard(old_cell)
            for oc in occ:
                j = pos_in_candidates.get(oc)
                if j is not None:
                    allowed[j] = False

        # spacing to all other stations (3000 m), using global _xy_all
        if n_stations > 1:
            others = np.delete(layout, s)
            if others.size > 0:
                cand_xy = _xy_all[candidate_cells]
                others_xy = _xy_all[others]
                dmin = np.min(np.linalg.norm(cand_xy[:, None, :] - others_xy[None, :, :], axis=2), axis=1)
                allowed &= (dmin >= MIN_DIST)

        # exclude staying put
        j_old = pos_in_candidates.get(old_cell)
        if j_old is not None:
            allowed[j_old] = False

        if not allowed.any():
            records.append({
                "iter": it, "fitness": np.nan, "moved_station": s,
                "old_cell": old_cell, "new_cell": None, "note": "skip_no_feasible_target"
            })
            continue

        probs = np.zeros_like(base_prob)
        probs[allowed] = base_prob[allowed]
        probs /= probs.sum()

        new_cell = rng.choice(candidate_cells, p=probs)
        layout[s] = new_cell

        f = fitness_function(None, layout, 0)
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

def _choose_best_of_k(
    layout: np.ndarray,
    station_idx: int,
    candidate_cells: np.ndarray,
    allowed_mask: np.ndarray,
    base_prob: np.ndarray,
    K: int,
    rng: np.random.Generator
) -> int:
    """
    在 allowed_mask 指定的候选集合中，按 base_prob 加权不放回抽取 K 个候选，
    对每个候选临时评估 fitness，返回使 fitness 最大的 cell（返回 -1 表示无可选）。
    """
    # 构造允许子集的概率
    probs = np.zeros_like(base_prob)
    probs[allowed_mask] = base_prob[allowed_mask]
    s = probs.sum()
    if s <= 0:
        return -1
    probs /= s

    allowed_idx = np.flatnonzero(allowed_mask)  # 这些是 candidate_cells 的“位置索引”
    if allowed_idx.size == 0:
        return -1

    K_eff = min(int(K), allowed_idx.size)
    # 在允许集合内按权重不放回抽样
    sampled_pos = rng.choice(allowed_idx, size=K_eff, replace=False, p=probs[allowed_idx])

    best_cell = -1
    best_f = -np.inf
    old_cell = layout[station_idx]

    for pos in sampled_pos:
        cand_cell = candidate_cells[pos]
        layout[station_idx] = cand_cell
        f = fitness_function(None, layout, 0)
        if f > best_f:
            best_f = f
            best_cell = cand_cell

    # 还原
    layout[station_idx] = old_cell
    return best_cell


def evaluate_random_layouts(
    current_layout_idx: np.ndarray,
    iterations: int,
    candidate_cells: np.ndarray,
    incident_freq: np.ndarray,
    MIN_DIST: float = 3000.0,
    epsilon: float = 1e-6,
    alpha: float = 1.5,
    uniform_mix_ratio: float = 0.05,
    random_state: int | None = 42,
    mutual_exclusion: bool = True
) -> pd.DataFrame:
    """
    Pure random single-station swap:
      - 每次迭代：随机挑一个站（按ownership权重抽样），在允许集合里随机选一个新cell（按demand权重）；
      - 不做best-of-K，只评估一次；
      - 返回 iter, fitness, moved_station, old_cell, new_cell, layout。
    """
    rng = np.random.default_rng(random_state)

    # --- demand 权重 ---
    w = np.maximum(incident_freq[candidate_cells].astype(float), epsilon) ** alpha
    if uniform_mix_ratio > 0.0:
        lam = 1.0 - float(uniform_mix_ratio)
        w = lam * w + (1.0 - lam) * 1.0
    base_prob = w / w.sum()

    layout = np.array(current_layout_idx, dtype=int).copy()
    n_stations = layout.size
    baseline_fitness = fitness_function(None, layout, 0)

    pos_in_candidates = {c: i for i, c in enumerate(candidate_cells)}

    # ownership 权重: 哪个站负责的 demand 多 → 更容易被抽到
    def compute_station_probs(layout_idx: np.ndarray) -> np.ndarray:
        sel = _time_matrix[:, layout_idx]  # (M, S)
        arg = np.argmin(sel, axis=1)
        w_st = np.bincount(arg, weights=_incident_freq, minlength=layout_idx.size).astype(float)
        if w_st.sum() <= 0:
            w_st = np.ones_like(w_st, dtype=float)
        return w_st / w_st.sum()

    prob_pick_station = compute_station_probs(layout)

    records = []

    for it in tqdm(range(1, iterations + 1), desc="Random walk", unit="iter", mininterval=1.0):
        # step 1: 挑一个站
        s = rng.choice(np.arange(n_stations), p=prob_pick_station)
        old_cell = layout[s]

        allowed = np.ones(candidate_cells.shape[0], dtype=bool)

        # mutual exclusion (except the moved station's own old_cell)
        if mutual_exclusion:
            occ = set(layout.tolist()); occ.discard(old_cell)
            for oc in occ:
                j = pos_in_candidates.get(oc)
                if j is not None:
                    allowed[j] = False

        # spacing constraint
        if n_stations > 1:
            others = np.delete(layout, s)
            if others.size > 0:
                cand_xy = _xy_all[candidate_cells]
                others_xy = _xy_all[others]
                dmin = np.min(np.linalg.norm(cand_xy[:, None, :] - others_xy[None, :, :], axis=2), axis=1)
                allowed &= (dmin >= float(MIN_DIST))

        # exclude staying put
        j_old = pos_in_candidates.get(old_cell)
        if j_old is not None:
            allowed[j_old] = False

        if not allowed.any():
            # 没有可行目标，跳过
            records.append({
                "iter": it,
                "fitness": np.nan,
                "moved_station": s,
                "old_cell": old_cell,
                "new_cell": None,
                "layout": layout.copy()
            })
            # ownership 不变
            continue

        # step 2: 随机挑一个新位置
        probs = np.zeros_like(base_prob)
        probs[allowed] = base_prob[allowed]
        probs /= probs.sum()

        new_cell = rng.choice(candidate_cells, p=probs)
        layout[s] = new_cell

        # step 3: evaluate
        f = fitness_function(None, layout, 0)

        records.append({
            "iter": it,
            "fitness": f,
            "moved_station": s,
            "old_cell": old_cell,
            "new_cell": new_cell,
            "layout": layout.copy()
        })

        # update station probs
        prob_pick_station = compute_station_probs(layout)

    df_runs = pd.DataFrame.from_records(records)
    df_runs.attrs["baseline_fitness"] = baseline_fitness
    return df_runs


if __name__ == "__main__":
    # Load data
    data = load_data()
    _xy_all = data["xy_all"]
    _time_matrix = data["time_matrix"]
    _incident_freq = data["incident_freq"]
    _partial_features = data["partial_features"]
    _rf_model = data["rf_model"]
    _total_incidents = data["total_incidents"]

    _neighbors = precompute_neighbors(_xy_all, radius=10_000)

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
    # df_runs = evaluate_demand_weighted_layouts(
    #     current_layout_idx=current_layout_idx,
    #     iterations=n_random_layouts,
    #     candidate_cells=feasible_cells,     # cells with _incident_freq > 0
    #     incident_freq=_incident_freq,
    #     MIN_DIST=2000,
    #     epsilon=1e-6,
    #     alpha=1.0,
    #     uniform_mix_ratio=0.0,
    #     random_state=42,
    #     mutual_exclusion=True
    # )

    # K_best
    # df_runs = evaluate_random_layouts(
    #     current_layout_idx=current_layout_idx,
    #     iterations=n_random_layouts,
    #     candidate_cells=feasible_cells,  # _incident_freq > 0
    #     incident_freq=_incident_freq,
    #     MIN_DIST=3000.0,
    #     K_best=30,  # 可调：30/50/100
    #     epsilon=1e-6, alpha=1.0, uniform_mix_ratio=0.0,
    #     random_state=42, mutual_exclusion=True
    # )

    # new
    # df_runs = evaluate_random_layouts(
    #     current_layout_idx=current_layout_idx,
    #     iterations=n_random_layouts,
    #     candidate_cells=feasible_cells,  # _incident_freq > 0
    #     incident_freq=_incident_freq,
    #     MIN_DIST=3000.0,
    #     epsilon=1e-6, alpha=1.0, uniform_mix_ratio=0.0,
    #     random_state=42, mutual_exclusion=True)

    df_runs = evaluate_demand_weighted_layouts_single(
        current_layout_idx=current_layout_idx,  # 基线布局 (n_stations,)
        iterations=1000,  # 迭代次数
        candidate_cells=feasible_cells,  # 可行格子 = incident>0 的索引
        incident_freq=_incident_freq,  # (N,)
        MIN_DIST=2000,  # 最小间距
        epsilon=1e-6,
        alpha=1.0,  # 需求权重幂指数
        uniform_mix_ratio=0.0,  # 与均匀分布混合比例
        random_state=42,
        mutual_exclusion=True
    )

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
    out_png = DATA_DIR.parents[0] / "analysis" / "single_swap_random_walk_hist_4.png"
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
    out_csv = DATA_DIR.parents[0] / "analysis" / "single_swap_runs_4.csv"
    df_runs.to_csv(out_csv, index=False)
    print(f"Saved runs to: {out_csv}")



