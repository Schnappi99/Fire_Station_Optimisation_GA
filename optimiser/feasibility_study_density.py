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
from typing import Optional, Dict, Iterable

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# set global variables
_xy_all = None
_time_matrix = None
_incident_freq = None
_partial_features = None
_rf_model = None
_total_incidents = None

# Centroid points of all grids (index = grid)
def make_buffers_from_xy(xy_all, buffer_m=1000):

    cent = gpd.GeoDataFrame(
        {'grid': np.arange(len(xy_all), dtype=int)},
        geometry=[Point(xy) for xy in xy_all],
        crs="EPSG:27700"
    )

    buffer_gdf = cent.copy()
    buffer_gdf['geometry'] = buffer_gdf.geometry.buffer(buffer_m)


    buffer_idx = buffer_gdf[['grid', 'geometry']].set_index('grid')
    buffer_for_join = buffer_idx.reset_index().rename(columns={'grid': 'grid_buf'})

    return cent, buffer_idx, buffer_for_join

# 2) 本次 layout 的站点点集（solution 直接当行号/索引使用）
def make_station_gdf(solution, xy_all):
    station = gpd.GeoDataFrame(
        {'grid': solution},
        geometry=[Point(xy) for xy in xy_all[solution]],
        crs="EPSG:27700"
    )
    return station.to_crs(27700)

def count_stations_per_buffer(xy_all, station_gdf, buffer_m=10000):
    """
    计算每个grid的缓冲区内station数量

    参数：
        xy_all: (N, 2) numpy 数组，所有grid中心点坐标（EPSG:27700）
        station_gdf: GeoDataFrame，列=['grid','geometry']，CRS=EPSG:27700
        buffer_m: 缓冲半径（米）
    返回：
        pd.Series，index=grid，值=station_num
    """
    assert str(station_gdf.crs) == "EPSG:27700"

    # 1) 构建grid中心点
    cent = gpd.GeoDataFrame(
        {'grid': np.arange(len(xy_all), dtype=int)},
        geometry=[Point(xy) for xy in xy_all],
        crs="EPSG:27700"
    )

    # 2) 构建缓冲区（带 grid_buf 列，方便 sjoin）
    buffer_for_join = cent.copy()
    buffer_for_join['geometry'] = buffer_for_join.geometry.buffer(buffer_m)
    buffer_for_join = buffer_for_join.rename(columns={'grid': 'grid_buf'})

    # 3) 空间连接（左=buffer，右=station）
    joined = gpd.sjoin(buffer_for_join, station_gdf, how='left', predicate='contains')

    # 4) 统计每个缓冲区内的station数量
    station_num = (joined.groupby('grid_buf')['index_right']
                   .count()
                   .reindex(buffer_for_join['grid_buf'], fill_value=0)
                   .astype(int)
                   .rename('station_num'))

    # 5) 索引名统一
    station_num.index.name = 'grid'
    return station_num

def fitness_function(ga_instance, solution, solution_idx):
    # 假设 travel_time_matrix 是 NumPy 数组，shape: (M, N)
    # 假设 solution 是 array，表示你选择的 station 的 grid_idx，长度为 40

    selected_times = _time_matrix[:, solution]
    nearest_times = selected_times.min(axis=1)

    station_gdf = make_station_gdf(solution, _xy_all)
    station_num = count_stations_per_buffer(_xy_all, station_gdf, buffer_m=10000)

    feature_names = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']

    X = pd.DataFrame(
        np.column_stack([nearest_times, _partial_features, station_num]),
        columns=feature_names
    )

    efficiency = _rf_model.predict(X)
    fitness = np.sum(efficiency * _incident_freq) / np.sum(_incident_freq)
    print(fitness)
    return float(fitness)

def evaluate_baseline_efficiency(station_ids):
    """
    station_ids: current layout idx (.npy)
    time_matrix: (M, N) incident→grid travel time matrix
    partial_features: (N, k) grid-level 特征
    rf_model: 已训练的随机森林模型
    incident_freq: (M,) 每个 incident 的火警频率
    """
    # 1. 取每个 incident 到原始 layout 所有站的时间
    selected_times = _time_matrix[:, station_ids]   # (M, len(stations))
    nearest_times = selected_times.min(axis=1)     # (M,)

    station_gdf = make_station_gdf(station_ids, _xy_all)
    station_num = count_stations_per_buffer(_xy_all, station_gdf, buffer_m=10000)

    feature_names = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']
    X = pd.DataFrame(
        np.column_stack([nearest_times, _partial_features, station_num]),
        columns=feature_names
    )
    # 4. 预测效率
    efficiency = _rf_model.predict(X)
    # 5. 计算全局加权平均效率
    baseline_fitness = np.sum(efficiency * _incident_freq) / np.sum(_incident_freq)

    return baseline_fitness

def _compute_quotas_density_constrained(
    candidate_cells: np.ndarray,
    n_stations: int,
    zone_id: np.ndarray,                 # 长度 = N，全局分区编号
    quota_strategy: str = "from_current",# "from_current" | "by_incident" | "by_feasible"
    incident_freq: Optional[np.ndarray] = None,
    current_layout: Optional[Iterable[int]] = None,
) -> Dict[int, int]:
    """
    根据策略生成每个分区的目标配额，总和 = n_stations。
    - from_current: 按现网在各区数量（需要 current_layout）
    - by_incident:  按各区需求总量比例（需要 incident_freq）
    - by_feasible:  按各区可行候选数量比例
    """
    zones, counts_feasible = np.unique(zone_id[candidate_cells], return_counts=True)
    zones = zones.astype(int)
    zone_list = list(zones)

    if quota_strategy == "from_current":
        if current_layout is None:
            raise ValueError("quota_strategy='from_current' requires current_layout.")
        z_curr = zone_id[np.asarray(list(current_layout))]
        zc, cc = np.unique(z_curr, return_counts=True)
        quota_raw = {int(z): 0 for z in zone_list}
        for z, c in zip(zc.astype(int), cc.astype(int)):
            if z in quota_raw:
                quota_raw[z] = c
        # 缩放/补齐到 n_stations
        total = sum(quota_raw.values())
        if total != n_stations:
            scale = n_stations / max(total, 1)
            quota = {z: int(np.floor(c * scale)) for z, c in quota_raw.items()}
            remainder = n_stations - sum(quota.values())
            if remainder > 0:
                # 按原始配额权重分配余量（没有就均分）
                weights = np.array([quota_raw[z] for z in zone_list], dtype=float)
                if weights.sum() == 0:
                    weights = np.ones_like(weights)
                probs = weights / weights.sum()
                order = np.argsort(-probs)
                i = 0
                while remainder > 0:
                    quota[zone_list[order[i % len(order)]]] += 1
                    remainder -= 1
                    i += 1
        else:
            quota = quota_raw

    elif quota_strategy == "by_incident":
        if incident_freq is None:
            raise ValueError("quota_strategy='by_incident' requires incident_freq.")
        # 各区总需求
        demand_by_zone = {}
        for z in zone_list:
            mask = (zone_id == z)
            demand_by_zone[z] = float(incident_freq[mask].sum())
        weights = np.array([demand_by_zone[z] for z in zone_list], dtype=float)
        if weights.sum() == 0:
            # 兜底：退回按可行数量
            weights = counts_feasible.astype(float)
        probs = weights / weights.sum()
        quota = {z: int(np.floor(p * n_stations)) for z, p in zip(zone_list, probs)}
        remainder = n_stations - sum(quota.values())
        # largest remainder
        frac = (probs * n_stations) - np.floor(probs * n_stations)
        order = np.argsort(-frac)
        i = 0
        while remainder > 0:
            quota[zone_list[order[i % len(order)]]] += 1
            remainder -= 1
            i += 1

    elif quota_strategy == "by_feasible":
        weights = counts_feasible.astype(float)
        probs = weights / weights.sum()
        quota = {z: int(np.floor(p * n_stations)) for z, p in zip(zone_list, probs)}
        remainder = n_stations - sum(quota.values())
        frac = (probs * n_stations) - np.floor(probs * n_stations)
        order = np.argsort(-frac)
        i = 0
        while remainder > 0:
            quota[zone_list[order[i % len(order)]]] += 1
            remainder -= 1
            i += 1

    else:
        raise ValueError("Unknown quota_strategy. Choose from {'from_current','by_incident','by_feasible'}.")

    return quota


# ---------- 2) 密度受限抽样 + 评估 ----------

def density_constrained_sample(
    candidate_cells: np.ndarray,           # 可行候选（全局索引）
    n_stations: int,                       # 需要抽取的站点数量
    zone_id: np.ndarray,                   # 长度=N 的分区编号数组
    quota_strategy: str = "from_current",  # "from_current" | "by_incident" | "by_feasible"
    incident_freq: Optional[np.ndarray] = None,
    current_layout: Optional[Iterable[int]] = None,
    use_demand_weight_within_zone: bool = True,
    epsilon: float = 1e-6,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    先为每个分区分配配额，再在分区内抽样（可选：按需求加权），合并得到总布局（不放回）。
    返回：长度 = n_stations 的全局索引（升序）。
    """
    if rng is None:
        rng = np.random.default_rng()

    quotas = _compute_quotas_density_constrained(
        candidate_cells=candidate_cells,
        n_stations=n_stations,
        zone_id=zone_id,
        quota_strategy=quota_strategy,
        incident_freq=incident_freq,
        current_layout=current_layout
    )

    chosen_list = []
    zones = np.unique(zone_id[candidate_cells]).astype(int)

    # 分区内抽样
    for z in zones:
        q = int(quotas.get(z, 0))
        if q <= 0:
            continue

        in_zone = candidate_cells[zone_id[candidate_cells] == z]
        if len(in_zone) == 0:
            continue

        if len(in_zone) <= q:
            # 候选不足，先全取
            chosen_list.append(in_zone)
            quotas[z] = len(in_zone)
            continue

        if use_demand_weight_within_zone and (incident_freq is not None):
            # 分区内按需求加权
            freq_subset = incident_freq[in_zone].astype(float)
            weights = np.maximum(freq_subset, epsilon) ** alpha
            probs = weights / weights.sum()
            chosen_z = rng.choice(in_zone, size=q, replace=False, p=probs)
        else:
            # 分区内等概率
            chosen_z = rng.choice(in_zone, size=q, replace=False)

        chosen_list.append(np.sort(chosen_z))

    if not chosen_list:
        raise ValueError("No cells selected per-zone. Check quotas and candidate_cells.")

    chosen = np.unique(np.concatenate(chosen_list))

    # 如果总数不足（因为部分区候选少于配额），在剩余可行里补齐
    deficit = n_stations - len(chosen)
    if deficit > 0:
        remaining = np.setdiff1d(candidate_cells, chosen, assume_unique=False)
        if len(remaining) < deficit:
            raise ValueError("Not enough remaining candidates to meet n_stations.")
        if use_demand_weight_within_zone and (incident_freq is not None):
            freq_subset = incident_freq[remaining].astype(float)
            weights = np.maximum(freq_subset, epsilon) ** alpha
            probs = weights / weights.sum()
            extra = rng.choice(remaining, size=deficit, replace=False, p=probs)
        else:
            extra = rng.choice(remaining, size=deficit, replace=False)
        chosen = np.sort(np.concatenate([chosen, extra]))

    # 若超出（理论上不该发生），随机剔除
    if len(chosen) > n_stations:
        drop = len(chosen) - n_stations
        drop_idx = rng.choice(np.arange(len(chosen)), size=drop, replace=False)
        chosen = np.sort(np.delete(chosen, drop_idx))

    return chosen


def evaluate_density_constrained_layouts(
    n_layouts: int,                # 采样次数
    n_stations: int,               # 每个布局的消防站数量
    candidate_cells: np.ndarray,   # 可行的候选位置（全局索引）
    incident_freq: np.ndarray,     # 历史事件频次（长度 = N）
    zone_id: np.ndarray,           # 分区编号（长度 = N）
    quota_strategy: str = "from_current",  # or "by_incident" / "by_feasible"
    current_layout: Optional[Iterable[int]] = None,
    use_demand_weight_within_zone: bool = True,
    epsilon: float = 1e-6,
    alpha: float = 1.0
) -> pd.DataFrame:
    """
    生成多组“密度/配额受限”的随机布局并计算其 fitness。
    返回包含布局索引和 fitness 的 DataFrame。
    """
    results = []
    print(f"Evaluating {n_layouts} density-constrained layouts with {n_stations} stations...")

    for _ in tqdm(range(n_layouts)):
        layout_indices = density_constrained_sample(
            candidate_cells=candidate_cells,
            n_stations=n_stations,
            zone_id=zone_id,
            quota_strategy=quota_strategy,
            incident_freq=incident_freq,
            current_layout=current_layout,
            use_demand_weight_within_zone=use_demand_weight_within_zone,
            epsilon=epsilon,
            alpha=alpha
        )
        # 计算 fitness（兼容你的接口）
        fitness_val = fitness_function(None, layout_indices, 0)
        results.append((layout_indices, fitness_val))

    return pd.DataFrame(results, columns=["layout", "fitness"])


if __name__ == "__main__":

    # load data
    data = load_data()
    _xy_all = data["xy_all"]
    _time_matrix = data["time_matrix"]
    _incident_freq = data["incident_freq"]
    _partial_features = data["partial_features"]
    _rf_model = data["rf_model"]
    _total_incidents = data["total_incidents"]

    # The baseline efficiency
    current_layout_idx = np.load(DATA_DIR / "current_layout_idx.npy")
    All_features = pd.read_csv(DATA_DIR / "All_features.csv")
    baseline_fitness = evaluate_baseline_efficiency(current_layout_idx)
    # baseline_fitness = calculate_real_baseline(All_features)

    # set parameters: number of station is import from config; n_random_layouts is import from config
    n_station = config["num_stations"]
    feasible_cells = np.arange(_xy_all.shape[0])

    # Sampling 1000 demand weighted random Layout
    df_density_constraint = evaluate_density_constrained_layouts(
        n_layouts=1000,
        n_stations=n_station,
        candidate_cells=feasible_cells,
        incident_freq=_incident_freq,
        epsilon=1e-6,
        alpha=1.0
    )

    # 查看前几行结果
    print(df_demand_weighted.head())

    # 统计平均 fitness
    print("Average fitness:", df_demand_weighted["fitness"].mean())

    # Save results to CSV
    out_path = DATA_DIR.parents[0] / "analysis" / "demand_weighted_layouts_with_efficiency.csv"
    df_demand_weighted.to_csv(out_path, index=False)
    print(f"Saved random layout results to: {out_path}")

    #df_random = pd.read_csv("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layouts_with_efficiency.csv")

    # Plot the fitness distribution of random layouts   # current layout fitness
    plt.figure(figsize=(10, 6))
    plt.hist(df_demand_weighted["fitness"], bins=30, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(baseline_fitness, color='red', linestyle='--', linewidth=2, label='Current layout efficiency')
    plt.title("Efficiency Distribution of 1000 Random Layouts")
    plt.xlabel("Efficiency")
    plt.ylabel("Number of Studies")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    # calculate the percentile of the current layout fitness
    percentile = percentileofscore(df_demand_weighted["fitness"], baseline_fitness)
    print(f"Current layout fitness: {baseline_fitness}")
    print(f"Percentile in random layouts: {percentile:.2f}%")

    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_fitness_histogram.png", dpi=300)