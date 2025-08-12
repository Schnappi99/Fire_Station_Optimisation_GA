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


def calculate_real_baseline(All_features):
    # 选出需要的列
    cols = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                     'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']

    # 转成 NumPy 数组
    arr = All_features[cols].to_numpy()

    efficiency = _rf_model.predict(arr)
    # 5. 计算全局加权平均效率
    baseline_fitness = np.sum(efficiency * _incident_freq) / np.sum(_incident_freq)

    return baseline_fitness




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
    # Run a random layout evaluation
    df_random = evaluate_random_layouts(n_random_layouts, n_station, feasible_cells)

    # Save results to CSV
    out_path = DATA_DIR.parents[0] / "analysis" / "random_layouts_with_efficiency.csv"
    df_random.to_csv(out_path, index=False)
    print(f"Saved random layout results to: {out_path}")

    #df_random = pd.read_csv("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layouts_with_efficiency.csv")

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
    

    # calculate the percentile of the current layout fitness
    percentile = percentileofscore(df_random["fitness"], baseline_fitness)
    print(f"Current layout fitness: {baseline_fitness}")
    print(f"Percentile in random layouts: {percentile:.2f}%")

    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_fitness_histogram.png", dpi=300)