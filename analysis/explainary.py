import numpy as np
import pandas as pd
from optimiser.config import DATA_DIR
from scipy.spatial.distance import pdist


stations = pd.read_csv(DATA_DIR/"station_locations.csv")
stations_xy = stations[["Easting", "Northing"]].to_numpy()

# 计算两两距离（压缩形式的向量）
distances = pdist(stations_xy, metric="euclidean")  # 返回 1D array，长度 n*(n-1)/2

# 排序
distances_sorted = np.sort(distances)

print(distances_sorted)
