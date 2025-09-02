# optimiser/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

@dataclass
class Evaluator:
    xy_all: np.ndarray              # (N,2)
    time_matrix: np.ndarray         # (N,N) —— 所有 demand, 所有候选
    incident_freq: np.ndarray  # (N,)
    partial_features: np.ndarray  # (N,p)
    rf_model: object
    demand_idx: np.ndarray          # (M,)  freq>0 的格子索引
    A_cover: csr_matrix             # (M,N) 稀疏覆盖矩阵
    _sum_incidents: float           # sum(freq[demand_idx])

    # ---------- 构建器 ----------
    @staticmethod
    def build_from_raw(
        *,
        xy_all: np.ndarray,                  # (N,2)
        time_matrix: np.ndarray,             # (N,N)
        incident_freq: np.ndarray,      # (N,)
        partial_features: np.ndarray,   # (N,p)
        rf_model: object,
        radius_m: float = 10_000.0,
        demand_idx: np.ndarray | None = None
    ) -> "Evaluator":
        N = xy_all.shape[0]
        if time_matrix.shape != (N, N):
            raise ValueError(f"time_matrix must be (N,N), got {time_matrix.shape}")
        if demand_idx is None:
            demand_idx = np.where(incident_freq > 0)[0]

        # 构建覆盖矩阵 (M,N)，M = len(demand_idx)
        demand_xy = xy_all[demand_idx]
        tree = cKDTree(demand_xy)
        col_hits = tree.query_ball_point(xy_all, r=radius_m)  # 每个候选列对应 demand 行
        rows, cols, data = [], [], []
        for j, rows_j in enumerate(col_hits):
            if rows_j:
                rows.extend(rows_j)
                cols.extend([j] * len(rows_j))
                data.extend([1] * len(rows_j))
        A_cover = csr_matrix((data, (rows, cols)), shape=(len(demand_idx), N), dtype=np.uint8)

        return Evaluator(
            xy_all=xy_all,
            time_matrix=time_matrix,
            incident_freq=incident_freq,
            partial_features=partial_features,
            rf_model=rf_model,
            demand_idx=demand_idx,
            A_cover=A_cover,
            _sum_incidents=float(incident_freq[demand_idx].sum())
        )

    # ---------- station_count ----------
    def station_count_from_layout(self, layout_idx: np.ndarray) -> np.ndarray:
        """返回 (M,) —— demand cells 的10km站点计数。"""
        counts = self.A_cover[:, np.asarray(layout_idx, dtype=int)].sum(axis=1).A1
        return counts.astype(np.int16, copy=False)

        # ---------- evaluate ----------

    def evaluate(
            layout_idx: np.ndarray,
            *,
            time_matrix: np.ndarray,  # (N,N)
            demand_idx: np.ndarray,  # (M,)
            incident_freq: np.ndarray,  # (N,)
            partial_features: np.ndarray,  # (N,p)
            rf_model: object,
            A_cover: csr_matrix,
            sum_incidents: float
    ) -> float:
        """
        给定 station 布局，返回 demand>0 子集的加权效率。
        """
        layout_idx = np.asarray(layout_idx, dtype=int)

        # 最近时间 (M,)
        subT = time_matrix[demand_idx[:, None], layout_idx]  # (M,|layout|)
        nearest_times = subT.min(axis=1)

        # 覆盖数量 (M,)
        station_num = A_cover[:, layout_idx].sum(axis=1).A1.astype(np.int16)

        # 特征 (M,p)
        features = partial_features[demand_idx]

        # Combine features
        feature_names = ['nearest_station_travel_time', 'neighbour_frequency_per_month',
                         'Agriculture - mainly crops', 'Deciduous woodland', 'station_count']

        # Predict the fire service efficiency
        X = np.column_stack([nearest_times, features, station_num])
        efficiency = rf_model.predict(X)

        # Calculate the fitness
        fitness = np.sum(efficiency * incident_freq) / np.sum(incident_freq)
        print(fitness)

        return float(fitness)


