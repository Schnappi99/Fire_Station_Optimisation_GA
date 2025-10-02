# utils/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Evaluator:
    """
    Fitness calculation (for GA and feasibility study)
    """
    xy_all: np.ndarray                # (N, 2)
    time_matrix: np.ndarray          # (N, M)
    A_matrix: np.ndarray             # (N, M) 0/1
    partial_features: np.ndarray     # (N, P) except nearest_time and station_count
    incident_freq: np.ndarray        # (N,)
    rf_model: object                 #  .predict(X_df) -> [0,1]
    total_incidents: float
    feature_names: list[str]         #  names of the five top important features

    def _station_count_all(self, solution: np.ndarray) -> np.ndarray:
        solution = np.asarray(solution, dtype=int)
        A_sub = self.A_matrix[:, solution]                               # (N, k)
        counts = np.asarray(A_sub.sum(axis=1)).ravel().astype(int)       # (N,)
        return counts

    def evaluate_layout(self, solution: np.ndarray):
        """
        Return:
          incidents_served: float
          eff_pct: float
          detail: pd.DataFrame(["nearest_time","station_count","incident_freq","efficiency","expected_served"])
        """
        solution = np.asarray(solution, dtype=int)
        if np.unique(solution).size != solution.size:
            raise ValueError("solution has duplicate indices; must be unique")

        selected_times = self.time_matrix[:, solution]                    # (N, k)
        nearest_times = selected_times.min(axis=1)                        # (N,)
        station_count = self._station_count_all(solution)                 # (N,)

        X = np.column_stack([nearest_times, self.partial_features, station_count])
        X_df = pd.DataFrame(X, columns=self.feature_names)

        eff = np.clip(self.rf_model.predict(X_df), 0.0, 1.0)             # (N,)
        expected_served = eff * self.incident_freq                        # (N,)

        incidents_served = float(expected_served.sum())
        eff_pct = incidents_served / self.total_incidents if self.total_incidents > 0 else 0.0

        detail = pd.DataFrame({
            "nearest_time": nearest_times,
            "station_count": station_count,
            "incident_freq": self.incident_freq,
            "efficiency": eff,
            "expected_served": expected_served,
        })
        return incidents_served, eff_pct, detail

    # Adapter for PyGAD (both signatures support)
    def fitness_pygad(self, solution, solution_idx) -> float:
        incidents_served, _, _ = self.evaluate_layout(np.asarray(solution, dtype=int))
        return float(incidents_served)

    def fitness_pygad_with_ga(self, ga_instance, solution, solution_idx) -> float:
        return self.fitness_pygad(solution, solution_idx)
