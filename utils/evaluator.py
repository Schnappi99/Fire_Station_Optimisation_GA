# utils/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# Calculate the time threshold
def compute_time_threshold(distance_km: float, speed_mph: float = 30.0) -> float:
    # mph -> m/s
    v_mps = speed_mph * 1609.34 / 3600.0
    D_m = distance_km * 1000.0
    return D_m / v_mps

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
    #total_incidents: float
    feature_names: list[str]         #  names of the five top important features
    station_time_threshold: float | None = None  # global time threshold τ (seconds)

    def _station_count_all_1(self, solution: np.ndarray) -> np.ndarray:
        solution = np.asarray(solution, dtype=int)
        A_sub = self.A_matrix[:, solution]                               # (N, k)
        counts = np.asarray(A_sub.sum(axis=1)).ravel().astype(int)       # (N,)
        return counts

    def _station_count_all(self, solution: np.ndarray) -> np.ndarray:
        """
        Count how many stations are within the time threshold (tau)
        for each incident cell.

        For a given solution (selected station indices):
          - Take the corresponding time columns from time_matrix (N, k).
          - Mark stations with travel_time <= station_time_threshold as 1, else 0.
          - Sum over stations to get the count per incident cell.
        """

        solution = np.asarray(solution, dtype=int)
        # (N, k): travel times from each incident to selected stations
        selected_times = self.time_matrix[:, solution]
        # (N, k) bool: True if station is within the time threshold τ
        within_threshold = selected_times <= self.station_time_threshold
        # (N,): count stations within τ for each incident
        counts = within_threshold.sum(axis=1).astype(int)
        return counts


    def evaluate_layout(self, solution: np.ndarray):
        """
        Return:
          total efficiency: float
          eff: float
        """
        solution = np.asarray(solution, dtype=int)
        if np.unique(solution).size != solution.size:
            raise ValueError("solution has duplicate indices; must be unique")

        selected_times = self.time_matrix[:, solution]                    # (N, k)
        nearest_times = selected_times.min(axis=1)                        # (N,)
        station_count = self._station_count_all(solution)                 # (N,)

        X = np.column_stack([nearest_times, self.partial_features, station_count])
        X_df = pd.DataFrame(X, columns=self.feature_names)

        # calculate eff of each cell
        eff = np.clip(self.rf_model.predict(X_df), 0.0, 1.0)

        # Each incident cell contributes according to its occurrence frequency (incident_freq)
        # Multiply predicted efficiency (eff) by incident frequency to get weighted service level per cell
        weighted_efficiency = eff * self.incident_freq

        # Sum across all incidents to get the total expected number (or proportion) of effectively served incidents
        total_efficiency = float(weighted_efficiency.sum())

        # fire_freq: sum = 1
        total_incidents = float(np.sum(self.incident_freq))
        if abs(total_incidents - 1.0) < 1e-6:
            eff_pct = weighted_efficiency  # already normalized
        else:
            eff_pct = weighted_efficiency / total_incidents if total_incidents > 0 else 0.0

        return total_efficiency, eff

    def fitness_pygad(self, solution, solution_idx) -> float:
        total_efficiency, eff = self.evaluate_layout(np.asarray(solution, dtype=int))
        return float(total_efficiency)

    def fitness_pygad_with_ga(self, ga_instance, solution, solution_idx) -> float:
        # ga_instance: which can get the current statement (generations, populations, best solution) of the GA.
        # which meets PyGAD 2.20.0 the method signature requirements.
        return self.fitness_pygad(solution, solution_idx)
