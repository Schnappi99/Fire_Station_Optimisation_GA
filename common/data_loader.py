import numpy as np
import pandas as pd
from joblib import load
from scipy.sparse import load_npz

from .config import DATA_DIR

def load_data():
    data = {
        "xy_all": np.load(DATA_DIR / "xy_all.npy"),
        "time_matrix": np.load(DATA_DIR / "driving_time_matrix_NN.npy"),
        "A_matrix": load_npz(DATA_DIR / "A_time_all_normal.npz"),
        "incident_freq": np.load(DATA_DIR / "incident_freq.npy"),
        "partial_features": np.load(DATA_DIR / "partial_features.npy", allow_pickle=True),   # without the number of stations and the shortest travel time
        "rf_model": load(DATA_DIR / "rf_model.joblib"),
    }
    data["total_incidents"] = np.sum(data["incident_freq"])
    return data