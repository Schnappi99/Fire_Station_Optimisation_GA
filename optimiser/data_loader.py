import numpy as np
import pandas as pd
from joblib import load
from .config import DATA_DIR

def load_data():
    data = {
        "xy_all": np.load(DATA_DIR / "xy_all.npy"),
        "time_matrix": np.load(DATA_DIR / "driving_time_matrix.npy"),
        "incident_freq": np.load(DATA_DIR / "incident_freq.npy"),
        "partial_features": np.load(DATA_DIR / "partial_features.npy", allow_pickle=True),
        "rf_model": load(DATA_DIR / "rf_model.joblib"),
    }
    data["total_incidents"] = np.sum(data["incident_freq"])
    return data