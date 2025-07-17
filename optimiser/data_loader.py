import numpy as np
import pandas as pd
from joblib import load
from .config import DATA_DIR

def load_data():
    data = {
        "xy_all": np.load(DATA_DIR / "xy_all.npy"),
        "incident_xy": np.load(DATA_DIR / "incident_xy.npy"),
        "incident_freq": np.load(DATA_DIR / "incident_freq.npy"),
        "features": np.load(DATA_DIR / "Five_features.npy", allow_pickle=True),
        "incident_grid_idx": np.load(DATA_DIR / "incident_grid_idx.npy"),
        "rf_model": load(DATA_DIR / "rf_model.joblib"),
    }
    data["total_incidents"] = np.sum(data["incident_freq"])
    return data