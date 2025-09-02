import osrm_utils
import numpy as np
import pandas as pd
from optimiser.config import DATA_DIR
from tqdm import tqdm
import requests


def load_data():
    """
    Load the data required for computing the driving time matrix.
    """
    data = {
   #     "events_grid": np.load(DATA_DIR / "events_grid.npy", allow_pickle=True),
        "candidate_grid": np.load(DATA_DIR / "xy_all.npy", allow_pickle=True),
    }

    return data

def get_osrm_time(incident_xy, station_xy, batch_size_src=50, batch_size_dst=100):
    """
    incident_xy: list/array of (lon, lat) pairs — fire incident locations
    station_xy: list/array of (lon, lat) pairs — candidate station locations
    Returns a numpy array of shape (num_incidents, num_stations)
    """
    all_durations = []

    for i in tqdm(range(0, len(incident_xy), batch_size_src), desc="Incident Batches"):  # outer loop — incident bacth
        batch_src = incident_xy[i:i + batch_size_src]
        src_coords = ";".join([f"{pt[0]},{pt[1]}" for pt in batch_src])
        src_indices = list(range(len(batch_src)))  # local index

        batch_durations = []

        for j in range(0, len(station_xy), batch_size_dst):
            batch_dst = station_xy[j:j + batch_size_dst]
            dst_coords = ";".join([f"{pt[0]},{pt[1]}" for pt in batch_dst])
            dst_indices = list(range(len(batch_dst)))

            try:
                # Request format: /table/v1/driving/src_coords?destinations=...
                url = f"http://127.0.0.1:5010/table/v1/driving/{src_coords};{dst_coords}"
                params = {
                    "sources": ";".join(map(str, src_indices)),
                    "destinations": ";".join(str(len(batch_src) + k) for k in dst_indices)  # shift index for dst
                }

                response = requests.get(url, params=params)
                data = response.json()

                if 'durations' not in data:
                    print(f" Missing durations (src {i}-{i+len(batch_src)}, dst {j}-{j+len(batch_dst)}):")
                    print(data)
                    batch_durations.append(np.full((len(batch_src), len(batch_dst)), np.inf))
                    continue

                durations = np.array(data['durations'])  # shape: (batch_src, batch_dst)
                batch_durations.append(durations)

            except Exception as e:
                print(f" Error in src batch {i}, dst batch {j}: {e}")
                batch_durations.append(np.full((len(batch_src), len(batch_dst)), np.inf))

        # Combine all destination batches for this source batch → shape: (batch_src, num_stations)
        all_durations.append(np.hstack(batch_durations))

    # Combine all source batches → shape: (num_incidents, num_stations)
    full_matrix = np.vstack(all_durations)

    return full_matrix

def compute_matrix(data):
    """
    Compute the driving time matrix using OSRM.
    """
    # Load data
    # events_grid = data["events_grid"]
    candidate_grid = data["candidate_grid"]

    # Transform coordinates from [x, y] to [lon, lat]
    # events_grid = osrm_utils._transform_coords(events_grid)
    candidate_grid= osrm_utils._transform_coords(candidate_grid)

    # Compute the driving time matrix
    driving_time_matrix = get_osrm_time(candidate_grid, candidate_grid, batch_size_src=50, batch_size_dst=100)

    return driving_time_matrix 


if __name__ == "__main__":
    # Load data
    # data = load_data()
    # driving_time_matrix = compute_matrix(data)

    driving_time_matrix = np.load(DATA_DIR / "driving_time_matrix_NN.npy", allow_pickle=True)
    data = np.load(DATA_DIR / "demand_index.npz", allow_pickle=True)
    demand_idx = data["idx"]
    grid_id = pd.read_parquet(DATA_DIR  / "all_index.parquet")
    station_idx = np.load(DATA_DIR/"current_layout_idx.npy", allow_pickle=True)

    selected_times = driving_time_matrix[np.ix_(demand_idx, station_idx)]

    min_time_per_demand = selected_times.min(axis=1)

    print(selected_times.shape)

    df_min_time = pd.DataFrame({
        "grid_idx": demand_idx,
        "min_time_sec": min_time_per_demand
    })

    df_min_time = df_min_time.merge(grid_id[["grid_idx", "grid_id"]], on="grid_idx", how="left")

    # 保存成 CSV
    out_path = DATA_DIR / "min_time_per_demand.csv"
    df_min_time.to_csv(out_path, index=False)

    print(f"Saved CSV: {out_path}")


    # Save the matrix
    np.save(DATA_DIR / "driving_time_matrix_NN.npy", driving_time_matrix)
    print("Driving time matrix saved.")




 
   