# --------------------------------------------------------------
# utils/driving_time_matrix.py
# --------------------------------------------------------------
# Call the local OSRM /table API, returning the driving time of each grid → the nearest fire station (seconds)
# V2: Precomputed and saved travel time & distance matrices, then loaded directly (much faster)
# --------------------------------------------------------------


from statsmodels.stats.dist_dependence_measures import distance_statistics
from scipy.sparse import save_npz, load_npz
import osrm_utils
import numpy as np
import pandas as pd
from optimiser.config import DATA_DIR
from tqdm import tqdm
import requests
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def load_data():
    """
    Load the data required for computing the driving time matrix.
    """
    data = {
        "candidate_grid": np.load(DATA_DIR / "xy_all.npy", allow_pickle=True),
        "incident_freq": np.load(DATA_DIR / "incident_freq.npy")
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

def get_osrm_time_and_dist(incident_xy, station_xy, batch_size_src=50, batch_size_dst=100, timeout=60):
    """
    incident_xy: list/array of (lon, lat) — Origin
    station_xy:  list/array of (lon, lat) — Destination
    Return: (durations_s, distances_m)
      durations_s: shape=(num_incidents, num_stations), seconds
      distances_m: shape=(num_incidents, num_stations), meters (road network distances)
    """
    all_durations = []
    all_distances = []

    for i in tqdm(range(0, len(incident_xy), batch_size_src), desc="Incident Batches"):  # outer loop — incident bacth
        batch_src = incident_xy[i:i + batch_size_src]
        src_coords = ";".join([f"{pt[0]},{pt[1]}" for pt in batch_src])
        src_indices = list(range(len(batch_src)))  # local index

        batch_durations = []
        batch_distances = []

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

                params = {
                    "sources": ";".join(map(str, src_indices)),
                    "destinations": ";".join(str(len(batch_src) + k) for k in dst_indices),
                    "annotations": "duration,distance"
                }

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if "durations" not in data or "distances" not in data:
                    print(f" Missing fields for src [{i}:{i + len(batch_src)}], dst [{j}:{j + len(batch_dst)}]: {data}")
                    batch_durations.append(np.full((len(batch_src), len(batch_dst)), np.inf))
                    batch_distances.append(np.full((len(batch_src), len(batch_dst)), np.inf))
                    continue

                durations = np.array(data['durations'], dtype=float)  # shape: (batch_src, batch_dst)
                distances = np.array(data["distances"], dtype=float)  # network distance
                batch_durations.append(durations)
                batch_distances.append(distances)

            except Exception as e:
                print(f" Error in src batch {i}, dst batch {j}: {e}")
                batch_durations.append(np.full((len(batch_src), len(batch_dst)), np.inf))
                batch_distances.append(np.full((len(batch_src), len(batch_dst)), np.inf))


        # Combine all destination batches for this source batch → shape: (batch_src, num_stations)
        all_durations.append(np.hstack(batch_durations))
        all_distances.append(np.hstack(batch_distances))

    durations_full = np.vstack(all_durations)
    distances_full = np.vstack(all_distances)

    return durations_full, distances_full

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

def compute_matrices(candidate_grid, DATA_DIR):
    """
    The driving time matrix and the road network distance matrix are calculated by OSRM (N,N).
    """
    candidate_lonlat = osrm_utils._transform_coords(candidate_grid)  # -> (N,2) WGS84

    durations_s, distances_m = get_osrm_time_and_dist(
        candidate_lonlat, candidate_lonlat,
        batch_size_src=50, batch_size_dst=100
    )

    np.save(DATA_DIR / "driving_time_seconds.npy", durations_s)
    np.save(DATA_DIR / "network_distance_meters.npy", distances_m)

    return durations_s, distances_m

def build_time_cover_matrix_from_T(
    driving_time_s: np.ndarray,      # T: (N, N), travel time in seconds
    demand_idx: np.ndarray,          # (M,)
    tau: np.ndarray,                 # (M,), time thresholds for each demand cell (seconds)
    candidate_idx: np.ndarray | None = None,  # (Nc,), indices of candidate station cells
    N_total: int | None = None       # total number of candidate stations (for output matrix width)
) -> csr_matrix:
    """
    Build the time-based coverage matrix A_time (M×N)：
      A_time[i,j] = 1  <=>  T[demand_idx[i], j] <= tau[i]
    - Only columns in `candidate_idx` are considered (if provided).
    - Non-finite values (inf/nan) in travel times or tau are ignored.
    Returns:
        A sparse CSR matrix of dtype uint8.
    """

    I = np.asarray(demand_idx)[:, None]
    J = np.asarray(candidate_idx)
    T = driving_time_s[I, J]  # (M, Nc)

    rows, cols, data = [], [], []
    M, Nc = T.shape
    for i in range(M):
        ok = np.where(T[i] <= tau[i])[0]  # 这些是 candidate_idx 内的列号
        if ok.size:
            rows.extend([i] * ok.size)
            cols.extend(J[ok].tolist())   # 回到全局列索引
            data.extend([1] * ok.size)

    A_time = csr_matrix((data, (rows, cols)), shape=(M, N_total), dtype=np.uint8)
    return A_time

def tau_from_network_local(
    distances_m: np.ndarray,    # (N,N)
    durations_s: np.ndarray,    # (N,N)
    r_network_m: float = 8000.0,
    p: float = 90.0,
    fallback: str = "global"    # "global" or "nan"
) -> np.ndarray:
    """
    Per-cell local time threshold tau_i: for each row i, take T_ij over j with L_ij <= r_network_m,
    then compute the p-th percentile. If a row has no valid neighbors, fallback to global percentile or NaN.

    Returns:
        tau_local: (N,) seconds
    """
    assert distances_m.shape == durations_s.shape
    N = durations_s.shape[0]

    # global fallback percentile on all valid pairs within r
    mask_all = (distances_m <= r_network_m) & np.isfinite(durations_s)
    ts_all = durations_s[mask_all]
    if ts_all.size == 0:
        raise RuntimeError("No valid pairs within the network radius for any row.")
    global_q = float(np.percentile(ts_all, p))

    tau = np.full(N, np.nan, dtype=float)
    for i in range(N):
        mask_i = (distances_m[i] <= r_network_m) & np.isfinite(durations_s[i])
        ti = durations_s[i, mask_i]
        tau[i] = float(np.percentile(ti, p)) if ti.size else (global_q if fallback == "global" else np.nan)
    return tau

def build_time_cover_matrix_global(
    driving_time_s: np.ndarray,  # (N, N)
    tau_global: float,           # scalar threshold (seconds)
    candidate_idx: np.ndarray | None = None,
    N_total: int | None = None
) -> csr_matrix:
    """
    Build a global time-based coverage matrix using one scalar threshold.
    A[i,j] = 1 if T[i,j] <= tau_global.
    """
    N = driving_time_s.shape[0]
    if candidate_idx is None:
        candidate_idx = np.arange(N, dtype=int)
    if N_total is None:
        N_total = N

    # Boolean mask of coverage
    mask = (driving_time_s[:, candidate_idx] <= tau_global)
    rows, cols = mask.nonzero()
    data = np.ones(len(rows), dtype=np.uint8)

    A_time = csr_matrix((data, (rows, candidate_idx[cols])), shape=(N, N_total))
    return A_time

def tau_from_speed(distance_m=10_000.0, speed_mph=30.0):
    # 1 mile = 1609.344 m
    speed_mps = speed_mph * 1609.344 / 3600.0
    return distance_m / speed_mps


if __name__ == "__main__":
    # Load data
    data = load_data()
    candidate_grid = data["candidate_grid"]
    incident_freq = data["incident_freq"]

    # Calculate the durations and distances by OSRM
    # durations_s, distances_m = compute_matrices(candidate_grid, DATA_DIR)


    # demand_idx = np.where(incident_freq > 0)[0]
    # candidate_idx = np.arange(data["candidate_grid"].shape[0], dtype=int)

    durations_s = np.load( DATA_DIR / "driving_time_seconds.npy", allow_pickle=True)
    distances_m = np.load( DATA_DIR / "network_distance_meters.npy", allow_pickle=True)

    demand_mask = (incident_freq > 0)
    demand_idx = np.flatnonzero(demand_mask)
    np.savez(DATA_DIR / "demand_index.npz", mask=demand_mask, idx=demand_idx)

    # tau_local = tau_from_network_local(distances_m, durations_s, r_network_m = 8000.0, p = 90.0,
    # fallback= "global")
    #
    # A_time = build_time_cover_matrix_global(durations_s, tau_local,
    #                                         candidate_idx=np.arange(durations_s.shape[1]),
    #                                         N_total=durations_s.shape[1])
    # print(tau_local)

    tau = tau_from_speed(distance_m=10_000.0, speed_mph=30.0)
    # 30 mph ≈ 13.4112 m/s → tau ≈ 10_000 / 13.4112 ≈ 745.6 s ≈ 12.43 min
    print(tau)

    N = durations_s.shape[0]
    cand = np.arange(N)
    mask = (durations_s[:, cand] <= tau) & np.isfinite(durations_s[:, cand])

    rows, cols = mask.nonzero()
    A_time = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cand[cols])),
        shape=(N, N), dtype=np.uint8
    )

    # A_time = build_time_cover_matrix_from_T(
    #     driving_time_s=durations_s,
    #     demand_idx=np.arange(durations_s.shape[0]),
    #     tau=tau,
    #     candidate_idx=np.arange(durations_s.shape[1]),
    #     N_total=durations_s.shape[1]
    # )

    # Save tau.npy and A_time.npz
    save_npz(DATA_DIR / "A_time_all_normal.npz", A_time)

    # Check A_time
    num_nonzero = A_time.nnz

    # Total number of elements (M * N)
    total_elements = A_time.shape[0] * A_time.shape[1]

    # Number of zero entries
    num_zero = total_elements - num_nonzero

    print(f"Total elements: {total_elements}")
    print(f"Non-zero elements (coverage relation = 1): {num_nonzero}")
    print(f"Zero elements (not covered = 0): {num_zero}")
    print(f"Sparsity: {num_zero / total_elements:.2%}")








 
   