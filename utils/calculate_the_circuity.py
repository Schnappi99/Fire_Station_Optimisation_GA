import numpy as np
from scipy.spatial import cKDTree
from optimiser.config import DATA_DIR


def load_data():
    """
    Load the data required for computing the driving time matrix.
    """
    data = {
        "candidate_grid": np.load(DATA_DIR / "xy_all.npy", allow_pickle=True),
        "incident_freq": np.load(DATA_DIR / "incident_freq.npy")
    }

    return data

def compute_circuity_stats(
    candidate_xy: np.ndarray,  # (N,2) candidate grid coords in meters (projected CRS, e.g. EPSG:27700)
    distances_m: np.ndarray,     # (N,N) OSRM network distance matrix in meters
    euclid_radius_m: float = 20_000.0,  # only evaluate pairs with Euclidean distance <= this radius
    per_cell: bool = True        # whether to compute per-row (per cell) median circuity
):
    """
    Compute circuity c = L_network / D_euclid for pairs within a Euclidean radius.
    Returns global distribution stats, and (optionally) per-cell median circuity.

    Tips:
    - Using a radius (e.g., 20 km) keeps memory/time reasonable and focuses on relevant pairs.
    - Make sure candidate_xy_m is in *meters* (projected CRS), not lon/lat.
    """
    assert distances_m.shape[0] == distances_m.shape[1] == candidate_xy.shape[0], \
        "distances_m must be (N,N) aligned with candidate_xy_m (N,2)."

    N = candidate_xy.shape[0]
    tree = cKDTree(candidate_xy)
    # neighbor lists within Euclidean radius
    nn_lists = tree.query_ball_point(candidate_xy, r=euclid_radius_m)

    pair_i = []
    pair_j = []
    # collect upper-triangle pairs (i < j) to avoid duplicates
    for i, js in enumerate(nn_lists):
        for j in js:
            if j > i:
                pair_i.append(i)
                pair_j.append(j)

    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)

    if pair_i.size == 0:
        raise RuntimeError("No pairs found within the given Euclidean radius; increase euclid_radius_m.")

    # Euclidean distances for those pairs
    diff = candidate_xy[pair_i] - candidate_xy[pair_j]
    D = np.linalg.norm(diff, axis=1)                  # Euclidean in meters
    L = distances_m[pair_i, pair_j]                   # Network distance in meters

    # valid mask: finite L, positive D
    valid = np.isfinite(L) & (D > 0)
    D = D[valid]
    L = L[valid]
    if D.size == 0:
        raise RuntimeError("No valid pairs after filtering (check distances_m and coordinates).")

    c = L / D                                         # circuity ratios

    # Global stats
    def p(x, q): return float(np.percentile(x, q))
    stats = {
        "count_pairs": int(c.size),
        "mean": float(np.mean(c)),
        "median": p(c, 50),
        "p80": p(c, 80),
        "p90": p(c, 90),
        "p95": p(c, 95),
        "min": float(np.min(c)),
        "max": float(np.max(c)),
    }

    per_cell_median = None
    if per_cell:
        # per-row median circuity, using all pairs touching row i
        per_cell_median = np.full(N, np.nan, dtype=float)

        # build adjacency by rows
        # gather indices for each row from the valid pair list
        # (i,j) and (j,i) both contribute to row-level stats
        rows_indices = [[] for _ in range(N)]
        valid_i = pair_i[valid]
        valid_j = pair_j[valid]
        c_valid = c  # already filtered

        for idx, (ii, jj) in enumerate(zip(valid_i, valid_j)):
            rows_indices[ii].append(idx)
            rows_indices[jj].append(idx)

        for i in range(N):
            if rows_indices[i]:
                per_cell_median[i] = np.median(c_valid[rows_indices[i]])

    return c, stats, per_cell_median



# candidate_xy: all candidate grid
# distances_m: OSRM road network distance matrix (N,N)
if __name__ == "__main__":
    # Load data
    data = load_data()
    candidate_grid = data["candidate_grid"]
    incident_freq = data["incident_freq"]

    distances_m = np.load(DATA_DIR / "network_distance_meters.npy", allow_pickle=True)
    c_all, circuity_stats, c_per_cell = compute_circuity_stats(
        candidate_xy=candidate_grid,
        distances_m=distances_m,
        euclid_radius_m=20_000.0,
        per_cell=True
    )


print("Circuity stats:", circuity_stats)


# "10 km straight line" is converted to "equivalent road network radius" = 10 km * median(c)
euclid_r = 10_000.0
net_r_equiv = euclid_r * circuity_stats["median"]
print(f"Euclidean 10km ≈ Network {net_r_equiv/1000:.2f} km (using median circuity)")

