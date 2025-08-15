import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimiser.config import DATA_DIR
from sklearn.neighbors import NearestNeighbors

# Set parameters
MUTUAL_ONLY = False  # True: Only draw each other's nearest neighbors, False: Each station to its nearest neighbor
LABEL_FONT_SIZE = 8

# Load data
stations = pd.read_csv(DATA_DIR / "station_locations.csv")
xy = stations[["Easting", "Northing"]].to_numpy()

# Columns for labeling
label_col = "StationID" if "StationID" in stations.columns else None

# Nearest neighbor
nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(xy)
dist, idx = nbrs.kneighbors(xy)

pairs = set()

if MUTUAL_ONLY:
    for i in range(len(xy)):
        j = idx[i, 1]
        if idx[j, 1] == i:
            pairs.add(tuple(sorted((i, j))))
else:
    for i in range(len(xy)):
        j = idx[i, 1]
        pairs.add(tuple(sorted((i, j))))

# Calculate the distance after deduplication
pair_distances = []
for i, j in pairs:
    d = np.linalg.norm(xy[i] - xy[j])
    pair_distances.append(d)

pair_distances = np.array(pair_distances)
mean_distance = pair_distances.mean()

# Plot scatter
plt.figure(figsize=(9, 9))
plt.scatter(xy[:, 0], xy[:, 1], label="Stations", zorder=2)

# Line
for i, j in pairs:
    plt.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], 'r--', linewidth=0.8, zorder=1)

# Add label
for i, (x, y) in enumerate(xy):
    txt = str(stations.loc[i, label_col]) if label_col else str(i)
    plt.annotate(txt, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=LABEL_FONT_SIZE)

# Add the average distance
plt.text(0.02, 0.98,
         f"Average nearest neighbor distance: {mean_distance:.2f} m",
         transform=plt.gca().transAxes,
         ha="left", va="top",
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlabel("Easting")
plt.ylabel("Northing")
plt.title("Nearest station connection")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Average nearest neighbor distance: {mean_distance:.2f} meters")
