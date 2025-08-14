import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from altair import Point
from datashader.layout import random_layout
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import MultiPoint

from typing import List, Optional
from optimiser.data_loader import load_data
import io
from PIL import Image
import ast
import re
from config import DATA_DIR
import imageio.v2 as imageio
from pathlib import Path



def _avg_nn_distance(coords: np.ndarray) -> float:
    """Average nearest-neighbor distance (exclude self)."""
    if len(coords) < 2:
        return np.nan
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(coords)
    dists, _ = nn.kneighbors(coords)  # [:,0] is 0 (self), [:,1] is NN
    return float(dists[:, 1].mean())

def _nni(coords: np.ndarray, area: float) -> float:
    """
    Nearest Neighbor Index = observed_mean_nn / expected_mean_nn (CSR).
    E[r] ≈ 0.5 * sqrt(A / n).
    """
    n = len(coords)
    if n < 2 or area <= 0:
        return np.nan
    r_obs = _avg_nn_distance(coords)
    r_exp = 0.5 * np.sqrt(area / n)
    return float(r_obs / r_exp)

def plot_fire_grid_heatmap_with_gif(
    xy_all: np.ndarray,            # (N, 2) all grid coords
    fire_freq: np.ndarray,         # (N,) fire frequency
    station_coords: np.ndarray,    # (M, 2) current station coords
    random_layouts: list,          # list of ndarrays (indices of new stations)
    candidate_cells: np.ndarray = None, # optional feasible cells
    gif_path: Path = Path("layouts_animation.gif"),
    fps: int = 3,
    point_size: int = 6,
    title_prefix: str = "Random layout"
):
    """
    Generate GIF: On the fire frequency heatmap,
    overlay the current layout by frame with the random layout
    """
    xy_all = np.asarray(xy_all)
    fire_freq = np.asarray(fire_freq).astype(float)
    station_coords = np.asarray(station_coords)

    xmin, ymin = xy_all.min(axis=0)
    xmax, ymax = xy_all.max(axis=0)
    area_bbox = float((xmax - xmin) * (ymax - ymin))

    xs = np.unique(xy_all[:, 0])
    ys = np.unique(xy_all[:, 1])
    is_regular_grid = (xs.size * ys.size == xy_all.shape[0])

    frames = []

    for i, layout_idx in enumerate(random_layouts):
        layout_coords = xy_all[layout_idx]

        fig, ax = plt.subplots(figsize=(9, 7))

        # Plot heatmap
        if is_regular_grid:
            x_to_ix = {v: i for i, v in enumerate(xs)}
            y_to_iy = {v: i for i, v in enumerate(ys)}
            grid = np.full((ys.size, xs.size), np.nan, dtype=float)
            for (x, y), val in zip(xy_all, fire_freq):
                j = x_to_ix[x]
                i_ = y_to_iy[y]
                grid[i_, j] = val
            extent = [xs.min(), xs.max(), ys.min(), ys.max()]
            im = ax.imshow(grid, origin="lower", extent=extent, aspect="equal")
            fig.colorbar(im, ax=ax, label="Fire frequency")
            if candidate_cells is not None:
                cc = xy_all[candidate_cells]
                ax.scatter(cc[:, 0], cc[:, 1], s=point_size, alpha=0.15)
        else:
            sc = ax.scatter(xy_all[:, 0], xy_all[:, 1], s=point_size, c=fire_freq)
            fig.colorbar(sc, ax=ax, label="Fire frequency")
            if candidate_cells is not None:
                cc = xy_all[candidate_cells]
                ax.scatter(cc[:, 0], cc[:, 1], s=point_size, alpha=0.15)

        # Overlay the current layout and random layout
        ax.scatter(station_coords[:, 0], station_coords[:, 1], s=30, marker="^", color="blue", label="Current stations")
        ax.scatter(layout_coords[:, 0], layout_coords[:, 1], s=30, marker="o", color="red", label="Random layout")

        # NNI index
        ann = _avg_nn_distance(layout_coords)
        nni_val = _nni(layout_coords, area_bbox)
        ax.set_title(f"{title_prefix} #{i+1}\nANN={ann:.1f} | NNI={nni_val:.2f}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        image = np.array(Image.open(buf))
        frames.append(image)
        plt.close(fig)


    # Save as GIF
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"GIF saved to {gif_path}")

def parse_layout_str(layout_str):
    return np.fromstring(layout_str.strip("[]"), sep=' ', dtype=int)

if __name__ == "__main__":
    # load data
    data = load_data()
    _xy_all = data["xy_all"]
    _time_matrix = data["time_matrix"]
    _incident_freq = data["incident_freq"]
    _partial_features = data["partial_features"]
    _rf_model = data["rf_model"]
    _total_incidents = data["total_incidents"]

    stations = pd.read_csv(DATA_DIR/"station_information_with_bsv.csv")
    stations_xy = stations[["Easting", "Northing"]].to_numpy()

    feasible_cells = np.arange(_xy_all.shape[0])

    parent_path = DATA_DIR.parent
    print(parent_path)
    df_layout = pd.read_csv(DATA_DIR.parent / "analysis" /"demand_weighted_layouts_2.csv")
    random_layouts = [parse_layout_str(s) for s in df_layout['layout'][:50]]

    plot_fire_grid_heatmap_with_gif(
        xy_all=_xy_all,
        fire_freq=_incident_freq,
        station_coords=stations_xy,
        random_layouts=random_layouts,
        candidate_cells=feasible_cells,  # optional
        gif_path=Path("random_layouts_2.gif"),
        fps=3 )




