import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from pandas import read_csv
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import mapclassify
import json

from shapely.geometry import Point

def export_layout_points(
    grid_path,
    idx_path,
    out_dir="outputs",
    grid_idx_col="grid_idx",
    keep_attrs=("grid_id", "grid_idx"),  # 需要保留到点图层的字段
    csv_lonlat=True
):
    """
    从整体grid + 优化后的索引，导出新的布局点(质心)为Shapefile，并可选导出WGS84经纬度CSV。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(grid_path)

    chosen_idx = pd.read_csv(idx_path)

    new_layout_idx = chosen_idx["candidate_index"]
    idx_list = new_layout_idx.to_list()

    grid_map = grid.set_index("grid_idx")
    selected = grid_map.loc[idx_list].reset_index()

    gdf = selected.copy()
    if gdf.crs is None:
        gdf.set_crs(27700, inplace=True)

    gdf["x"] = gdf.geometry.x  # Easting
    gdf["y"] = gdf.geometry.y  # Northing

    gdf[["grid_idx", "grid_id", "x", "y"]].to_csv(out_dir / "layout_points_bng27700.csv", index=False)
    gdf[["grid_idx", "grid_id", "geometry"]].to_file(out_dir / "layout_points.shp", driver="ESRI Shapefile")


dir = Path("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga")
grid_path = dir / "data/output.shp"
dir.mkdir(parents=True, exist_ok=True)

result = export_layout_points(
    grid_path= dir / "data/output.shp",
    idx_path= dir / "optimiser/outputs2/run_latest/best_solution.csv",
    out_dir= dir / "data/exports",
    grid_idx_col="grid_idx",
    keep_attrs=("grid_id", "grid_idx")
)
print(result)