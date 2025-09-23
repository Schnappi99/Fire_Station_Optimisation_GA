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
from typing import Tuple

# Plot a comprehensive fire service efficiency heatmap
# including boundaries, fire stations, and other information

from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

## Add a compass
def add_north(ax, labelsize=12, loc_x=0.05, loc_y=1, width=0.04, height=0.06, pad=0.14):
    """
    :param ax: the Axes instance where the compass will be drawn. Use plt.gca() to obtain the current Axes
    :param labelsize: Font size of the 'N'
    :param loc_x: X-axis position ratio for the compass center (value between 0 and 1)
    :param loc_y: Y-axis position ratio for the compass center (value between 0 and 1)
    :param width: Width of the compass as a proportion of the Axes
    :param height: Height of the compass as a proportion of the Axes
    :param pad: Padding between the ‘N’ label and the compass as a proportion of the Axes
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
    right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
    top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
    center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen*loc_x,
            y=miny + ylen*(loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)

## Add a scalebar (for projection coordinate system)
def add_scalebar(ax, x0, y0, length, size):

    ax.hlines(y=y0, xmin=x0, xmax=x0+length, colors="black", ls="-", lw=1, label='%d km' % (length/1000))
    ax.vlines(x=x0, ymin=y0-size, ymax=y0+size, colors="black", ls="-", lw=1)
    ax.vlines(x=x0+length/2, ymin=y0-size, ymax=y0+size, colors="black", ls="-", lw=1)
    ax.vlines(x=x0+length, ymin=y0-size, ymax=y0+size, colors="black", ls="-", lw=1)
    ax.text(x0+length, y0+size+1000, '%d' % (length/1000), horizontalalignment='center')
    ax.text(x0+length/2, y0+size+1000, '%d' % (length/2000), horizontalalignment='center')
    ax.text(x0, y0+size+1000, '0', horizontalalignment='center')
    ax.text(x0+length/2*1.5+3500, y0+size+1000, 'km', horizontalalignment='center')


def calculate_grid_coordinates_vectorized(
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_size: float,
        origin_x: float,
        origin_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    grid_x: Array of grid x indices
    grid_y: Array of grid y indices
    grid_size: Grid cell side length (in same units as coordinates)
    origin_x: Easting coordinate of grid system origin
    origin_y: Northing coordinate of grid system origin

    Returns:
    (x_coords, y_coords) - Both of shape (n,2), where each row contains [min, max] coordinates
"""

    x_min = origin_x + grid_size * grid_x
    y_min = origin_y + grid_size * grid_y
    x_coords = np.column_stack([x_min, x_min + grid_size])
    y_coords = np.column_stack([y_min, y_min + grid_size])
    return x_coords, y_coords

def plot_fire_efficiency_heatmap(
        ax: plt.Axes,
        grid: pd.DataFrame,
        incidents: pd.DataFrame,
        stations: pd.DataFrame,
        city_boundary: gpd.GeoDataFrame = None,
        grid_size: float = 500,
        padding: int = 3
) -> None:
    """
    Parameters:
    ax: matplotlib axes object
    grid: DataFrame containing grid efficiency data, must include grid_x, grid_y and in_time_response_rate columns
    incidents: DataFrame containing incident data, used to determine origin coordinates
    stations: DataFrame containing fire station locations
    city_boundary: City boundary data
    grid_size: Grid size in meters
    padding: Number of padding grids around plot boundaries

    """
    # Calculate actual coordinate range
    easting_start = incidents['EASTINGS'].min()
    northing_start = incidents['NORTHINGS'].min()

    # Calculate plot boundaries
    x_min = easting_start - padding * grid_size
    x_max = easting_start + (grid['grid_x'].max() + padding + 1) * grid_size
    y_min = northing_start - padding * grid_size
    y_max = northing_start + (grid['grid_y'].max() + padding + 1) * grid_size

    # Generate grid line positions
    x_ticks = np.arange(easting_start, x_max, grid_size)
    y_ticks = np.arange(northing_start, y_max, grid_size)

    # Set grid lines but hide tick marks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    # Define color mapping
    norm = mcolors.Normalize(vmin=grid['in_time_response_rate'].min(),
                             vmax=grid['in_time_response_rate'].max())
    # cmap = plt.get_cmap('Reds')
    # cmap = plt.get_cmap('OrRd')
    cmap = plt.get_cmap('YlOrRd')

    # Draw light gray background grids
    for x in x_ticks:
        for y in y_ticks:
            rect = plt.Rectangle((x, y), grid_size, grid_size,
                                 facecolor='lightgray', edgecolor='#C0C0C0', linewidth=0.2)
            ax.add_patch(rect)

    # Calculate all grid coordinates using vectorized function
    x_coords, y_coords = calculate_grid_coordinates_vectorized(
        grid['grid_x'].values,
        grid['grid_y'].values,
        grid_size,
        easting_start,
        northing_start
    )

    # Draw grids with fire service efficiency values
    for i, row in grid.iterrows():
        color = cmap(norm(row['in_time_response_rate']))
        rect = plt.Rectangle((x_coords[i, 0], y_coords[i, 0]), grid_size, grid_size,
                             facecolor=color, edgecolor='#B0B0B0', linewidth=0.2)
        ax.add_patch(rect)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Fire service efficiency')

    # Plot boundaries
    city_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, linestyle='-')

    # Add label of city boundaries
    angle_dict = {
        'Birmingham': 80,
        'Coventry': 115,
        'Wolverhampton': 80,
        'Dudley': 120,
        'Walsall': 45,
        'Solihull': 80,
        'Sandwell': -100
    }

    length_dict = {
        'Birmingham': {'extend': 3000, 'line': 3000},
        'Coventry': {'extend': 3000, 'line': 3000},
        'Wolverhampton': {'extend': 3000, 'line': 3000},
        'Dudley': {'extend': 2000, 'line': 3000},
        'Walsall': {'extend': 3000, 'line': 3000},
        'Solihull': {'extend': 3000, 'line': 3000},
        'Sandwell': {'extend': 8000, 'line': 6000},
    }

    # Plot fire stations
    ax.scatter(
        stations['Easting'],
        stations['Northing'],
        color='blue',
        s=35,
        edgecolor='black',
        linewidth=0.5,
        marker='o',
        label='Fire Station'
    )

    # Add scale bar and north arrow
    x0, y0 = incidents['EASTINGS'].min(), incidents['NORTHINGS'].min()
    add_scalebar(ax, x0, y0, length=8000, size=250)
    add_north(ax)

    # Configure legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='black', linestyle='-', label='Local Authority Boundary'),
        plt.Line2D([0], [0], marker='o', color='w', markersize=8,
                   markerfacecolor='blue', markeredgecolor='black', label='Fire Station')
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        framealpha=1,
        title='Legend'
    )

    # Set coordinate range and labels
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_title("Heatmap of Fire Service Efficiency")
    ax.set_aspect('equal')

def build_stations_from_candidate_idx(
    stations_idx,
    candidate_cells,
    grid,
    incidents,
    *,
    cell_id_col='cell_id',     # grid 里标识格子的列名（真实 cell_id）
    gx='grid_x',
    gy='grid_y',
    ix='EASTINGS',
    iy='NORTHINGS',
    grid_size=500
) -> pd.DataFrame:
    """
    将“候选数组里的索引 stations_idx” → “真实 cell_id” → “(grid_x, grid_y)” → “(Easting, Northing)”

    参数
    - stations_idx: Iterable[int]，指向 candidate_cells 的下标（优化结果）
    - candidate_cells: np.ndarray / list，保存真实 cell_id（与 grid[cell_id_col] 可对应）
    - grid: DataFrame，至少含 [cell_id_col, gx, gy]
    - incidents: DataFrame，至少含 [ix, iy]，用于确定原点（min EASTINGS/NORTHINGS）
    - grid_size: 单格边长（米）

    返回
    - stations_new: DataFrame，列包含 ['Easting','Northing','cell_id','grid_x','grid_y']
    """
    stations_idx = np.asarray(stations_idx, dtype=int)
    candidate_cells = np.asarray(candidate_cells)

    # 1) 索引合法性检查
    if (stations_idx.min() < 0) or (stations_idx.max() >= len(candidate_cells)):
        raise IndexError("stations_idx 中存在越界下标，请检查与 candidate_cells 的对应关系。")

    # 2) 取真实 cell_id
    real_cell_ids = candidate_cells[stations_idx]

    # 3) 建映射表：cell_id -> (grid_x, grid_y)
    lut = (grid[[cell_id_col, gx, gy]]
           .drop_duplicates()
           .set_index(cell_id_col))

    # 4) 按 cell_id 取出 (grid_x, grid_y)
    sel = pd.DataFrame({cell_id_col: real_cell_ids}).join(lut, on=cell_id_col, how='left', validate='m:1')

    # 缺失检查
    missing = sel[sel[gx].isna() | sel[gy].isna()]
    if not missing.empty:
        raise ValueError(f"有 {len(missing)} 个 cell_id 在 grid 中找不到 {gx}/{gy}，请检查数据一致性。"
                         f" 例如：{missing[cell_id_col].head(5).tolist()}")

    # 5) 计算实际坐标（EPSG:27700 米）
    easting_start  = float(incidents[ix].min())
    northing_start = float(incidents[iy].min())
    sel['Easting']  = easting_start  + sel[gx].to_numpy() * grid_size
    sel['Northing'] = northing_start + sel[gy].to_numpy() * grid_size

    # output
    stations_new = sel[['Easting','Northing', cell_id_col, gx, gy]].reset_index(drop=True)
    return stations_new

out_dir = Path("outputs2/run_latest")
out_dir.mkdir(parents=True, exist_ok=True)
stations_idx = pd.read_csv(out_dir+ "/best_solution.csv", index_col=0)
candidate_cells = pd.read_csv("")
grid = gpd.read_file("../grid_geometry_with_metrics.shp")
incidents = pd.read_excel("../Data/wmfs_incidents.xlsx")

stations_new =  gpd.read_file("")
stations_old = read_csv()
# Load boundary data（EPSG:27700）
# boundary_b = gpd.read_file("birmingham_boundary.geojson")
# boundary_c = gpd.read_file("Coventry_boundary.geojson")
# city_boundary = gpd.read_file("Major_Towns_and_Cities_Dec_2015_Boundaries_V2_2022_7629900866091194896.geojson")
# Local_Authority_Districts_December_2021_UK_BGC_2022_-6651079422179559093.geojson

boundaries = gpd.read_file(
'../Data/boundary_json/Local_Authority_Districts_December_2024_Boundaries_UK_BFC_-8514277369542505193.geojson')
if boundaries.crs.to_epsg() != 27700:
    boundaries = boundaries.to_crs(epsg=27700)
target_cities = ['Birmingham', 'Coventry', 'Wolverhampton', 'Dudley', 'West Bromwich', 'Walsall', 'Sandwell',
                 'Solihull']

city_boundary = boundaries[boundaries.LAD24NM.isin(target_cities)]
city_boundary.to_file('WM_boundary.geojson', driver='GeoJSON')

gdf_city_town_major_centroid = gpd.GeoDataFrame(city_boundary, geometry=city_boundary.geometry.centroid,
                                                crs=city_boundary.crs)

if city_boundary.crs.to_epsg() != 27700:
    city_boundary = city_boundary.to_crs(epsg=27700)

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

plot_fire_efficiency_heatmap(
    ax=ax,
    grid=grid,
    incidents=incidents,
    #stations_old=stations,      # 旧站点
    stations_new=stations_new,  # 新站点（刚刚生成）
    city_boundary=city_boundary,
    grid_size=500,
    padding=3)
plt.tight_layout()
plt.show()