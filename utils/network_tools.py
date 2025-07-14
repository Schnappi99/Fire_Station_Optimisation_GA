# --------------------------------------------------------------
# utils/network_tools.py
# --------------------------------------------------------------
# 调用本地 OSRM /table API，返回每个 grid → 最近消防站的行驶时间（秒）
# --------------------------------------------------------------
import numpy as np
import requests
import pandas as pd
import json
from pyproj import CRS, Transformer
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List
       

def _transform_coords(coords):
    """
    coords: (n,2)  [x, y]  →  (n,2)  [lon, lat]  
    """
    transformer = Transformer.from_crs(CRS.from_epsg(27700), CRS.from_epsg(4326), always_xy=True)
    return np.array(transformer.transform(coords[:, 0], coords[:, 1])).T  

def _coord_str(coords):
    # coords: (n,2)  [lon, lat]  →  "lon,lat;lon,lat;..."
    return ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)


OSRM_URL      = "http://127.0.0.1:5000"
INIT_BATCH    = 200      # 初始 batch 尺寸
MIN_BATCH     = 20       # 最小降到 20 仍然失败就报错
TIMEOUT       = 60       # 单次请求超时
RETRY_NUM     = 3        # 每批最多重试 3 次
SLEEP_SEC     = 1        # 失败后等待时间

def _coord_str(coords: np.ndarray) -> str:
    """ 
    coords: (n,2)  [x, y]  →  (n,2)  [lon, lat]  
    coords (n,2) [lon,lat] → 'lon,lat;lon,lat;...' 
    """
    transformer = Transformer.from_crs(CRS.from_epsg(27700), CRS.from_epsg(4326), always_xy=True)
    return ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)

def _post_table(st_xy: np.ndarray, batch_xy: np.ndarray) -> np.ndarray:
    """
    单次 POST /table 请求，返回 (batch,) 最近行驶时间（秒）
    st_xy    : (N_station,2)
    batch_xy : (batch,2)
    """
    n_station = len(st_xy)
    coords    = np.vstack([st_xy, batch_xy])

    body = {
        "coordinates":  coords.tolist(),                 # [[lon,lat], ...]
        "sources":      list(range(n_station)),          # 站点索引
        "destinations": list(range(n_station, n_station + len(batch_xy))),
        "annotations":  "duration"                       # 要时长矩阵
    }

    resp = requests.post(
        f"{OSRM_URL}/table/v1/car",
        data=json.dumps(body),
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT
    )
    resp.raise_for_status()

    durations = np.array(resp.json()["durations"])       # (n_station, batch)
    return np.min(durations, axis=0)                     # (batch,)

# ----------------------------------------------------------------
def robust_osrm_matrix(station_xy: np.ndarray,
                       grid_xy:    np.ndarray,
                       init_batch: int = INIT_BATCH) -> np.ndarray:
    """
    station_xy : (N_station, 2) [lon,lat]
    grid_xy    : (N_cell,   2)  [lon,lat]
    返回       : (N_cell,)  每格到最近站点的行驶时间（秒）
    · 自动降批量：失败时 batch //=2，直至成功或 < MIN_BATCH
    """
    results: List[np.ndarray] = []
    i, B = 0, init_batch

    while i < len(grid_xy):
        batch = grid_xy[i : i + B]

        # ---- 带重试 ----
        success = False
        for attempt in range(RETRY_NUM):
            try:
                min_times = _post_table(station_xy, batch)      # (batch,)
                results.append(min_times)
                i += B
                success = True
                break
            except Exception as e:
                if attempt < RETRY_NUM - 1:
                    time.sleep(SLEEP_SEC)
                else:
                    # 如果重试次数用尽，但可以再降批量，就降一半继续
                    if B > MIN_BATCH:
                        B //= 2
                        print(f"[OSRM] Batch failed, reduce batch to {B} (idx {i})")
                    else:
                        raise RuntimeError(
                            f"OSRM request repeatedly failed at idx {i}: {e}"
                        ) from e
        if not success and B <= MIN_BATCH:
            # 说明降批量后依旧失败（已在上面 raise），这里仅防御
            break

    return np.concatenate(results)

if __name__ == "__main__":
    # Example usage
    station_df = pd.read_csv("/Users/zhaoyuxin/Repos/Fire_service_efficiency/Code/fire_station_optimisation_ga/data/station_information_with_bsv.csv")
    incident_df = pd.read_csv("/Users/zhaoyuxin/Repos/Fire_service_efficiency/Code/fire_station_optimisation_ga/data/incident_temp.csv")
    
    station_xy_27700 = station_df[["Easting", "Northing"]].to_numpy()  # shape = (n_station, 2)
    incident_xy_27700 = incident_df[["EASTINGS", "NORTHINGS"]].to_numpy()  # shape = (n_incident, 2)
 
    station_xy = _transform_coords(station_xy_27700)   # → 可直接送入 OSRM
    incident_xy = _transform_coords(incident_xy_27700)

    min_times = robust_osrm_matrix(station_xy, incident_xy)
    print(min_times)


    





  