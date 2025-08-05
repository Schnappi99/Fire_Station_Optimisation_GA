# --------------------------------------------------------------
# utils/network_tools.py
# --------------------------------------------------------------
# Call the local OSRM /table API, returning the driving time of each grid → the nearest fire station (seconds)
# --------------------------------------------------------------

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time




def _transform_coords(coords):
    """
    coords: (n,2)  [x, y]  →  (n,2)  [lon, lat]  
    """
    transformer = Transformer.from_crs(CRS.from_epsg(27700), CRS.from_epsg(4326), always_xy=True)
    return np.array(transformer.transform(coords[:, 0], coords[:, 1])).T

def table_parser(loc : np.ndarray, sources_list : list):
    out1, sou, des = '', '', ''
    for i in range(loc.shape[0]):
        out1 += str(loc[i, 0]) + ',' + str(loc[i, 1]) + ';'
        if i in sources_list:
            sou += str(i) + ';'
        else:
            des += str(i) + ';'
    out1 = out1[:-1]
    out2 = {'sources': sou[:-1], 'destinations': des[:-1]}
    return out1, out2

def get_osrm_time(event, station):
    """
    event:  (n,2)  [lon, lat]
    station: (40,2)  [lon, lat]
    This function is to calculate the minimal driving time from the fire station to each incident
    """

    min_times = []
    start_time = time.time()  # record the start time

    sample_size = 200  # the number of events to handle

    for i in tqdm(range(0, event.shape[0], sample_size)):
        # the event index of the current batch
        event_list = [j for j in range(i, min(i + sample_size, event.shape[0]))]

        # combine the coordinate list as the type what OSRM need（station + event_batch）
        temp = np.concatenate([station, event[event_list]], axis=0)

        # URL
        locs, kv = table_parser(temp, [j for j in range(station.shape[0])])  # station 是 source

        try:
            response = requests.get('http://127.0.0.1:5010/table/v1/driving/' + locs, params=kv)
            durations = np.array(response.json()['durations']).T  # shape: (num_events, num_stations)
            batch_min_times = np.min(durations, axis=1)
            min_times.append(batch_min_times)
        except Exception as e:
            print(f" Error in batch {i}: {e}")

    # end time record
    end_time = time.time()

    min_times_array = np.concatenate(min_times)
    print("Done. Shape:", min_times_array.shape)
    print(f" Total runtime: {end_time - start_time:.2f} seconds")

    return min_times_array






if __name__ == "__main__":
    # Example usage
    station_df = pd.read_csv("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/data/station_information_with_bsv.csv")
    incident_df = pd.read_csv("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/data/incident_temp.csv")
    
    station_xy_27700 = station_df[["Easting", "Northing"]].to_numpy()  # shape = (n_station, 2)
    incident_xy_27700 = incident_df[["EASTINGS", "NORTHINGS"]].to_numpy()  # shape = (n_incident, 2)
 
    station_xy = _transform_coords(station_xy_27700)   
    incident_xy = _transform_coords(incident_xy_27700)

    min_times = get_osrm_time(incident_xy, station_xy)
    print(min_times)


    





  