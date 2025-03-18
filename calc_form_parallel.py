#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute braided form (parallel computation)

do in the environment 'geopand'
Created on Tue Oct  8 12:43:57 2024

@author: safiya
"""

## run in environment 'geopand' !!!

import os
import math
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio.mask import mask
import glob
import rasterio.features
from multiprocessing import Pool
from PIL import Image

#%% define functions

# Function to extract raster values along a linestring
def extract_raster_values_along_line(linestring, raster):
    num_points = int(linestring.length)  # or adjust based on your need
    points = [linestring.interpolate(float(i) / num_points, normalized=True) for i in range(num_points + 1)]
   
    # Convert points to numpy arrays of x and y coordinates
    x_coords = np.array([point.x for point in points])
    y_coords = np.array([point.y for point in points])
    
    # Get the raster indices for all points
    rows, cols = np.array(raster.index(x_coords, y_coords))

    # Check if any point is out of bounds
    if np.any(rows < 0) or np.any(rows >= raster.height) or np.any(cols < 0) or np.any(cols >= raster.width):
        return None  # Skip the entire line if any point is out of bounds

    # Read the raster data once
    raster_data = raster.read(1)

    # Extract the values using the indices
    raster_values = raster_data[rows, cols]
    
    return raster_values.tolist()

river = 'brahmaputra_pandu'
def process_river(river, results_base, all_river_tiffs, years):
    print(f"Processing river: {river}")

    centerline_fol = os.path.join(results_base, river)
    
    # fig, ax = plt.subplots(5, 5, figsize=(15, 15), dpi=True, sharex=True, sharey=True)
    # ax = ax.ravel()
    error_years = []
    thread_widths = np.array([])
    for a, year in enumerate(years):
        save_path = os.path.join(results_base, river, f'form-master_{year}.csv')
        # save_path_tw = os.path.join(results_base, f'000_thread_widths/{river}_threadwidths.npy')
        if os.path.exists(save_path):
            print(f'{year} continue')
            continue
        try:
            # transect_path = os.path.join(centerline_fol, str(year), f'{river}_meshlines.shp')
            transect_path = os.path.join(centerline_fol, str(year), f'brahmaputra_pandu_meshlines.shp')
            if not os.path.exists(transect_path):
                continue
            
            transect = gpd.read_file(transect_path)
            
            raster_path = glob.glob(os.path.join(all_river_tiffs, river, 'mask/1999on', f'*{year}*.tif'))
            if not raster_path:
                continue
            
            raster = rasterio.open(raster_path[0])
            
            if transect.crs != raster.crs:
                transect = transect.to_crs(raster.crs)
                
            transect_df = pd.DataFrame(columns=['min', 'max', 'mean', 'sd', 5, 10, 15, 20, 25, 30, 25, 40,
                                                  45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 
                                                  'ebi', 'wetted_width', 'thread_count'], index=transect['FID'])    
            
            for idx, row in transect.iterrows():
                linestring = row.geometry
                raster_values = extract_raster_values_along_line(linestring, raster)
                
                if raster_values:
                    raster_values = np.array(raster_values, dtype=int)
                    
                    edges = np.diff(raster_values)
                    l_edges = np.where(edges == 1)[0]
                    r_edges = np.where(edges == -1)[0]
                    
                    if len(l_edges) > len(r_edges):
                        widths = r_edges - l_edges[:len(r_edges)]
                    
                    elif len(r_edges)>len(l_edges):
                        widths = r_edges[1:] - l_edges
                    
                    else:
                        widths = r_edges - l_edges
                    
                    wetted_width = np.nansum(widths)
                
                    
                    if wetted_width > 30:
                        
                        widths = widths[widths>=30]
                        thread_widths = np.append(thread_widths, widths)

                        if len(widths)>0:
                            quantiles = np.nanquantile(widths, np.arange (.05, .96, .05))
                            descrip = np.array([np.nanmin(widths), np.nanmax(widths), np.nanmean(widths), np.nanstd(widths)])
                            
                            ebi = -1 * np.nansum((widths / wetted_width) * np.log2(widths / wetted_width))
                            ebi = [2 ** ebi]
                            
                            dataarray = np.concatenate((descrip, quantiles, ebi, [wetted_width], [len(widths)]), axis = 0)
                            transect_df.loc[idx, :] = dataarray
                            # print(widths)
                    else:
                        continue
                    # print(year, transect_df)
                    
                    # transect_ebi.loc[idx, 'ebi'] = ebi
                    # transect_ebi.loc[idx, 'wetted_width'] = wetted_width
            
            # ax[a].plot(transect_ebi.index.values, transect_ebi['ebi'])
            # ax[a].set_title(f'{river} river, {year}')
            # ax[a].set_xlabel('transect number')
            # ax[a].set_ylabel('ebi')
            
            transect_df.to_csv(os.path.join(results_base, river, f'form-master_{year}.csv'))
        except Exception as e:
            print(f"Error processing year {year} for river {river}: {e}")
            error_years.append(year)
    np.savetxt(f'/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated/000_threadwidths/{river}_threadwidths.txt', thread_widths) 

    # plt.savefig(os.path.join(results_base, river, f'{river}_ebi.png'))

    if error_years:
        print(f"Errors occurred for river {river} in years: {error_years}")
# Main script
if __name__ == '__main__':
    all_river_tiffs = '/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
    all_rivers = os.listdir(all_river_tiffs)
    exclude_list = ['brahmaputra_yangcun', '.DS_Store', 'congo_new', 'agubh2', 'missing_colville',
                    'brahmaputra_pandu', 'brahmaputra_pandu_allyr_2012_bad', 'satellite_imagery.zip']
    all_rivers = [riv for riv in all_rivers if riv not in exclude_list]
    
    results_base = '/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated'
    years = np.arange(1999, 2024)

    with Pool() as pool:
        pool.starmap(process_river, [(river, results_base, all_river_tiffs, years) for river in all_rivers])

#%% PROCESSING LOG

# Error processing year 2001 for river southsask: zero-size array to reduction operation fmin which has no identity
# Errors occurred for river southsask in years: [np.int64(2001)]

# Error processing year 2019 for river rakaia: zero-size array to reduction operation fmin which has no identity
# Errors occurred for river rakaia in years: [np.int64(2019)]

# Error processing year 2006 for river bhareli_wide: zero-size array to reduction operation fmin which has no identity
# Error processing year 2007 for river bhareli_wide: zero-size array to reduction operation fmin which has no identity
# Error processing year 2015 for river bhareli_wide: zero-size array to reduction operation fmin which has no identity
# Errors occurred for river bhareli_wide in years: [np.int64(2006), np.int64(2007), np.int64(2015)]

# Error processing year 2018 for river lena: zero-size array to reduction operation fmin which has no identity
# Errors occurred for river lena in years: [np.int64(2018)]
