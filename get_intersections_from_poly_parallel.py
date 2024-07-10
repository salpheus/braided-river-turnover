#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:37:41 2024
Trying to use to extract intersections form tiffs, not working, bu this is wehre I am https://py.geocompx.org/05-raster-vector
@author: safiya
"""
import os
# import fiona
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
# import rasterstats
#%%
# Function to extract raster values along a linestring
def extract_raster_values_along_line(linestring, raster):
    # Generate points along the line at regular intervals
    num_points = int(linestring.length)  # or adjust based on your need
    points = [linestring.interpolate(float(i) / num_points, normalized=True) for i in range(num_points + 1)]
    
    # Extract raster values at the points
    raster_values = []
    for point in points:
        row, col = raster.index(point.x, point.y)
        raster_values.append(raster.read(1)[row, col])
        
    return raster_values, points



## import river names
all_river_tiffs = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
all_rivers = os.listdir(all_river_tiffs)
exclude_list = ['brahmaputra_yangcun', '.DS_Store', 'congo_new', 'agubh2']
all_rivers = [riv for riv in all_rivers if riv not in exclude_list]

results_base = '/Volumes/SAF_Data/remote-data/rivgraph_centerlines'

years = np.arange(1999, 2024) ## we already have ebi data for agubh2

for river in all_rivers:
    mask_files = glob.glob(f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{river}/mask/1999on/*.tif')
    centerline_fol = os.path.join(results_base, river)
    
    for year in years:
        transect_path = os.path.join(centerline_fol, str(year), f'{river}_meshlines.shp')
        transect = gpd.read_file(transect_path)
        
        raster_path = glob.glob(os.path.join(all_river_tiffs, river, 'mask/1999on', f'*{year}*.tif').format(year = year)) ## pull the right tif
        raster = rasterio.open(raster_path[0])
        
        # Ensure both are in the same CRS (reproject if necessary)
        if transect.crs != raster.crs:
            transect = transect.to_crs(raster.crs)

        transect_ebi = pd.DataFrame(columns = ['ebi', 'wetted_width'], index = transect['FID'])
        # Analyze each linestring in the shapefile, calculating the ebi
        for idx, row in transect.iterrows():
            linestring = row.geometry
            raster_values, points = extract_raster_values_along_line(linestring, raster)
            
            raster_values = np.array(raster_values, dtype = int)
            
            edges = np.diff(raster_values)
            l_edges = np.where(edges==1)[0]
            r_edges = np.where(edges==-1)[0]
            
            widths = r_edges-l_edges
            
            wetted_width = np.nansum(widths) ##find the discharge across the section
            
            ebi = -1* np.nansum((widths/wetted_width)*np.log2(widths/wetted_width))
            ebi = 2**ebi
            transect_ebi.loc[idx, 'ebi'] = ebi
            transect_ebi.loc[idx, 'wetted_width'] = wetted_width
            
        transect_ebi.to_csv(os.path.join(results_base, river, f'ebi_ww_{year}.csv'))
