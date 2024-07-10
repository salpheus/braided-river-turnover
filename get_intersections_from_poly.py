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

# You might need to adjust this path based on your environment
# os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

river = 'betsiboka'
mask_files = glob.glob(f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{river}/mask/1999on/*.tif')

mask = mask_files[-1]
year = 2016
transect_path = (f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{river}/mask/1999on/betsiboka_meshlines.shp')

transect = gpd.read_file(transect_path)

raster = rasterio.open(mask)

# Ensure both are in the same CRS (reproject if necessary)
if transect.crs != raster.crs:
    lines_gdf = lines_gdf.to_crs(raster.crs)

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

# Analyze each linestring in the shapefile
for idx, row in transect.iterrows():
    linestring = row.geometry
    raster_values, points = extract_raster_values_along_line(linestring, raster)
    
    # Plot the terrain profile
    distances = np.linspace(0, linestring.length, num=len(raster_values))
    plt.figure()
    plt.plot(distances, raster_values)
    plt.title(f'Terrain Profile for Line {idx}')
    plt.xlabel('Distance along line (m)')
    plt.ylabel('Elevation (m)')
    plt.show()

    # Here you can perform further analysis on `raster_values` if needed
