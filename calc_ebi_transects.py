#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:08:39 2024

Load geojsons created using rivgraph for a maask and compute ebi???
@author: safiya
"""
import numpy as np
import pandas as pd
import json
import os
import xarray as xr
import glob
import matplotlib.pyplot as plt
# import geopandas as gpd

import numpy as np
# from skimage.draw import line

all_river_tiffs = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
all_rivers = os.listdir(all_river_tiffs)
all_rivers.remove('brahmaputra_yangcun')
all_rivers.remove('.DS_Store')
all_rivers.remove('congo_new')
all_rivers.remove('agubh2')

results_base = '/Volumes/SAF_Data/remote-data/rivgraph_centerlines'

## for riv in all_rivers:
riv = all_rivers[1]    
riv_base = os.path.join(results_base, riv)

all_masks = glob.glob(os.path.join(all_river_tiffs, riv, 'mask/1999on/*.tif'))   
years = np.arange(1999, 1999+len(all_masks))  ### create a list of years that we will make output folders for dependig on the number of masks

for f, yr in enumerate(years):
    
    
    results_folder = os.path.join(riv_base, str(yr))
    
    linef = open(os.path.join(results_folder, f'{riv}_meshlines.json'))
    linestrings = json.load(linef)    
    shapefile = gpd.read_file(os.path.join(results_folder, f'{riv}_meshlines.json'))
    transects = np.empty([len(linestrings['features']), 4])  ### create umpy array to store coordinates then convert to df
    
    for i, feature in enumerate(linestrings['features']):
        coordinates = feature['geometry']['coordinates']
        
        transects[i, 0] = coordinates[0][0]
        transects[i, 1] = coordinates[0][1]
        transects[i, 2] = coordinates[1][0]
        transects[i, 3] = coordinates[0][1]
        
    transects_df = pd.DataFrame(transects, columns = ['startx', 'starty', 'endx', 'endy'])
    transects_df.to_csv(os.path.join(results_folder, 'centerline_transects.csv'))
    
    #%%
plt.figure(figsize = (10, 4), dpi = 300)
transects = pd.read_csv('/Volumes/SAF_Data/remote-data/rivgraph_centerlines/amudaryanew/2023/centerline_transects.csv', header = None).to_numpy()
    
for i in range(len(transects)):
    plt.plot([transects[i, 0], transects[i, 2]], [transects[i, 1], transects[i, 3]])
# plt.imshow(all_masks[-1])    

def get_line_pixels(transect_row):
    rr, cc = line(transect_row[1], transect_row[0], transect_row[1], transect_row[0])
    return list(zip(rr, cc))

# Example usage
start_point = (50, 50)
end_point = (300, 300)

line_pixels = get_line_pixels(transects[0, :].astype(int))
print("Line Pixels:", line_pixels)



#%% 

ds = xr.load_dataset('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc/amudaryanew_masks.nc')
ds2023 = ds.sel(year = 2023)
fig, ax = plt.subplots(figsize = (10, 5), dpi = 300)
# xr.plot.imshow(ds2023.masks, x = 'lon', y = 'lat', ax = ax)

for i in range(1, len(transects)):
    # ax.plot([transects[i, 1], transects[i, 3]], [transects[i, 2], transects[i,4]])
    ax.scatter(transects[i, 1], transects[i, 2])


