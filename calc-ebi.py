#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:51:19 2024
 
Code to use rivgraph to find centerlines in an image and make transects along which you'll calculate ebi???'
@author: safiya
"""
from rivgraph.classes import river
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import xarray as xr
import numpy as np
#%%

## define exit sides
exits = pd.read_csv('/Volumes/SAF_Data/remote-data/rivgraph_centerlines/exit_sides.csv', index_col = 0, header = 0)

# define river
rivname = 'southsask'

#%%

##define netcdf path
# ncpath = f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc/{rivname}_masks.nc'
# ncfile = xr.load_dataset(ncpath)
## define path to mask
mask_root = ('/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may')
mask_paths = glob.glob(os.path.join(mask_root, rivname, 'mask/1999on', '*.tif'))

## define results folder
results_root = ('/Volumes/SAF_Data/remote-data/rivgraph_centerlines')
results_folder = os.path.join(results_root, rivname)

if not os.path.exists(results_folder):
    os.mkdir(results_folder)
    
#%% instantiate the river class and pull files
    
work_river = river(rivname, 
                   mask_paths[4], 
                   results_folder, 
                   exit_sides = exits.loc[rivname, 'es'])## name for the river itemto be instantiated bc the name of the class is river smfh
#%% do intermediate steps: view mask, skeletonize, compute network, prune 
plt.figure(dpi = 300)
plt.imshow(work_river.Imask)

work_river.skeletonize()
plt.figure(dpi = 300)
plt.imshow(work_river.Iskel)


# save geotiff idk why but lets see
work_river.to_geotiff('skeleton')

work_river.compute_network()
work_river.prune_network()

work_river.compute_link_width_and_length()

centerline = work_river.centerline

#%% define mesh parameters and calculate mesh
smooth = 0.25
work_river.compute_mesh(smoothing = smooth)

# plt.hist(work_river.links['len_adj'], bins = 50)
meshline_dem = work_river.meshlines
## see centerline
plt.figure(dpi = 300)
ax = plt.gca() 
xr.plot.imshow(ncfile.masks[0], ax = ax, figsize = (10, 10), aspect = 'equal')
ax.plot(centerline[0], centerline[1], 'r--', lw = 2) 

for ln in meshline_dem:
    lines = np.array(ln)
    ax.plot((lines[0, 0], lines[1, 0]), (lines[0, 1], lines[1, 1]), 'b-', lw = 0.5)
# ax = plt.gca()
# ax.set_aspect('equal')