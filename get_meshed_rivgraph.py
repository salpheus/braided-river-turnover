#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:47:22 2024

@author: safiya
"""
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from rivgraph.classes import river
import pandas as pd
# import geopandas as gpd
# import rasterio

#%% one river, one tiff

# results_folder = '/Volumes/SAF_Data/remote-data/rivgraph_centerlines/test-river'
# rivname = 'betsiboka'
# riv_folder = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/'
# es = 'SN'
# tif = 'betsiboka_2023_01_01_2023_12_31_mask.tif'


# riv_test = river(rivname, os.path.join(riv_folder, rivname, 'mask/1999on', tif), exit_sides = es, verbose = True)
# linestrings = riv_test.compute_mesh()

# riv_test.to_geovectors('mesh', ftype = 'shp')

all_river_tiffs = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
all_rivers = os.listdir(all_river_tiffs)
all_rivers.remove('brahmaputra_yangcun')
all_rivers.remove('.DS_Store')
all_rivers.remove('congo_new')
all_rivers.remove('agubh2')

results_base = '/Volumes/SAF_Data/remote-data/rivgraph_centerlines'

exits = pd.read_csv(os.path.join(results_base, 'exit_sides.csv'), index_col = 0, header = 0)

for riv in all_rivers:
# riv = all_rivers[0] 
    print(riv)
    es = exits.loc[riv, 'es'] ## define the exits
    
    riv_base = os.path.join(results_base, riv)
    if not os.path.exists(riv_base):
        os.makedirs(riv_base)
    
    all_masks = glob.glob(os.path.join(all_river_tiffs, riv, 'mask/1999on/*.tif'))   
    years = np.arange(1999, 1999+len(all_masks))  ### create a list of years that we will make output folders for dependig on the number of masks
    
    for f, yr in enumerate(years):
        results_folder = os.path.join(riv_base, str(yr))
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        rivmap = river(riv, all_masks[f], results_folder, exit_sides = es, verbose = False) ##instantiate the river
        # rivmap = river(riv, tif, results_folder, exit_sides = es, verbose = True) ##instantiate the river
        # plt.imshow(rivmap.Imask)
        rivmap.compute_network()
        rivmap.prune_network()    
        rivmap.compute_mesh(smoothing = 0.25)
        rivmap.to_geovectors('mesh', ftype = 'shp')
        rivmap.to_geovectors('centerline', ftype = 'json')
        


