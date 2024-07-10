#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:43:34 2024
Run rivgraph centerline extraction in arallel
using the average channel width as the grid spacing this will be smll for some rivers bc of thread width distribution

@author: safiya
"""

import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rivgraph.classes import river

all_river_tiffs = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
all_rivers = os.listdir(all_river_tiffs)
exclude_list = ['brahmaputra_yangcun', '.DS_Store', 'congo_new', 'agubh2']
all_rivers = [riv for riv in all_rivers if riv not in exclude_list]

results_base = '/Volumes/SAF_Data/remote-data/rivgraph_centerlines'
exits = pd.read_csv(os.path.join(results_base, 'exit_sides.csv'), index_col=0, header=0)

def process_river(riv):
    print(riv)
    es = exits.loc[riv, 'es']  # define the exits
    
    riv_base = os.path.join(results_base, riv)
    if not os.path.exists(riv_base):
        os.makedirs(riv_base)
    
    all_masks = glob.glob(os.path.join(all_river_tiffs, riv, 'mask/1999on/*.tif'))
    years = np.arange(1999, 1999 + len(all_masks))  # create a list of years for output folders
    
    for f, yr in enumerate(years):
        results_folder = os.path.join(riv_base, str(yr))
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        
        rivmap = river(riv, all_masks[f], results_folder, exit_sides=es, verbose=False)  # instantiate the river
        # rivmap = river(riv, tif, results_folder, exit_sides = es, verbose = True) ##instantiate the river
        # plt.imshow(rivmap.Imask)
        rivmap.compute_network()
        rivmap.prune_network()
        rivmap.compute_mesh(smoothing=0.25)
        rivmap.to_geovectors('mesh', ftype='shp')
        rivmap.to_geovectors('centerline', ftype='json')

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(process_river, all_rivers)
