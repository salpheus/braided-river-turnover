#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:56:55 2024
get time to wet maps and cdfs for all rivers
@author: safiya
"""
import numpy as np
import pandas as pd
import copy
import xarray as xr
from labellines import labelLines
import os 
import glob
import matplotlib.pyplot as plt

#%% create water palette

mstack_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc/'
save_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/water_palette/'
maskstacks = glob.glob(os.path.join(mstack_path, '*.nc'))
names = [os.path.basename(x).split('_masks.nc')[0] for x in glob.glob(os.path.join(mstack_path, '*.nc'))]

water_stats = pd.DataFrame(columns = np.arange(1999, 2025), index = names)
maskstacks = maskstacks[-2]
for maskstack in maskstacks:
    rivname = maskstack.split('/')[-1].split('_masks.nc')[0] 
    
    masks = xr.load_dataset(maskstack).masks.to_numpy()
    
    water_palette = copy.deepcopy(masks[0, :, :])
    
    water_palette[water_palette == 0] = -999 
    water_palette[water_palette==1] = 0

    for ts in range(1, len(masks)):
          mix = water_palette + masks[ts, :, :] 
          water_palette[mix==-998] = ts 
         
    ## get water_palette statistics for dataframe
    wpal_unique, wpal_counts = np.unique(water_palette[water_palette>=0], return_counts = True)
    total_wet = np.sum(wpal_counts)
    cumsum_wet = np.cumsum(wpal_counts)
    prop_wetted = cumsum_wet/total_wet
    
    if rivname != 'agubh2':
        prop_wetted = np.concatenate((prop_wetted, [np.nan]))
    water_stats.loc[rivname, :] = prop_wetted        
        
    np.save(os.path.join(save_path, f'{rivname}_wpal.npy'), water_palette)
# water_stats = water_stats.dropna(axis = 0, how = 'all')
# water_stats.to_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/prop_to_wet.csv')    
    
plt.figure(figsize = (10, 10), tight_layout = True, dpi = 300)
for river in water_stats.index.values:
    plt.plot(water_stats.loc[river, :], label = river)
    
xvals = np.arange(1999, 2025)
lines = plt.gca().get_lines()
labelLines(lines, xvals = xvals, align = True);
    
plt.xlabel('Year')
plt.ylabel('Prop total wetted area converted')    
    
    
    
    
    
    