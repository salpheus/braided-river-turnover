#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:33:29 2024

@author: safiya
"""
import xarray as xr
import numpy as np
import pandas as pd
from pull_proportion_turnover import get_ncfiles

base_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt-flags'
files = get_ncfiles(base_path)

# turnfrequency = pd.DataFrame(index = np.arange(1, 26))

# for file in files: 
#     name = file.split('/')[-1].split('.')[0].split('_masks_full')[0]
    
#     riv = xr.load_dataset(file)
#     turnover_times = riv.PTT.to_numpy()
    
#     freq, edges = np.histogram(turnover_times, bins = np.arange(0, 26))
    
#     turnfrequency[name] = pd.DataFrame(freq)

## DO NOT SAVE UNLESS YOU ARE OVERWRITING ALL DATA    
# turnfrequency.to_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/length_turnover_frequencies.csv')    
