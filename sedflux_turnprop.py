#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:23:20 2024

@author: safiya
"""
import xarray as xr
import pandas as pd
import glob

#%% pull area turnover per time
pttpath = '/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt-flags/*.nc'
ncs = glob.glob(pttpath)

results_list = []

# Iterate over each file
for file in ncs:
    ds = xr.load_dataset(file)
    nm = file.split('/')[-1].split('.')[0]
    print(nm)
    # Iterate over each year
    for year in ds['year'].values:
        data = ds.PTTFlags.sel(year=year).values
        wentwet = data[data > 0].sum()
        wentdry = data[data < 0].sum()
        totalturn = (data**2).sum()  # Convert non-zero values to 1 and then sum
        
        # Append the results to the list
        results_list.append({
            'filename': nm,
            'year': year,
            'totalturn': totalturn,
            'turnwet': wentwet,
            'turndry': wentdry
        })

# Convert the list to a DataFrame
results = pd.DataFrame(results_list)
        
# Display the results DataFrame
print(results)


