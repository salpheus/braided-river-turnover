#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:44:59 2025


code to make time series of where 1s and 0s are in each river

Actually this code is to compute the amount of turnover but including the zeros, but within the channel corridor

@author: safiya
"""
import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib as mpl
import numpy.ma as ma
import matplotlib.pyplot as plt
import glob as glob
import copy
import scipy.stats as stats
#%% batch load xr and export 25 histogram of 1s and 0s

list_xrs = glob.glob('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt-flags/*.nc')
list_filled_polys = glob.glob('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/holes_filled_polys/*.npy')
years = np.arange(1999, 2025)
turnover_ts = pd.DataFrame(index = years)
bins = np.arange(0, 25)
turnover_numbers = pd.DataFrame(index = np.arange(0, 24))
turnover_props = pd.DataFrame(index = np.arange(0, 24))

#%% for loop

for turn, poly in zip(list_xrs, list_filled_polys):
    # fix yukon if errors
    # turn = list_xrs[-1]
    # poly = list_filled_polys[-2]
    name = turn.split('/')[-1].split('_masks_full.nc')[0]
    if name == 'brahmaputra_pandu':
        continue
    print(name)
    
    river = abs(xr.load_dataset(turn).PTTFlags) ##load the flags, get the absolute values
    rivsum = river.sum(axis = 0) ## find the sum of the turnover events for each pixel
    
    divby = np.sum(np.load(poly).astype(int))
    
    rivsum_mask = rivsum.where(np.load(poly))
    
    turnover_numbers[name] = np.histogram(rivsum_mask, bins = bins)[0]
    turnover_props[name] = np.histogram(rivsum_mask, bins = bins)[0]/divby


#     test = xr.open_dataset(x).PTTFlags
#     test_sums = np.squeeze(np.apply_over_axes(np.sum, np.abs(test), [1, 2]), axis = 2)
    
#     if name == 'agubh2':
#         df = pd.DataFrame(test_sums, columns = [name], index = years)
#     else: 
#         df = pd.DataFrame(test_sums, columns = [name], index = years[:-1])
#     turnover_ts = pd.concat((turnover_ts, df), axis = 1)

#%%

for i in turnover_numbers.columns:
    data = np.repeat(np.arange(0, 24), turnover_numbers[i])
    np.savetxt(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/{i}_turnoverfreq_0s.csv', data, delimiter = ',')

# turnover_ts_norm = turnover_ts/turnover_ts.sum(axis = 0)
# turnover_numbers.to_csv('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/turnover_including_0s_corridor.csv')
# turnover_props.to_csv('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/turnover_props_including_0s_corridor.csv')
#%%

## make clusetring groups just to visualise the histograms
pca_style1 = ['congo_destriped', 'tanana', 'yukon', 'lena', 'ob_down', 'ob_up', 'kasai_destriped']
pca_style2 = ['rakaia', 'agubh2', 'colville', 'yukon_eagle', 'southsask']
pca_style3 = ['indus_r2', 'brahmaputra_pandu_allyr', 'amudaryadown', 'amudaryanew', 'irrawaddy_up', 'irrawaddy', 'bhareli_wide', 'mangoky', 'betsiboka']


vis_style1 = ['congo_destriped',  'yukon', 'lena', 'ob_down', 'ob_up', 'yukon_eagle'] ## most blue/purple
vis_style2 = ['rakaia', 'agubh2', 'betsiboka', 'mangoky' , 'southsask'] ## most yellow
vis_style3 = ['indus_r2', 'brahmaputra_pandu_allyr','tanana', 'amudaryadown', 'amudaryanew', 'kasai_destriped', 'irrawaddy_up', 'irrawaddy', 'bhareli_wide'] ## mix


#%%

fig, ax = plt.subplots(2, 3, figsize = (12, 10), dpi = 300, sharex = True, sharey = True)
ax = ax.ravel()
lw = 3


for riv in turnover_props.columns:
    if riv in pca_style1:
        ax[0].plot(turnover_props[riv], label = riv, lw = lw, c = 'xkcd:kelly green', alpha = 0.4)
    elif riv in pca_style2:
        ax[1].plot(turnover_props[riv], label = riv, lw = lw, c = 'xkcd:light brown', alpha = 0.4)
    elif riv in pca_style3: 
        ax[2].plot(turnover_props[riv], label = riv, lw = lw, c = 'xkcd:dark orange', alpha = 0.4)
        
for riv in turnover_props.columns:
    if riv in vis_style1:
        ax[3].plot(turnover_props[riv], label = riv, lw = lw, c = 'xkcd:barney purple', alpha = 0.4)
    elif riv in vis_style2:
        ax[4].plot(turnover_props[riv], label = riv, lw = lw, c = 'xkcd:golden yellow', alpha = 0.4)
    elif riv in vis_style3: 
        ax[5].plot(turnover_props[riv], label = riv, lw = lw, c = 'xkcd:barbie pink', alpha = 0.4)
        


ax[0].set_xlabel('Number of turnovers')
ax[1].set_xlabel('Number of turnovers')
ax[2].set_xlabel('Number of turnovers')
ax[0].set_ylabel('Count/Corridor Area')

ax[0].set_title('Vegetation & eBI')
ax[1].set_title('Slope & Bedload')
ax[2].set_title('uQw, DV')

ax[3].set_xlabel('Number of turnovers')
ax[4].set_xlabel('Number of turnovers')
ax[5].set_xlabel('Number of turnovers')
ax[3].set_ylabel('Count/Corridor Area')

ax[3].set_title('Vis 1: Less turnover than avg \n most exponential?')
ax[4].set_title('Vis 2: More turnover than avg \n most gamma?')
ax[5].set_title('Vis 3: Avg turnover behav \n in-between?')

#%% calculate shannon entropy of the distributions

abr_rivnames = ['MOD', 'ADD', 'ADU', 'BET', 'BHA', 'BRA', 'COL', 'CON', 'IND', 'IRA', 'IRU', 'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK', 'TAN', 'YUE', 'YUK']

entropies = stats.entropy(turnover_props, axis = 0, base = 2)
entropies_unlog = np.exp2(entropies)

plt.figure(dpi = 300, figsize = (10, 3))
plt.plot(entropies_unlog, lw = 0, marker = 'o')

ax = plt.gca()
ax.set_xticks(np.arange(0, len(entropies)), labels = abr_rivnames, rotation = 30);

ax.set_ylabel('$2^{entropy}$')


ax2 = ax.twinx()
ax2.plot(np.arange(0, len(entropies)), turnover_props)

#%% read distribution statistics for the turnover






























