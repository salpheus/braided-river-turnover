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
from scipy.optimize import curve_fit
import matplotlib as mpl
import copy
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 8}

mpl.rc('font', **font)


#%% create water palette

# mstack_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc/'
save_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/water_palette/'
# maskstacks = glob.glob(os.path.join(mstack_path, '*.nc'))
# names = [os.path.basename(x).split('_masks.nc')[0] for x in glob.glob(os.path.join(mstack_path, '*.nc'))]

# water_stats = pd.DataFrame(columns = np.arange(1999, 2025), index = names)
# maskstacks = maskstacks[-2]
# for maskstack in maskstacks:
#     rivname = maskstack.split('/')[-1].split('_masks.nc')[0] 
    
#     masks = xr.load_dataset(maskstack).masks.to_numpy()
    
#     water_palette = copy.deepcopy(masks[0, :, :])
    
#     water_palette[water_palette == 0] = -999 
#     water_palette[water_palette==1] = 0

#     for ts in range(1, len(masks)):
#           mix = water_palette + masks[ts, :, :] 
#           water_palette[mix==-998] = ts 
         
#     ## get water_palette statistics for dataframe
#     wpal_unique, wpal_counts = np.unique(water_palette[water_palette>=0], return_counts = True)
#     total_wet = np.sum(wpal_counts)
#     cumsum_wet = np.cumsum(wpal_counts)
#     prop_wetted = cumsum_wet/total_wet
    
#     if rivname != 'agubh2':
#         prop_wetted = np.concatenate((prop_wetted, [np.nan]))
#     water_stats.loc[rivname, :] = prop_wetted        
        
#     np.save(os.path.join(save_path, f'{rivname}_wpal.npy'), water_palette)
# water_stats = water_stats.dropna(axis = 0, how = 'all')
# water_stats.to_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/prop_to_wet.csv')    
    
palettes = glob.glob(os.path.join(save_path, '*.npy'))
# water_stats = pd.DataFrame(columns = np.arange(1999, 2025), index = water_stats.index)


# for palette in palettes:
#     rivname = palette.split('/')[-1].split('_wpal.npy')[0]
    
#     rivpal = np.load(palette)
#     wpal_unique, wpal_counts = np.unique(rivpal[rivpal>=0], return_counts = True)
    
#     if rivname == 'congo_destriped':
#         wpal_counts = np.concatenate((wpal_counts, [np.nan], [np.nan]))
    
#     elif rivname != 'agubh2':
#         wpal_counts = np.concatenate((wpal_counts, [np.nan]))
#     water_stats.loc[rivname, :] = wpal_counts
    
# water_stats.to_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/sum_to_wet.csv')        

## this csv file is the prop turned wet per year - total wettable area (including 1999)
to_wet = pd.read_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/totwet-wet_per_yr_2000on.csv', 
                     header = 0, index_col = 0)  

plt.figure(figsize = (10, 10), tight_layout = True, dpi = 300)    

for river in to_wet.index.values:
    
    wet_props = np.cumsum(to_wet.loc[river, :])
    plt.plot(np.arange(0, len(wet_props)), wet_props, label = river, marker = '.', lw = 1)
    
xvals = np.arange(1999, 2025)
lines = plt.gca().get_lines()
labelLines(lines, xvals = xvals, align = True);
    
plt.xlabel('Years since first timestep')
plt.ylabel('Proportion of total wetted area converted since first timestep');
    
#%%### get exponents using LS regresion as another peremater

def func(x, A, B, C):
    return A * np.exp(-B * x) + C


# LS regression for curve y= Ae - Bx
fit_data = pd.DataFrame(columns = ['A', 'B', 'ct'], index = to_wet.index) ## for scipy curve fit parameters
rivlist = fit_data.index.values

# riv = rivlist
# riv = 'brahmaputra_pandu'

# obs_x = np.arange(0, len(to_wet.columns))
# obs_y = np.cumsum(to_wet.loc[riv, :]).to_numpy()

## trying inversion from inverse class but i think im doing it wrong. 
# A_exp = np.ones([len(obs_y), 3]) ## 1, x term and exponent x term
# A_exp[:, 1] = obs_x
# A_exp[:, 2] = np.exp(-obs_x)

# xls = np.linalg.inv(np.transpose(A_exp)@A_exp) @ np.transpose(A_exp)@obs_y ## least squares parameter solutions

# yls = xls[1] * np.exp(-xls_[2]*obs_x) + xls[0]

# yls = popt[0]*np.exp(-popt[1]*obs_x) + popt[2]

fig, ax = plt.subplots(3, 7, dpi = 300, tight_layout = True, sharex = True, figsize = (20, 7))
ax = ax.ravel()

for a, riv in enumerate(rivlist):
    obs_x = np.arange(0, len(to_wet.columns))
    obs_y = np.cumsum(to_wet.loc[riv, :]).to_numpy()

    popt, pcov = curve_fit(func, obs_x, obs_y) 
    fit_data.loc[riv, :] = popt    
    
    
    ax[a].plot(obs_x, obs_y, lw = 0, marker = '.')
    ax[a].plot(obs_x, func(obs_x, *popt), 'r--')

    ax[a].set_ylabel('Prop of total area gone wet')
    ax[a].set_xlabel('Year')
    
    ax[a].set_title(f'{riv}, ABct = {np.round(popt, 2)}')

        
    
    
fit_data_ln = copy.deepcopy(fit_data)
fit_data['A'] = np.log(fit_data['A'])















    