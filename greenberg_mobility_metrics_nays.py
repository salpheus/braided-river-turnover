#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:07:37 2024

@author: safiya
"""
import os
from mobility import get_mobility_rivers
from gif import get_stats
from gif import get_mobility
import glob
import numpy as np
import copy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import natsort
import os 
import rasterio
#%% have to change scale in get_mobility_yearly to 1 because these pixels are 1m

base = f'/Volumes/SAF_Data/remote-data/watermasks/agubh2_tifs/mask_nonan'
rivers = 'mask_nonan'

#%%
# rivers.remove('.DS_Store')
def get_paths_nays(root):
    # Get the rivers
    fps = glob.glob(os.path.join(root, '*.tif'))
    out_paths = {root: fps}

    return out_paths

# for river in rivers:
paths = get_paths_nays(base)

poly = "/Volumes/SAF_Data/remote-data/watermasks/gpkgs_fnl/agubh2_test.gpkg"

mobility_csvs = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/agubh2/mobcsvs'
out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/agubh2' # blocked fit_stats outputs
fitout = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/agubh2/fit_stats' # blocked fit_stats outputs
mobility_out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/agubh2/mobility'
figout = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/agubh2/figs'
#uncomment line 95 if you want to recalculate mobility stats
# get_mobility_rivers(poly, paths, 'mask_nonan') ## gets mobility yearly, makes a csv in the root folder of mobility data

#%%

# ds = rasterio.open('/Volumes/SAF_Data/remote-data/watermasks/agubh2_tifs/agubh2/agubh2_92_modelts.tif')
stop = 1 ## used to be 30

allcsvs = glob.glob(os.path.join(mobility_csvs, '*.csv')) 
path_list = natsort.natsorted(allcsvs)

''' If you uncomment these lines to redo the calculations change the estimates in the fit_curve lines of gif.py to be the mean of the area data here 
i used [0.001 and 1e13] for all area calculations so aw and fR    '''
# get_stats(os.path.join(mobility_csvs, f'{rivers[0]}_yearly_mobility.csv'), rivers[0], fitout, 30)
# get_mobility(pd.read_csv(os.path.join(fitout, f'{rivers[0]}_fit_stats.csv'), header = 0),
#                   os.path.join(mobility_out, f'{rivers[0]}_mobility.csv')) 

#%%

## create raw xy data for eq4
# out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/agubh2/mobcsvs'

def extract_location(name):
    parts = name.split('_')
    if len(parts) > 2 and parts[-1] == 'mobility.csv':
        return '_'.join(parts[:-2])
    
## build a full dataset of noramlised and non-normalised mobility metrics in the workspace
full_dataset = pd.DataFrame(columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
full_dataset_wick = pd.DataFrame(columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
for file in glob.glob(os.path.join(mobility_csvs, '*.csv')):
    riv = file.split("/")[-1]
    riv = extract_location(riv)

    df = pd.read_csv(file, index_col = 0, header = 0)
    ivals = df['i'].unique()
    
    for rng in ivals:
        med = np.median(df['O_avg'][df['i']==rng])/1e6
        avgs = np.mean(df['O_avg'][df['i']==rng])/1e6
        q1 = np.quantile((df['O_avg'][df['i']==rng])/1e6, .25)
        q3 = np.quantile((df['O_avg'][df['i']==rng])/1e6, .75)
        
        
        df_river = pd.DataFrame([[rng, 
                                  riv, 
                                 ((df['O_avg'][df['i']==rng])/1e6).min(),
                                 med,
                                 avgs,
                                 ((df['O_avg'][df['i']==rng])/1e6).max(),
                                 q1,
                                 q3]], columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
        
        med = np.median(df['O_wick'][df['i']==rng])
        avgs = np.mean(df['O_wick'][df['i']==rng])
        q1 = np.quantile((df['O_wick'][df['i']==rng]), .25)
        q3 = np.quantile((df['O_wick'][df['i']==rng]), .75)
        
        df_wick = pd.DataFrame([[rng, 
                                  riv, 
                                 ((df['O_wick'][df['i']==rng])).min(),
                                 med,
                                 avgs,
                                 ((df['O_wick'][df['i']==rng])).max(),
                                 q1,
                                 q3]], columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
                                 
        full_dataset = pd.concat((full_dataset, df_river), axis = 0, ignore_index = True)
        full_dataset_wick = pd.concat((full_dataset_wick, df_wick), axis = 0, ignore_index = True)
        
        
## pull floodplain reworking statistics

fr_dataset = pd.DataFrame(columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
fr_dataset_wick = pd.DataFrame(columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
for file in glob.glob(os.path.join(mobility_csvs, '*.csv')):
    riv = file.split("/")[-1]
    riv = extract_location(riv)

    df = pd.read_csv(file, index_col = 0, header = 0)
    ivals = df['i'].unique()
    
    for rng in ivals:
        med = np.median(df['fR'][df['i']==rng])/1e6
        avgs = np.mean(df['fR'][df['i']==rng])/1e6
        q1 = np.quantile((df['fR'][df['i']==rng])/1e6, .25)
        q3 = np.quantile((df['fR'][df['i']==rng])/1e6, .75)
        
        
        df_river = pd.DataFrame([[rng, 
                                  riv, 
                                 ((df['fR'][df['i']==rng])/1e6).min(),
                                 med,
                                 avgs,
                                 ((df['fR'][df['i']==rng])/1e6).max(),
                                 q1,
                                 q3]], columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
        
        med = np.median(df['fR_wick'][df['i']==rng])
        avgs = np.mean(df['fR_wick'][df['i']==rng])
        q1 = np.quantile((df['fR_wick'][df['i']==rng]), .25)
        q3 = np.quantile((df['fR_wick'][df['i']==rng]), .75)
        
        df_wick = pd.DataFrame([[rng, 
                                  riv, 
                                 ((df['fR_wick'][df['i']==rng])).min(),
                                 med,
                                 avgs,
                                 ((df['fR_wick'][df['i']==rng])).max(),
                                 q1,
                                 q3]], columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
                                 
        fr_dataset = pd.concat((fr_dataset, df_river), axis = 0, ignore_index = True)
        fr_dataset_wick = pd.concat((fr_dataset_wick, df_wick), axis = 0, ignore_index = True)


#%% make plots
rivers = ['mask_nonan', 'mask']
full_dataset = full_dataset.sort_values('tbase')
fr_dataset = fr_dataset.sort_values('tbase')
linear_reworking = pd.DataFrame(columns = rivers, index = ['Rct']) ##store fp constants
linear_ov_decay = pd.DataFrame(columns = rivers, index = ['Mct']) ##store reworking constant

params = ['Cm', 'Pm', 'Aw']
channel_params = pd.DataFrame(index = params, columns = rivers)

fr_params = ['Cr', 'Pr', 'Aw']
fp_params = pd.DataFrame(index = fr_params, columns = rivers)
## calculating 3e using the average across percentiles
for r, river in enumerate(rivers):
    plt.figure(figsize = (4, 3), dpi = 300, tight_layout = True) ##plot all overlap area data on one axis

    plt.plot(full_dataset[full_dataset['river']==river]['tbase'], 
             full_dataset[full_dataset['river']==river]['median'], lw = 0, ms = 5, label = river)
    
    plt.fill_between(full_dataset[full_dataset['river']==river]['tbase'].astype(int), 
                     full_dataset[full_dataset['river']==river]['Q1'], 
                     full_dataset[full_dataset['river']==river]['Q3'], alpha = 0.25, label = river)
    
    # load data for curve fitting and store it in the channel_params df
    fits = pd.read_csv(os.path.join(out, f'fit_stats/{river}_fit_stats.csv'), header = 0, index_col = 0 )
  
    ##pull median best fit parameters to draw the best fit curve
    aw50 = fits['Aw'][1]/1e6 #[km2, i px is 1m2 this is the interpolated dataset] median water surface area 
    cm50 = fits['CM'][1] #[1/T] # median channel overlap decay rate
    pm50 = fits['PM'][1]/1e6 #[km2] ## longterm channel area memory, median
    
    channel_params.loc[:, river] = [cm50, pm50, aw50]
    
    dt = np.arange(0, len(full_dataset[full_dataset['river']==river]))
    
    ami = ((aw50-pm50) * np.exp(-1*cm50*dt) + pm50)
    
    Mct = cm50*(1-(pm50/aw50))
    linear_ov_decay.loc['Mct', river] = Mct
    
    plt.plot(dt, ami, zorder = 10000)
    
    # plt.yscale('log')
    plt.ylabel('Channel overlap area, m2');
    plt.xlabel('Duration since baseline');
    plt.title(f'{river}, 3e = {np.round(3/cm50, 1)}, A$_m$* = {np.round(1-(pm50/aw50), 1)}')
    plt.savefig((f'{out}/figs/choverlap/{river}_choverlap.svg'))


for r, river in enumerate(rivers):
    plt.figure(figsize = (4, 3), dpi = 300, tight_layout = True) ##plot all overlap area data on one axis
    plt.plot(fr_dataset[fr_dataset['river']==river]['tbase'], 
             fr_dataset[fr_dataset['river']==river]['median'], lw = 0, ms = 5, label = river)
    
    plt.fill_between(fr_dataset[fr_dataset['river']==river]['tbase'].astype(int), 
                     fr_dataset[fr_dataset['river']==river]['Q1'], 
                     fr_dataset[fr_dataset['river']==river]['Q3'], alpha = 0.25, label = river)
    
    # load data for curve fitting and store it in the channel_params df
    fits = pd.read_csv(os.path.join(out, f'fit_stats/{river}_fit_stats.csv'), header = 0, index_col = 0 )
  
    ##pull median best fit parameters to draw the best fit curve
    aw50 = fits['Aw'][1]/1e6 #[m2]
    cr50 = fits['CR'][1] #[1/T] ### med. fp reworking growth rate somehow this is 001 and constant for all data. somehting is wrong
    pr50 = fits['PR'][1]/1e6 # est active fp area [m2]
    
    fp_params.loc[:, river] = [cr50, pr50, aw50]
    dt = np.arange(0, len(fr_dataset[fr_dataset['river']==river]))
    
    ari = (-1*pr50) * np.exp(-1*cr50*dt) + pr50
    
    Rct = cr50*pr50/aw50
    linear_reworking.loc['Rct', river] = Rct
    
    plt.plot(dt, ari, zorder = 10000)

    plt.yscale('log')
    plt.ylabel('Floodplain reworking, km2');
    plt.xlabel('Duration since baseline');
    plt.title(f'{river}, 3e = {np.round(3/cr50, 1)}, \n A$_r$* = {np.round(pr50/aw50, 1)}')
    plt.savefig(f'{out}/figs/fprework/{river}_fprework.svg')


