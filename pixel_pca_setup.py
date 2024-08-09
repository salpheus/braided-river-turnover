#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:37:44 2024
Code to collate all pixels and attributes to perform megaPCA on turnover data

@author: safiya
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os

ndvi_path = (f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/ndvi_stacks_withwater/*.npy') ## path to ndvi arrays
turnstats_path = (f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt_bulkstats/*.nc') ## path to the turnover statistics
nturns_path = (f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats')
inventory = pd.read_excel('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', index_col=0)
# pull main static attributes
pca_params = ['mean_annu_qw_sc','bed-ssc_qw_m3yr', 'part_size_mm', 'bed_prop_of_total',
              'Tm_timescale', 'Tr_timescale', 'terrain_slope',
              'med_ebi', 'r_len_shape', 'r_ebi_shape', 'r_nturn_shape', 'r_meantt_shape',
              'mean_tt_length', 'max_tt_mean', 'num_turns_mean',
              'mean_slope', 'mean_arid_idx_catch',
              '2yr/mean', 'ndvi_med']
inventory = inventory.loc[:, pca_params]
# set up df to store pixels
mega_df = pd.DataFrame()

#%% import data and add to dataframe
## get image of average ndvi
for arr, nc in zip(glob.glob(ndvi_path), glob.glob(turnstats_path)[1:]): ## leaving agubh2 out
    
    rivname = nc.split('/')[-1].split('.nc')[0]

    avg_ndvi = np.nanmean(np.load(arr), axis = 0)
    nturns = xr.open_dataset(os.path.join(nturns_path, f'{rivname}_masks_stats.nc'))
    stats = xr.open_dataset(nc)
    
    mask_idx = np.where(~np.isnan(stats.meantime))
    
    ndvi = avg_ndvi[mask_idx].ravel()
    ## only using median and mean full arrays bc theres different amounts of data in each
    mean = stats.meantime.values[~stats.meantime.isnull().values].ravel()
    # meanw4 = stats.meantime_wetfor.values[~stats.meantime_wetfor.isnull().values].ravel()
    # meand4 = stats.meantimed_dryfor.values[~stats.meantimed_dryfor.isnull().values].ravel()
    
    med = stats.medtime.values[~stats.medtime.isnull().values].ravel()
    # medw4 = stats.medtime_w4.values[~stats.medtime_w4.isnull().values].ravel()
    # medd4 = stats.medtimed_w4.values[~stats.medtimed_w4.isnull().values].ravel()
    
    numturns = nturns.numturns.values[nturns.numturns.values != 0].ravel()
    rividx = np.repeat([rivname], len(numturns))
    
    df = pd.DataFrame({'river': rividx, 
                       'ndvi': ndvi, 
                       'meantt': mean, 
                       'medtt': med, 
                       'nturns': numturns})
    df.to_csv(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/megadf_pca/{rivname}.csv')

#%% ## load all csvs and merge into a megamerge

csv_paths = glob.glob(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/megadf_pca/*.csv')

merged_df = pd.read_csv(csv_paths[0])

for file in csv_paths[1:]:
    merged_df = pd.concat((merged_df, pd.read_csv(file)), axis = 0, ignore_index = True)

merged_df.to_csv(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/megadf_pca/megamerge_pca.csv')















