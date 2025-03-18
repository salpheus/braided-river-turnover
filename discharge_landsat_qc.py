#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:55:12 2024
plot discharge and landsat coverage
@author: safiya
"""

import glob 
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)


qc_root = '/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/'
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
qw_root = '/Volumes/SAF_Data/SAF_Data/CHAPTER2/turnover_discharge_records/monthly_discharge_averages/'

#%% load dataframes and plot data

qccsvs = glob.glob(os.path.join(qc_root, '*_maskqc.csv'))
qcdfs = [pd.read_csv(csv) for csv in qccsvs]

qwcsvs = glob.glob(os.path.join(qw_root, '*.csv'))
qwdfs = [pd.read_csv(csv, index_col=0) for csv in qwcsvs]

#%%

rivlist = ['Amu Darya (down.)', 'Amu Darya (up.)', 'Betsiboka', 'Bhareli', 'Brahmaputra',
           'Colville', 'congo_lukolela_bolobo', 'Indus_r', 'Irrawaddy', 'Irrawaddy (up.)',
           'Kasai', 'Lena', 'Mangoky', 'Ob (down.)', 'Ob (up.)', 'Rakaia', 'South Saskatchewan',
           'Tanana', 'Yukon (@ Eagle)', 'Yukon (@ Circle)',]

## code to plot just one average imaging series per river 
## avergae plot per river
fig, ax = plt.subplots(4, 5, figsize = (18, 10), tight_layout = True, sharex = True, sharey = True, dpi = 300)
ax = ax.ravel()


for r, riv in enumerate(rivlist):
    qcdf = qcdfs[r] ## the qc data for one river
    qwdf = qwdfs[r] ## the qw data for one river
    qcdf['year'] = pd.DatetimeIndex(qcdf['date']).year
    qcdf['month'] = pd.DatetimeIndex(qcdf['date']).month
    
    qc_yr_avg = qcdf.groupby('month').mean(numeric_only = True)
    ax[r].yaxis.grid(True)
    # ax[r].axhline(0.5, 1, 13, c = 'r', ls = '--', lw = 0.5)
    ax[r].yaxis.set_major_locator(MultipleLocator(.25))

    ax[r].set_xticks(range(1, 13))
    ax[r].set_xticklabels(months, rotation=45)
    ax[r].set_ylabel('Avg prop img used')
    ax[r].scatter(qc_yr_avg.index, qc_yr_avg['perc_unmasked'], c='r', label='monthly Mean') 
    
    ax[r].set_title(riv)
    ax2 = ax[r].twinx()
    ax2.step(range(1, 13), qwdf['average'], where = 'mid')
    ax2.set_ylabel('Monthly avg discharge m3/s')
    

# for r, riv in enumerate(rivlist):
#     # min_coverage = pd.DataFrame(columns = np.arange(1, 13), index = np.arange(1999, 2024))  
#     # mean_coverage = pd.DataFrame(columns = np.arange(1, 13), index = np.arange(1999, 2024))  
#     # med_coverage = pd.DataFrame(columns = np.arange(1, 13), index = np.arange(1999, 2024))  
#     # max_coverage = pd.DataFrame(columns = np.arange(1, 13), index = np.arange(1999, 2024))  
#     qcdf = qcdfs[r]
#     qcdf['year'] = pd.DatetimeIndex(qcdf['date']).year
#     qcdf['month'] = pd.DatetimeIndex(qcdf['date']).month
    
#     qc_yr_avg = qcdf.groupby('month').mean(numeric_only = True)
    
#     qwdf = qwdfs[r]
    
    
#     ## avergae plot per river
#     fig, ax = plt.subplots(4, 5, figsize = (18, 14), tight_layout = True, sharex = True, sharey = True, dpi = 300)
#     ax = ax.ravel()
#     for a, riv in enumerate(rivlist):
#         ax[a].set_xticks(range(1, 13))
#         ax[a].set_xticklabels(months, rotation=45)
        
#         ax[a].plot(qc_yr_avg.index, qc_yr_avg['perc_unmasked'], c='r', label='monthly Mean') 
        
#         ax[a].set_title(riv)
#         ax2 = ax[a].twinx()
#         ax2.step(range(1, 13), qwdf['average'])
        
    
        
#     # fig, ax = plt.subplots(5, 5, figsize = (18, 10), tight_layout = True, sharey = True, dpi = 300)
#     # ax = ax.ravel()
#     # fig.suptitle(riv)
#     # for a, yr in enumerate(qcdf['year'].unique()):
#     #     # ax[a].scatter(qcdf['month'][qcdf['year']==yr],qcdf['perc_unmasked'][qcdf['year']==yr])
        
#     #     # Set x-ticks to be the numeric months (1 to 12)
#     #     ax[a].set_xticks(range(1, 13))
#     #     # Set the labels to be the month names
#     #     ax[a].set_xticklabels(months, rotation=45)

        
#     #     ax[a].set_title(yr)
#     #     ax[a].set_ylabel('% mask used')
#     #     ax[a].set_xticklabels(months, rotation=45);
    
    
#     #     means = qcdf[qcdf['year']==yr].groupby(['month'])['perc_unmasked'].mean()
#     #     maxs = qcdf[qcdf['year']==yr].groupby(['month'])['perc_unmasked'].max()
#     #     mins = qcdf[qcdf['year']==yr].groupby(['month'])['perc_unmasked'].min()
#     #     meds = qcdf[qcdf['year']==yr].groupby(['month'])['perc_unmasked'].median()
        
#     #     mean_coverage.loc[yr, means.index.values] = np.round(means.values, 3)*100
#     #     med_coverage.loc[yr, meds.index.values] = np.round(meds.values, 3)*100
#     #     max_coverage.loc[yr, maxs.index.values] = np.round(maxs.values, 3)*100
#     #     min_coverage.loc[yr, mins.index.values] = np.round(mins.values, 3)*100
        
#     #     ax[a].plot(means.index, means, c='r', label='Mean')
#     #     # ax[a].plot(maxs.index, maxs, c='k', linestyle='--', label='Max')
#     #     # ax[a].plot(mins.index, mins, c='b', linestyle='--', label='Min')
        
#     #     ax2 = ax[a].twinx()
    
#     #     ax2.step(range(1, 13), qwdf['average'])
        
#     #     # ax2 = ax[a].twinx()
#     #     # # Use only the available months for qwdf['average'], assuming it's aligned by month index
#     #     # ax2.step(range(1, 13), qwdf['average'][:12], label='QW Data')  # [:12] just to ensure 12 months plotted
        
#     # plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/perc_unmasked/{riv}_maskstats.png')

#     # med_coverage.to_csv(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/perc_unmasked/medians/{riv}_medians.csv')
#     # mean_coverage.to_csv(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/perc_unmasked/means/{riv}_means.csv')
#     # max_coverage.to_csv(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/perc_unmasked/maxs/{riv}_maxs.csv')
#     # min_coverage.to_csv(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/perc_unmasked/mins/{riv}_mins.csv')
