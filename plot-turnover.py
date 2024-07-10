#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:38:03 2024

@author: safiya
"""
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl
import numpy.ma as ma
import os
import pandas as pd
import glob
import seaborn as sns
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)

#%% load the data
abr_rivnames = ['AGU', 'ADD', 'ADN', 'BET', 'BHA', 'BHP', 'COL', 'CON', 'IND', 'IRA', 'IRU', 'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK', 'TAN', 'YUE', 'YUK']

stats_path = f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/'
# test = xr.load_dataset(os.path.join(stats_path, 'turnstats', 'congo_destriped_masks_stats.nc'))
inventory = pd.read_excel('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx')

inventory_alphabetical = inventory.sort_values(by = 'river', axis = 0, ascending = True)
flierprops = dict(marker='o', markerfacecolor='xkcd:gray', markersize=2,  markeredgecolor='xkcd:gray')
meanprops = dict(marker = 'o', markerfacecolor = 'blue', ms = 0, mec = 'k', mew = 0, linestyle = '--', linewidth = 1.5, color = 'k')
meanlineprops = dict(linestyle = '-', lc = 'k', lw = 2)
boxprops = dict(color = 'k', linewidth = 1.5)
capprops = dict(color = 'k', linewidth = 1.5)
whiskerprops = dict(color = 'k', linecolor = 'k')
boxwidth = 0.35
linewidth = 1.5
#%% build some csvs or something that are easier to work with 

# nclist = glob.glob(os.path.join(stats_path, 'turnstats', '*.nc'))
# # nclist = ['/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats/kasai_destriped_masks_stats.nc',
# #           '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats/congo_destriped_masks_stats.nc']
# nturns = pd.DataFrame()
# maxturns = pd.DataFrame()
# for nc in nclist:
#     statstable = xr.load_dataset(nc)
    
#     rivname = nc.split('/')[-1].split('_masks_stats.nc')[0]
#     n_turns = statstable.numturns.to_numpy().ravel()
#     n_turns = pd.DataFrame(n_turns[n_turns!=0], columns = [rivname])
    
    
#     mx_turns = statstable.maxtime.to_numpy().ravel()
#     mx_turns = pd.DataFrame(mx_turns[~np.isnan(mx_turns)], columns = [rivname])
    
#     nturns = pd.concat([nturns, n_turns], axis = 1)
#     maxturns = pd.concat([maxturns, mx_turns], axis = 1)
    
# maxturns.to_csv(os.path.join(stats_path, 'masters', 'max_turntime_master.csv'))    
# nturns.to_csv(os.path.join(stats_path, 'masters', 'num_turnovers_master.csv'))    

#____________________________________
# load csvs

boxpos = np.arange(0, 21)
maxturns = pd.read_csv(os.path.join(stats_path, 'masters', 'max_turntime_master.csv'), index_col=0)
nturns = pd.read_csv(os.path.join(stats_path, 'masters', 'num_turnovers_master.csv'), index_col=0)
len_turns = pd.read_csv(os.path.join(stats_path, 'masters', 'length_turnover_frequencies.csv'), index_col = 0)

nturns_ds = pd.DataFrame(nturns.describe()) ## descriptive statistics for the number of turnover times 
maxturns_ds = pd.DataFrame(maxturns.describe()) ## descriptive statistics for the max turnover length 

# # quick cleanup
# nturns = nturns.drop(columns = ['kasai'])
# maxturns = maxturns.drop(columns = ['kasai'])

#%% build mega merge df

## load swatches
hex_codes = pd.read_csv('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-hex_codes.csv')
hex_codes['rgb'] = [tuple(int(value) / 255.0 for value in rgb.split(',')) for rgb in hex_codes['rgb']]

print(hex_codes.columns)
swatches = ['#ffffe4', '#fff4b8', '#ffe98b', '#ffde5a', '#ffd126', '#ffc31d', '#ffb416', '#ffa50f', '#ff9608', '#ff8500']

cmap = mcolors.ListedColormap(swatches)


#%% Plot boxplots pixel scale
colorby_attr = 'bedload_qw'
fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout = True, dpi = 150, figsize = (15, 6))

ntbox = nturns.boxplot(ax=ax1, whis=[5, 95], showfliers=True, flierprops=flierprops, grid=False,
                        positions=boxpos, 
                        capprops=capprops, widths=boxwidth,  return_type='dict', patch_artist = True)

for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
    [ax1.add_artist(artist) for artist in ntbox[element]]
    
for patch, color in zip(ntbox['boxes'], hex_codes[colorby_attr]):
    patch.set_facecolor(color)
    patch.set_edgecolor('k')
    patch.set_linewidth(1.5)
    
for whisker in ntbox['whiskers']:
    whisker.set_linewidth(1.5)
    whisker.set_color('k')
    
for median in ntbox['medians']:
    median.set_linewidth(1.5)
    median.set_color('k')    
ax1.set_xticks(boxpos, labels = abr_rivnames);
ax1.set_ylabel('Number of turnovers per px')


mtbox = maxturns.boxplot(ax=ax2, whis=[5, 95], showfliers=True, flierprops=flierprops, grid=False,
                        positions=boxpos,
                        capprops=capprops, widths=boxwidth, boxprops=boxprops, return_type='dict', patch_artist = True)

for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps', 'means']:
    [ax2.add_artist(artist) for artist in mtbox[element]]

for patch, color in zip(mtbox['boxes'], hex_codes[colorby_attr]):
    patch.set_facecolor(color)
    patch.set_edgecolor('k')
for whisker in mtbox['whiskers']:
    whisker.set_linewidth(1.5)
    whisker.set_color('k')
for median in mtbox['medians']:
    median.set_linewidth(1.5)
    median.set_color('k')
    

    
ax2.set_xticks(boxpos, labels = abr_rivnames);
ax2.set_ylabel('Max turntime length')    

norm = mcolors.Normalize(vmin=inventory[colorby_attr].min(), vmax=inventory[colorby_attr].max())

# Create a scalar mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add colorbar to the plot
cbar = fig.colorbar(sm, ax=ax2, orientation='horizontal')
cbar.set_label(colorby_attr)


#%% plot boxplot of fraction of the channel area turned over per year

csvs = glob.glob(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats/propturn/*_propturn.csv')
rivlist = [c.split('/')[-1].split('_propturn.csv')[0] for c in csvs] ## get a list of names
dataframes = [pd.read_csv(f, header = 0, index_col = 0) for f in csvs]
propturn_ds = pd.DataFrame(columns = rivlist, index = nturns_ds.index.values)

for r, df in enumerate(dataframes):
    propturn_ds[rivlist[r]] = df['prop_turnover'].describe()
#%% turnoverr prop boxplots
fig, ax = plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True, sharey = True)
ax = ax.ravel()


for df in range(len(rivlist)):
    ax[df].plot(dataframes[df].index.values, dataframes[df]['prop_turnover'], ls = '-', c = '#F896D8', lw = 2.5)
    ax[df].plot(dataframes[df].index.values, dataframes[df]['prop_turnwet'], ls = '--', c = '#1E78C2', lw = 2)
    ax[df].plot(dataframes[df].index.values, dataframes[df]['prop_turndry'], ls = '--', c = '#E28413', lw = 2)
    ax[df].fill_between(dataframes[df].index.values, 1, 3, color = '#9093B6', alpha = .5)
    ax[df].set_title(rivlist[df])
    ax[df].set_ylabel('Turnover prop')
    ax[df].set_xlabel('Year')
    ax[df].xaxis.set_minor_locator(MultipleLocator(1)) 
ax[df].set_xlim(1999, 2023)
ax[df].set_ylim(0, 2)
#%%boxlot of proportion of bed turnover

plt.figure(figsize = (15, 4), dpi = 300)
plt.ylabel('Proportion of bed turned over');
plt.xlabel('River');

all_boxes = []
for df in range(len(rivlist)):
    time_bp =  plt.boxplot(dataframes[df]['prop_turnover'], whis = [5, 95], 
                           positions=[df], showfliers = True, flierprops=flierprops, 
                           notch = True, capprops = capprops, widths = boxwidth, patch_artist = True)
    all_boxes.append(time_bp['boxes'][0]) ##collect the boxes


# Apply colors to the collected boxes
for patch, color in zip(all_boxes, hex_codes['terrain_slope']):
    patch.set_facecolor(color)
    patch.set_edgecolor('k')
    patch.set_linewidth(1.5)

# Apply styles to whiskers and medians
for bp in [time_bp]:
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
        whisker.set_color('k')
    for median in bp['medians']:
        median.set_linewidth(1.5)
        median.set_color('k')
    
plt.xticks(boxpos, labels = abr_rivnames);

#%% make some scatter plots

colorby_attr = 'efficiency'
color_df = inventory_alphabetical

fig, ax = plt.subplots(1, 3, figsize = (10, 4), tight_layout = True, dpi = 300) 

nt = ax[0].scatter(inventory_alphabetical.loc[:, 'bed-ssc_qw_m3yr'],  nturns_ds.loc['mean', :], 
                   c = color_df[colorby_attr], ec = 'k',
                   cmap = 'Blues')#, norm = mcolors.LogNorm());

maxt = ax[1].scatter(inventory_alphabetical.loc[:, 'bed-ssc_qw_m3yr'], maxturns_ds.loc['mean', :], 
                      c = color_df[colorby_attr], ec = 'k',
                      cmap = 'Blues')#, norm = mcolors.LogNorm());

propt = ax[2].scatter(inventory_alphabetical.loc[:, 'bed-ssc_qw_m3yr'], propturn_ds.loc['mean', :], 
                      c = color_df[colorby_attr], ec = 'k',
                      cmap = 'Blues')#, norm = mcolors.LogNorm());

# propt = ax[2].scatter(inventory_alphabetical.loc[:, 'bed-ssc_qw_m3yr'], inventory_alphabetical.loc[:, '3e_overlap'], 
#                       c = color_df[colorby_attr], ec = 'k',
#                       cmap = 'Blues', norm = mcolors.LogNorm());
fig.colorbar(propt, ax = ax[2], orientation = 'vertical', label = colorby_attr)

ax[0].set_ylabel('mean nturns');
ax[0].set_xlabel('sed flux vol');
ax[1].set_ylabel('Mean max turnover length');
ax[1].set_xlabel('sed flux vol');
ax[2].set_ylabel('3e');
ax[2].set_xlabel('sed flux vol');


# ax[2].set_xscale('log')

#%% test histors

fig, ax = plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
ax = ax.ravel()

for a, col in enumerate(nturns.columns):
    ax[a].hist(nturns.loc[:, col]/inventory_alphabetical['3e_overlap'].loc[a], bins = np.arange(0, 2, .25), fc = 'xkcd:light grey', ec = 'k')

    ax[a].set_title(f'num/3eov {col}')



fig, ax = plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
ax = ax.ravel()

for a, col in enumerate(nturns.columns):
    ax[a].hist(maxturns.loc[:, col], bins = np.arange(0, 25), fc = 'xkcd:light grey', ec = 'k')

    ax[a].set_title(f'max {col}')

#%% plot the distribution of the lengths of turnover times per pixel

fig, ax = plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
ax = ax.ravel()

for a, col in enumerate(len_turns.columns):
    ax[a].bar(len_turns.index.values, len_turns[col], width = 1, align = 'edge', color = 'xkcd:light grey', edgecolor = 'k', linewidth = .5)
    
    ax[a].set_title(col)
    ax[a].set_xlabel('Length of turnover')
    ax[a].set_ylabel('Frequency')



fig, ax = plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
ax = ax.ravel()
for a, col in enumerate(nturns.columns):
    ax[a].hist(nturns[col], bins = np.arange(0, 25), width = 1, color = 'xkcd:light grey', edgecolor = 'k', linewidth = .5)
    
    ax[a].set_title(col)
    ax[a].set_xlabel('Number of turnovers')
    ax[a].set_ylabel('Frequency')


#%% plot multiple ddescriptive statistics at once

bulk_stats_ncs = glob.glob(os.path.join(stats_path, 'ptt_bulkstats', '*.nc'))

maxfig, maxax =  plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
meanfig, meanax =  plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
medfig, medax =  plt.subplots(3, 7, figsize = (20, 8), dpi = 300, tight_layout = True, sharex = True)
maxax = maxax.ravel()
medax = medax.ravel()
meanax = meanax.ravel()

binwidth = 1
bins = np.arange(0, 25, binwidth)
for a, ncfile in enumerate(bulk_stats_ncs):
    
    xrstat = xr.load_dataset(ncfile)
    name = ncfile.split('/')[-1].split('.')[0]
    
    xr.plot.hist(xrstat.maxtime, ax = maxax[a], bins = bins, ec = 'k', fc = 'xkcd:light grey')
    xr.plot.hist(xrstat.maxtime_w4, ax = maxax[a], bins = bins, ec = 'b', histtype = 'step', lw = 1.5)
    xr.plot.hist(xrstat.maxtimed_d4, ax = maxax[a], bins = bins, ec = 'xkcd:burnt orange', histtype = 'step', lw = 1.5)
    maxax[a].set_title(name)
    maxax[a].set_xlabel('Max turnover time')
    
    
    xr.plot.hist(xrstat.medtime, ax = medax[a], bins = bins, ec = 'k', fc = 'xkcd:light grey')
    xr.plot.hist(xrstat.medtime_w4, ax = medax[a], bins = bins, ec = 'b', histtype = 'step', lw = 1.5)
    xr.plot.hist(xrstat.medtimed_w4, ax = medax[a], bins = bins, ec = 'xkcd:burnt orange', histtype = 'step', lw = 1.5)
    medax[a].set_title(name)
    medax[a].set_xlabel('median turnover time')
    
    xr.plot.hist(xrstat.meantime, ax = meanax[a], bins = bins, ec = 'k', fc = 'xkcd:light grey')
    xr.plot.hist(xrstat.meantime_wetfor, ax = meanax[a], bins = bins, ec = 'b', histtype = 'step', lw = 1.5)
    xr.plot.hist(xrstat.meantimed_dryfor, ax = meanax[a], bins = bins, ec = 'xkcd:burnt orange', histtype = 'step', lw = 1.5)
    meanax[a].set_title(name)
    meanax[a].set_xlabel('mean turnover time')

#%% plot hotspot maps uing the bulk stts data

hotspot_cols = ['#001524', '#0B3B49', '#15616D', '#8AA79F', '#FFECD1', '#FFB569', '#FF9935', '#FF7D00', '#BC5308', '#78290F'] ## colour map from coolors, not saved dont delete
hot_cols = mcolors.LinearSegmentedColormap.from_list('hotspotcols', hotspot_cols, N = 26)  # define the colormap

# define the bins and normalize
bounds = np.arange(0, 26)
norm = mcolors.BoundaryNorm(bounds, hot_cols.N)

for a, ncfile in enumerate(bulk_stats_ncs):
    
    xrstat = xr.load_dataset(ncfile)
    name = ncfile.split('/')[-1].split('.')[0]

    plt.figure(tight_layout = True, dpi = 300)
    xr.plot.imshow(xrstat.meantime, cmap = hot_cols, norm = norm, add_colorbar=True)
    plt.title(f' meantime, {name}')
    
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    plt.savefig(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnfigs/hotspotmaps/mean_TT_{name}.png')
    






