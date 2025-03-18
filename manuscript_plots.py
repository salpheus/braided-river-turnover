#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:43:53 2024
Chapter 2 dissertation plots. 
@author: safiya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as mcol
import glob
import os
from labellines import labelLines
import scipy.stats as stats
import matplotlib.cm as cm
from xml.dom import minidom
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 12}

# plt.rc('legend',fontsize=1, title_fontsize=8)
mpl.rc('font', **font)

def func(x, a, b, c):
    return a * np.exp(-b * x) + c
#%% make contnuous cmaps
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcol.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

#%% Import datafiles
inventory = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', 
                          sheet_name='inventory_py', index_col = 0)
small = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', 
                          sheet_name='small-rivers', index_col = 0)
dv_drain= pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', 
                          sheet_name='river_dv_area', index_col = 0)

abr_rivnames = ['MOD', 'ADD', 'ADU', 'BET', 'BHA', 'BRA', 'COL', 'CON', 'IND', 'IRA', 'IRU', 'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK', 'TAN', 'YUE', 'YUK']

propturn_wetarea = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/real_prop_change.xlsx', 
                     sheet_name = 'jus_delta', header = 0, index_col = 0, usecols='A:AA', skiprows = 73, nrows = 21)  

propturn_corridor= pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/real_prop_change.xlsx', 
                    sheet_name = 'jus_delta', header = 0, index_col = 0, usecols='A:AA', skiprows = 48, nrows = 21)  

med_ann_turnover =  pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/turnover_cells_annual.xlsx', 
                     header = 0, index_col = 0, sheet_name = 'Sheet2')  

inventory['turn_freq'] = inventory['num_turns_mean']/25
combo_params = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', #, 
                  'mean_slope', 'DVIc', 'ndvi_med',  #'max_tt_mean',
                'efficiency', 'med_ebi',]
#%% PLOT DV METRICS COLOURED BY CATCHMENT AREA

# fig, ax = plt.subplots(1, 3, figsize = (12,4), tight_layout = True, dpi = 300, sharex = True)

# ax[0].scatter(dv_drain['Area'], dv_drain['DVIc'], s = 125, marker = 'o', c = 'k', edgecolor = 'k', linewidth = .75, label = 'DVIc', alpha = .75)#, c = dv_drain['Area'], alpha = .75,
# ax[0].scatter(dv_drain['Area'], dv_drain['DVIa'], s = 125, marker = 'o', c = 'w', edgecolor = 'k', linewidth = .75, label = 'DVIa')#, c = dv_drain['Area'], alpha = .75,
# ax[1].scatter(dv_drain['Area'], dv_drain['uqw'], s = 125, marker = 'o', c = '#064789', edgecolor = 'k', linewidth = .75, label = 'DVIa')#, c = dv_drain['Area'], alpha = .75,
# ax[2].scatter(dv_drain['Area'], dv_drain['uqs'], s = 125, marker = 'o', c = '#F2F3ae', edgecolor = 'k', linewidth = .75, label = 'DVIa')#, c = dv_drain['Area'], alpha = .75,
#             # norm = mcol.LogNorm(vmin = 10000, 
#             #                     vmax = 1000000))
# # plt.scatter(dv_drain['Area'], dv_drain['DVIc'], s = 250, marker = 'o', edgecolor = 'k', linewidth = .75, c = dv_drain['Width'], 
# #             norm = mcol.LogNorm(vmin = 100, 
# #                                 vmax = 2000))
# for n, names in enumerate(dv_drain.index.values): 
#     # plt.annotate(names, (dv_drain['DVIa'][n], dv_drain['DVIc'][n]), 
#     if names in ['MBA', 'KAB', 'BUR', 'BKH', 'KKA', 'BMJ', 'PLA', 'SUS']:
#         ax[0].annotate(names, (dv_drain['Area'][n], dv_drain['DVIa'][n]), 
#                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
#         ax[0].annotate(names, (dv_drain['Area'][n], dv_drain['DVIc'][n]), 
#                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
#         ax[1].annotate(names, (dv_drain['Area'][n], dv_drain['uqw'][n]), 
#                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
#         ax[2].annotate(names, (dv_drain['Area'][n], dv_drain['uqs'][n]), 
#                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
# # plt.colorbar(label = 'Catchment Area $km^{2}$')
# # plt.colorbar(label = 'Width, m')
# # plt.xlabel('Average Discharge Variability, DVIa')
# ax[0].set_xlabel('Catchment Area $km^{2}$')
# ax[1].set_xlabel('Catchment Area $km^{2}$')
# ax[2].set_xlabel('Catchment Area $km^{2}$')
# ax[0].legend()
# ax[0].set_ylabel('Discharge Variability')
# ax[1].set_ylabel('Unit Discharge, $m^{2}/s$')
# ax[2].set_ylabel('Unit Sedient Flux, $m^{2}/yr$')
# plt.yscale('log')
# plt.xscale('log')
# ax[0].axvline(25000, ls = '--')
# ax[1].axvline(25000, ls = '--')
# ax[2].axvline(25000, ls = '--')

# ax[0].set_yscale('log')
# # plt.figure(figsize = (6,6), tight_layout = True, dpi = 300)

# # plt.scatter(dv_drain['Area'], dv_drain['DVIc'], s = 125, marker = 'o', c = 'k', edgecolor = 'k', linewidth = .75, label = 'DVIc', alpha = .75)#, c = dv_drain['Area'], alpha = .75,
# # plt.scatter(dv_drain['Area'], dv_drain['DVIa'], s = 125, marker = 'o', c = 'w', edgecolor = 'k', linewidth = .75, label = 'DVIa')#, c = dv_drain['Area'], alpha = .75,
# #             # norm = mcol.LogNorm(vmin = 10000, 
# #             #                     vmax = 1000000))
# # # plt.scatter(dv_drain['Area'], dv_drain['DVIc'], s = 250, marker = 'o', edgecolor = 'k', linewidth = .75, c = dv_drain['Width'], 
# # #             norm = mcol.LogNorm(vmin = 100, 
# # #                                 vmax = 2000))
# # for n, names in enumerate(dv_drain.index.values): 
# #     # plt.annotate(names, (dv_drain['DVIa'][n], dv_drain['DVIc'][n]), 
# #     if names in ['MBA', 'KAB', 'BUR', 'BKH', 'KKA', 'BMJ', 'PLA', 'SUS']:
# #         plt.annotate(names, (dv_drain['Area'][n], dv_drain['DVIa'][n]), 
# #                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
# #         plt.annotate(names, (dv_drain['Area'][n], dv_drain['DVIc'][n]), 
# #                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
# # # plt.colorbar(label = 'Catchment Area $km^{2}$')
# # # plt.colorbar(label = 'Width, m')
# # # plt.xlabel('Average Discharge Variability, DVIa')
# # plt.xlabel('Catchment Area $km^{2}$')
# # plt.legend()
# # plt.ylabel('Discharge Variability')
# # plt.yscale('log')
# # plt.xscale('log')
# # plt.axvline(25000, ls = '--')

# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/small_comprison.png', transparent = True)

fig, ax = plt.subplots(1, 2, figsize = (8, 4), tight_layout = True, dpi = 300, sharex = True, sharey = True)

plt.yscale('log')
# plt.xscale('log')

big = dv_drain[dv_drain['size_flag']<2]
smol = dv_drain[dv_drain['size_flag']>1]
ax[0].scatter(big['DVIa'], big['DVIc'], s = 125, marker = 'o', c = big['kgcol'], edgecolor = 'k', linewidth = .75, label = 'Study dataset', alpha = .75)#, c = dv_drain['Area'], alpha = .75,
catch = ax[1].scatter(big['DVIa'], big['DVIc'], s = 125, marker = 'o', c = big['Area'], edgecolor = 'k', linewidth = .75, label = 'Study dataset', 
              alpha = .75, norm = mcol.LogNorm(vmin = 1000, vmax = 1000000), cmap = 'hsv')

ax[1].scatter(smol['DVIa'], smol['DVIc'], s = 125, marker = 'P', c = smol['Area'], edgecolor = 'k', linewidth = .75, label = 'Small rivers',
              alpha = .75, norm = mcol.TwoSlopeNorm(vmin = 1000, vcenter=100000, vmax = 1000000), cmap = 'seismic')
ax[0].scatter(smol['DVIa'], smol['DVIc'], s = 125, marker = 'P', c = smol['kgcol'], edgecolor = 'k', linewidth = .75, label = 'Study dataset', 
              alpha = .75) 

ax[0].set_xlabel('DVIa')
# ax[1].set_xlabel('DVIa')
ax[0].legend()
ax[1].legend()
ax[0].set_ylabel('DVIc')
# ax[1].set_ylabel('DVIc')


# plt.figure(figsize = (6,6), tight_layout = True, dpi = 300)

# plt.scatter(dv_drain['Area'], dv_drain['DVIc'], s = 125, marker = 'o', c = 'k', edgecolor = 'k', linewidth = .75, label = 'DVIc', alpha = .75)#, c = dv_drain['Area'], alpha = .75,
# plt.scatter(dv_drain['Area'], dv_drain['DVIa'], s = 125, marker = 'o', c = 'w', edgecolor = 'k', linewidth = .75, label = 'DVIa')#, c = dv_drain['Area'], alpha = .75,
#             # norm = mcol.LogNorm(vmin = 10000, 
#             #                     vmax = 1000000))
# # plt.scatter(dv_drain['Area'], dv_drain['DVIc'], s = 250, marker = 'o', edgecolor = 'k', linewidth = .75, c = dv_drain['Width'], 
# #             norm = mcol.LogNorm(vmin = 100, 
# #                                 vmax = 2000))
# for n, names in enumerate(dv_drain.index.values): 
#     # plt.annotate(names, (dv_drain['DVIa'][n], dv_drain['DVIc'][n]), 
#     if names in ['MBA', 'KAB', 'BUR', 'BKH', 'KKA', 'BMJ', 'PLA', 'SUS']:
#         plt.annotate(names, (dv_drain['Area'][n], dv_drain['DVIa'][n]), 
#                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
#         plt.annotate(names, (dv_drain['Area'][n], dv_drain['DVIc'][n]), 
#                     fontsize = 8, color = 'k', zorder = 10, xytext=(2, 2), textcoords='offset points');
plt.colorbar(catch, label = 'Catchment Area $km^{2}$', shrink = .5)


# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/DVmetric_comp_nocb.svg', transparent = True)


#%% Plot bivariate plots to show correlation
fig, ax = plt.subplots(2, 4, figsize = (12, 6), tight_layout = True, sharey = True, dpi = 300)

labels = ['Bedload proportion', 'Unit Discharge ($m^{2}$/s)', 'Unit Sediment Flux ($m^{2}$/yr)', 'Slope', 
          'Cumulative Discharge Variability', 'NDVI', 'Median eBI', 'Efficiency (Tm/Tr)']#'Tr_timescale', 'Tm_timescale']
bv_params = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', #, 
                  'mean_slope', 'DVIc', 'ndvi_med',  #'max_tt_mean',
                 'med_ebi','efficiency']#'Tr_timescale', 'Tm_timescale']
ax = ax.ravel()
small_riv = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr']

colours = ['#423629', '#064789', '#F2F3ae', 'xkcd:light grey', '#ef8354', '#4d8b31', '#b5ffe9', '#a6b1e1']
for a, var in enumerate(bv_params):
    
    scatter = ax[a].scatter(inventory[var], inventory['turn_freq'], c = colours[a], s = inventory['w_scaler']*50, edgecolor = 'k', linewidth = .75)   
    
    ax[a].set_xlabel(labels[a])
    # for n, names in enumerate(abr_rivnames): 
    #     ax[a].annotate(names, (inventory[var][n], inventory['turn_freq'][n]), 
    #                 fontsize = 10, color = 'k', zorder = 100, xytext=(2, 2), textcoords='offset points');
    if var in ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', 'part_size_mm', 
                      'mean_slope', 'Tm_timescale', 'Tr_timescale', 'stream_pow']:    
        ax[a].set_xscale('log')
        
        # slope, c, r, p, se = stats.linregress(np.log10(inventory[var].to_numpy()), inventory['turn_freq'].to_numpy())
        # x = np.arange(np.log10(inventory[var].min()), np.log10(inventory[var].max()))
        # y = slope*x+c
        
        # ax[a].plot(x, y, 'k--', alpha = .6)
        # popt, pcov = curve_fit(func, inventory[var], inventory['turn_freq'])
        # x = np.linspace(inventory[var].min(), inventory[var].max())
        # ax[a].plot(x, func(x, *popt), 'k--')
        
   
    # slope, c, r, p, se = stats.linregress(inventory[var].to_numpy(), inventory['mean_tt_length'].to_numpy())
    # x = np.linspace(inventory[var].min(), inventory[var].max())
    # y = slope*x+c
        
    # ax[a].plot(x, y, 'k--', alpha = .6)
    # if c < 0:
    #     ax[a].set_title(f'y = {np.round(slope, 2)}x - {np.round(abs(c), 2)} R$^2$ = {np.round(r, 1)}')
    # else:
    #     ax[a].set_title(f'y = {np.round(slope, 2)}x + {np.round(abs(c), 2)} R$^2$ = {np.round(r, 1)}')
    # if var in small_riv:
    #     ax[a].vlines(small[var], 0, 0.3)
    # ax[a].set_ylim(inventory['turn_freq'].min()), inventory['turn_freq'].max()
    
        # produce a legend with a cross-section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax[-1].legend(handles, labels, loc="upper right", title="Sizes")
ax[0].set_ylabel('Mean Turnover frequency');
# ax[3].set_ylabel('CV Turnover frequency');
ax[4].set_ylabel('Mean Turnover frequency');

# plt.figure
plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/correlationplots_meanturnfreq_nolabels.svg', transparent = True)


#%% plot the turntime rate plots

# fig, ax = plt.subplots(2, 1, figsize - (12, 6), tight_layout = True, sharey = True, sharex = True)

#                           sheet_name='inventory_py', index_col = 0)
# tr = inventory['Tr_timescale'].to_numpy()
# tm = inventory['Tm_timescale'].to_numpy()

# dvia_csvs = glob.glob('/Volumes/SAF_Data/SAF_Data/CHAPTER2/turnover_discharge_records/DVIa/*.csv')
# dvia_dfs = [pd.read_csv(f, header = 0, index_col = 0) for f in dvia_csvs]
# inventory['num_turns_ent'] = stats.gamma.entropy(inventory['r_nturn_shape'], scale=inventory['r_nturn_shape'])
# inventory['ebi_ent'] = stats.gamma.entropy(inventory['r_ebi_shape'], scale=inventory['r_ebi_shape'])
#%% ADDED to XL file!!----add dvia statistics to inventory to make plotting easier
# idxnames = ['dvia_count', 'dvia_mean', 'dvia_std', 'dvia_min', 'dvia_q1', 'dvia_med', 'dvia_q3', 'dvia_max']
# dvia_stats = pd.DataFrame()
# for f, df in enumerate(dvia_dfs):
#     nm = dvia_csvs[f].split('/')[-1].split('_dvia.csv')[0]    
#     dvia_stats[nm] = df['Discharge'].describe()
    
# for i, idx in enumerate(dvia_stats.index.values):
#     dvia_stats = dvia_stats.rename(index={idx: idxnames[i]})
    
# inventory = pd.concat((inventory, dvia_stats.T), axis = 1)
# inventory.iloc[0, -7:] = 1
#%% map pf efficiency

plt.figure(figsize = (4, 4), tight_layout = True, dpi = 300)
# plt.scatter(inventory['Tr_timescale'], inventory['Tm_timescale'], 
#             marker = 'o', fc = 'w', ec = 'k' )
sed = plt.scatter(inventory['Tr_timescale'], inventory['Tm_timescale'], 
            marker = 'o', c = inventory['unit_sedflux_m2yr'], ec = 'k' , cmap = 'inferno', 
            norm = mcol.LogNorm(vmin = inventory['unit_sedflux_m2yr'].min(), 
                                vmax = inventory['unit_sedflux_m2yr'].max()))
plt.axline(xy1=(0, 0), slope = 1, c = 'k', ls = '--', zorder = 0)
plt.ylabel('Overap Decay Timescale, $T_M$ (yr)')
plt.xlabel('Floodplain Reworking Timescale, $T_R$ (yr)')

plt.colorbar(sed, label = 'unit sed flux')

for i in range(len(abr_rivnames)):
    plt.annotate(abr_rivnames[i], (tr[i], tm[i]), fontsize = 6)

plt.xscale('log')
plt.yscale('log')
# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/efficiency_labels.svg', transparent = True)

#%% plot boxplots of attribut data

### NDVI colrbar = greens starting 60935D asparagus ['#002200', '#063e0e', '#275928', '#447642', '#61945e', '#7fb47b', '#9ed49a', '#bef5b9']
### EBI colorbar = prup-blue-green: 4c2c69, 87bcde, e0ca3c ['#4c2c69', '#625a8e', '#768ab5', '#6abe9e', '#29795d', '#003921']
### DVIa colorbar = maybe something from grey to orange? grey to brown sugar A1674A ['#aeaeae', '#b2a7a3', '#b5a198', '#b89a8d', '#ba9482', '#bb8d77', '#bd876d', '#bd8062']
### Slope = grey? starting rich black 011627 ['#011627', '#1c2e41', '#37485d', '#53647a', '#718298', '#8fa1b8', '#afc1d9', '#cfe1fa']

ndvi_cols = ['#002200', '#063e0e', '#275928', '#447642', '#61945e', '#7fb47b', '#9ed49a', '#bef5b9']
ebi_cols = ['#4c2c69', '#625a8e', '#768ab5', '#6abe9e', '#29795d', '#003921']
dvia_cols = ['#aeaeae', '#b2a7a3', '#b5a198', '#b89a8d', '#ba9482', '#bb8d77', '#bd876d', '#bd8062']
slope_cols = ['#011627', '#1c2e41', '#37485d', '#53647a', '#718298', '#8fa1b8', '#afc1d9', '#cfe1fa']
slope_cmap = get_continuous_cmap(slope_cols)
ndvi_cm = mcol.ListedColormap(ndvi_cols, name = 'ndvi_cmap')
ebi_cm= mcol.ListedColormap(ebi_cols, name = 'ebi_cmap')
dvia_cm = mcol.ListedColormap(dvia_cols, name = 'dvia_cmap')
slope_cm = mcol.LinearSegmentedColormap('clope_cmap', slope_cols)

#hot_cols = mcolors.LinearSegmentedColormap.from_list('hotspotcols', hotspot_cols, N = 26)  # define the colormap
# define the bins and normalize
# bounds = np.arange(0, 26)
# norm = mcolors.BoundaryNorm(bounds, hot_cols.N)
# 

ndvinorm = mcol.Normalize(inventory['ndvi_min'].min(), inventory['ndvi_max'].max())
ebinorm = mcol.Normalize(1, inventory['max_ebi'].max())
dvianorm = mcol.CenteredNorm(vcenter = 2.0, halfrange = 2.0)
slopenorm = mcol.LogNorm(vmin = 1e-5, vmax = 1e-2)
qwnorm = mcol.Normalize(inventory['unit_discharge_m2s'].min(), inventory['unit_discharge_m2s'].max())
qsnorm = mcol.Normalize(inventory['unit_sedflux_m2yr'].min(), inventory['unit_sedflux_m2yr'].max())
# turn_cols =['#002753', '#003c69', '#005380', '#17705d', '#44883c', '#6d9e1e', '#94b300', '#b9c820', '#d7df4e', '#fff291']
# turn_cols =['#003138', '#27430e', '#3b530d', '#566211', '#786f19', '#997d22', '#ba8a2a', '#db9733', '#f5a754', '#fdc065']
# turn_cols =['#0014a5', '#0c399b', '#145195', '#1c688f', '#237d89', '#2f9283', '#4fa67a', '#6fb871', '#deb54e', '#ffc353']
turn_cols =['#000079', '#3f137a', '#681f7b', '#8d2a7b', '#b2357c', '#cb4c84', '#e3638b', '#fb7b92', '#ffa43d', '#ffc635', '#ffe556']









colours = ['#60935D', '#4c2c69', '#E0CA3C', '#87bcde', '#A1674A', '#D8D8F6']
#%% make the plots
turnover_cm = get_continuous_cmap(turn_cols)

colby = 'num_turns_mean'
# colbynorm = mcol.Normalize(np.floor(inventory[colby].min()), np.ceil(inventory[colby].max()))
colbynorm = mcol.Normalize(3, 6.5)
nturns_norm = mcol.Normalize(1, 25)
#%%
fig, ax = plt.subplots(2, 3, figsize = (12, 9), tight_layout = True, sharey = True, dpi = 150)
ax = ax.ravel()

cmaps = [ndvi_cm, ebi_cm, dvia_cm, slope_cm, 'Blues', 'Oranges']
norms = [ndvinorm, ebinorm, dvianorm, slopenorm, qwnorm, qsnorm]
varslist = ['ndvi_med', 'med_ebi', 'dvia_med', 'unit_discharge_m2s', 'unit_sedflux_m2yr', 'mean_slope'] 

for v, var in enumerate(varslist):
    plot = ax[v].scatter(inventory[var], inventory['num_turns_mean'], c = inventory[colby], cmap = turnover_cm, norm = colbynorm,
                    s = 125, ec = 'k')
    ax[v].set_xlabel(var)
    ax[v].set_ylabel('Mean number of turnovers')

ax[4].set_xscale('log')
ax[5].set_xscale('log')

# define error bars
ndvi_eb = inventory.loc[:, ['ndvi_q1', 'ndvi_q3']].T.to_numpy()
ebi_eb = inventory.loc[:, ['q1_ebi', 'q3_ebi']].T.to_numpy()
dvia_eb = inventory.loc[:, ['dvia_q1', 'dvia_q3']].T.to_numpy()

ndvi_eb[0, :] = inventory['ndvi_med']-ndvi_eb[0, :]
ndvi_eb[1, :] = ndvi_eb[1, :]-inventory['ndvi_med']

ebi_eb[0, :] = inventory['med_ebi']-ebi_eb[0, :]
ebi_eb[1, :] = ebi_eb[1, :]-inventory['med_ebi']

dvia_eb[0, :] = inventory['dvia_med']-dvia_eb[0, :]
dvia_eb[1, :] = dvia_eb[1, :]-inventory['dvia_med']

eblist = [ndvi_eb, ebi_eb, dvia_eb]
for v, var in enumerate(varslist[:3]):
    ax[v].errorbar(inventory[var], inventory['num_turns_mean'], xerr = eblist[v],
                    ecolor = 'k', elinewidth = 1, capsize = 5, capthick = 1, fmt = "none", zorder = 0)
    ax[v].set_xlabel(var)
    ax[v].set_ylabel('mean number of turnovers')

# cbax = ax[-2].inset_axes([0, -.25, 1, .3], transform=ax[-2].transAxes)
# cbax.axis('off')
# fig.colorbar(plot, ax = cbax,
#               pad = 0.1, orientation = 'horizontal', label = colby, panchor = False)
for var, ax in zip(varslist, ax): 
    for i in range(len(abr_rivnames)):
        ax.annotate(abr_rivnames[i], (inventory[var][i], inventory['num_turns_mean'][i]), fontsize = 6, color = 'm', zorder = 10)

# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/bivariate-plots-labelled_cbmean_nocb.svg')


#%% 


fig, ax = plt.subplots(1, 3, figsize = (7*1.5, 3*1.5), tight_layout = True, dpi = 300, 
                       gridspec_kw = {'wspace':0.2, 'hspace':0})

ax[0].scatter(inventory['unit_discharge_m2s'], inventory['unit_sedflux_m2yr'], c = inventory[colby], cmap = turnover_cm, 
              norm = colbynorm, s = 150, ec = 'k')
# ax[0].errorbar(inventory['unit_discharge_m2s'], inventory['dvia_med'], yerr = dvia_eb,
#                ecolor = 'k', elinewidth = 1, capsize = 5, capthick = 1, fmt = "none", zorder = 0)

ax[1].scatter(inventory['mean_slope'], inventory['bed_prop_of_total'], c = inventory[colby], cmap = turnover_cm, 
              norm = colbynorm, s = 150, ec = 'k')
# ax[1].errorbar(inventory['ndvi_med'], inventory['unit_sedflux_m2yr'], xerr = ndvi_eb,
#                ecolor = 'k', elinewidth = 1, capsize = 5, capthick = 1, fmt = "none", zorder = 0)

frm = ax[2].scatter(inventory['ndvi_med'], inventory['DVIc'], c = inventory[colby], cmap = turnover_cm, 
              norm = colbynorm, s = 150, ec = 'k')
ax[2].errorbar(inventory['ndvi_med'], inventory['DVIc'], xerr = ndvi_eb,
               ecolor = 'k', elinewidth = 2, capsize = 0, capthick = 0, fmt = "none", zorder = 0, alpha = .35)

for n, names in enumerate(abr_rivnames): 
    ax[0].annotate(names, (inventory['unit_discharge_m2s'][n], inventory['unit_sedflux_m2yr'][n]), 
                   fontsize = 6, color = 'm', zorder = 10)
    ax[1].annotate(names, (inventory['mean_slope'][n], inventory['bed_prop_of_total'][n]), 
                   fontsize = 6, color = 'm', zorder = 10)
    ax[2].annotate(names, (inventory['ndvi_med'][n], inventory['DVIc'][n]), 
                   fontsize = 6, color = 'm', zorder = 10)
ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_xlabel('Unit dscharge, m2s')
ax[0].set_ylabel('Unit Sed. Flux')

ax[1].set_xlabel('Slope')
ax[1].set_ylabel('Bedload Proportion')

ax[2].set_xlabel('NDVI')
ax[2].set_ylabel('DVIc')


cbax = ax[2].inset_axes([.97, 0.05, .25, 1], transform=ax[2].transAxes)
cbax.axis('off')
fig.colorbar(frm, ax = cbax,
              pad = 0.05, orientation = 'vertical', label = colby, panchor = False)

plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/xy-plots-labelled_cbmean_cb.png')

#%% plot the turnover per time plots

# Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar().
# Make a user-defined colormap.
# colour prop lines by num turns
cpick = cm.ScalarMappable(norm=colbynorm,cmap=turnover_cm)  
cpick.set_array([])

fig, ax = plt.subplots(figsize = (7*1.5, 3.5*1.5), tight_layout = True, sharex = True, sharey = True, dpi = 300)

for i, idx in enumerate(propturn_corridor.index.values):
    ptrun = ax.plot(np.arange(1999, 1999+len(propturn_corridor.iloc[i,  :])), 
               propturn_corridor.iloc[i, :].cumsum(), c = cpick.to_rgba(inventory[colby][i]), lw = inventory['efficiency'][i], label = abr_rivnames[i])
# for n, nm in enumerate(abr_rivnames):
#     ax.annotate(nm, (2023.1, propturn_corridor.iloc[n, :].sum()), fontsize = 6)
    
ax.set_title('flooded corridor')
ax.set_ylabel('Proportion of channel corridor turned over')
ax.set_xlabel('Year')
ax.set_xlim(1999, 2023.5)

# fig.colorbar(ax, cpick, label = 'mean number of turnovers')
## I've calculated it the right wat, but using the wetted area, you converge to 1 at the end so i am just making one plot for the corridor only
# for i, idx in enumerate(propturn_corridor.index.values):
#     ax[1].plot(np.arange(1999, 1999+len(propturn_wetarea.iloc[i, :])), 
#                propturn_wetarea.iloc[i, :].cumsum())
# for n, nm in enumerate(abr_rivnames):
#     ax[1].annotate(nm, (2023, propturn_wetarea.iloc[n, :].sum()))
 
# ax[1].set_title('wet area')


cbax = ax.inset_axes([.83, 0.02, .25, 1], transform=ax.transAxes)
cbax.axis('off')
fig.colorbar(frm, ax = cbax,
              pad = 0.01, orientation = 'vertical', label = colby, panchor = False)

lines = plt.gca().get_lines()
labelLines(lines, xvals = np.linspace(2003, 2024, 21), align = True, fontsize = 6);

# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/prop_turn_cbnturns.png')

#%% weird sed flux bypass plot
prop_turned = propturn_corridor.sum(axis = 1)
turn_delta = med_ann_turnover['median delta']#.loc[:, ['med_turndry_aream2', 'med_turnwet_aream2']].mean(axis = 1)
sedflux = (inventory['unit_sedflux_m2yr']).copy().to_numpy()
sedflux[0] = np.nan#(sedflux[0]/365.25) ## sedflux back to s, then upscale to same rate ts (12 modelts = 24hrs so 1day?)
# medturn_normqs = turn_delta/sedflux
medturn_normqs = med_ann_turnover['med_turndry_aream2']/sedflux
plt.figure(figsize = (6, 6), dpi = 300)
# plt.scatter(medturn_normqs, prop_turned)
plt.scatter(med_ann_turnover['med_turndry_aream2'], sedflux)
for n, nm in enumerate(abr_rivnames):
    # plt.annotate(nm, (medturn_normqs[n], prop_turned[n]))
    plt.annotate(nm, (med_ann_turnover['med_turndry_aream2'][n], sedflux[n]))
# plt.xscale('log')
plt.axvline(x =0, c = 'k', alpha = .2, zorder = 0)
# plt.ylabel('Proportion of channel corridor turned over after 25 years')
# plt.xlabel('median area change (dry-wet, med)/unit sedflux/yr')

plt.xlabel('median area turned dry')    
plt.ylabel('Unit sed flux m2/yr')    
#%% violin plots of ebi

flierprops = dict(marker='o', markerfacecolor='xkcd:gray', markersize=2,  markeredgecolor='xkcd:gray')
meanprops = dict(marker = 'o', markerfacecolor = 'blue', ms = 0, mec = 'k', mew = 0, linestyle = '--', linewidth = 1.5, color = 'k')
meanlineprops = dict(linestyle = '-', lc = 'k', lw = 2)
boxprops = dict(color = 'k', linewidth = 1.5)
capprops = dict(color = 'k', linewidth = 1.5)
whiskerprops = dict(color = 'k', linecolor = 'k')
boxwidth = 0.7
linewidth = 1.5


plt.figure(figsize = (7*1.5, 2*1.5), tight_layout = True, dpi = 300)
for idx, csv in enumerate(glob.glob('/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated/0_ebi_masters/*.csv')):
    name = csv.split('/')[-1].split('_master_ebi.csv')[0]
    ebidata = pd.read_csv(csv, header = 0, index_col = 0)
    # plt.boxplot(ebidata['ebi'].dropna(), positions = [idx], whis = (.5, .95), showfliers=True, flierprops=flierprops, 
    #                         capprops=capprops, widths=boxwidth)
    ebi_reliable = ebidata['ebi'][ebidata['ebi']>=1].dropna()
    print(ebi_reliable.min())
    violins = plt.violinplot(ebi_reliable, positions = [idx+1], 
                    showmedians = False, showextrema = True, widths=boxwidth, bw_method = 0.3)
    print(name, ebidata['ebi'].min())
    for pc in violins['bodies']:
        pc.set_facecolor(cpick.to_rgba(inventory[colby].iloc[idx+1]))
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        pc.set_linewidth(.5)
    # Change the color of the extrema lines
    # for line in :  # Min line
        violins['cmins'].set_color('k')
        violins['cmins'].set_linewidth(1)
        violins['cmaxes'].set_color('k')
        violins['cmaxes'].set_linewidth(1)
        violins['cbars'].set_color('k')
        violins['cbars'].set_linewidth(1)

    emin, q1, med, q3, emax = np.nanquantile(ebidata['ebi'], [.05, .25, .5, .75, .95])

    # plt.vlines(idx+1, ymin = emin, ymax = emax, ec = 'k', zorder = 100)
    plt.vlines(idx+1, ymin = q1, ymax = q3, ec = 'k', lw  = 5, zorder = 101)
    plt.scatter(idx+1, med, c = 'w', marker = 'o', s = 5, zorder = 102)

agubh2ebi = pd.read_excel('/Volumes/SAF_Data/SAF_Data/bar-manuscript_sum22/data-interp/BI-EBI.xlsx', sheet_name = 'ebi_chap2',
                          header = 0, index_col= 0).values.ravel()

violins = plt.violinplot(agubh2ebi, positions = [0], showmedians = False, 
               widths=boxwidth,  bw_method = 0.3);

emin, q1, med, q3, emax = np.nanquantile(agubh2ebi, [.05, .25, .5, .75, .95])

# plt.vlines(0, ymin = emin, ymax = emax, ec = 'k', zorder = 100)
plt.vlines(0, ymin = q1, ymax = q3, ec = 'k', lw  = 5, zorder = 101)
plt.scatter(0, med, c = 'w', marker = 'o', s = 5, zorder = 102)

for pc in violins['bodies']:
    pc.set_facecolor(cpick.to_rgba(inventory[colby].iloc[0]))
    pc.set_edgecolor('black')
    pc.set_linewidth(.5)
    pc.set_alpha(1)
    
violins['cmins'].set_color('k')
violins['cmins'].set_linewidth(1)
violins['cmaxes'].set_color('k')
violins['cmaxes'].set_linewidth(1)
violins['cbars'].set_color('k')
violins['cbars'].set_linewidth(1)


ax = plt.gca()
ax.set_xticks(np.arange(idx+2)) ## alays set tick range before prescribing labels
ax.set_xticklabels(abr_rivnames);
ax.set_ylabel('eBI')
plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/ebi_violinplot_cutoff1.svg')

#%% load data for plots of length and number of turnovers
stats_path = '/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/'
lengths = pd.read_csv('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/length_turnover_frequencies.csv', index_col = 0)
nturns = pd.read_csv(os.path.join(stats_path, 'masters', 'num_turnovers_master.csv'), index_col=0)

#%% plot violins
lfig, lax = plt.subplots(1, 21, figsize = (18, 5), tight_layout = True, sharey = True, sharex = True, dpi = 300,  gridspec_kw = {'wspace':0, 'hspace':0})
nfig, nax = plt.subplots(1, 21, figsize = (18, 5), tight_layout = True, sharey = True, sharex = True, dpi = 300,  gridspec_kw = {'wspace':0, 'hspace':0})

lax = lax.ravel()
nax = nax.ravel()
for a, riv in enumerate(lengths.columns):
    vals = lengths.index.values[:-1]
    freqs = lengths[riv][~np.isnan(lengths[riv])].to_numpy(dtype = int)
    
    distribution = np.repeat(vals, freqs) ## reproduce the distribution of turnover lengths across the entire river
    lax[a].hist(distribution, bins = np.arange(1, 25, 1), histtype = 'step', density = True, align = 'mid', orientation = 'horizontal', 
                ec = 'k', lw = 1.5)
   
    nax[a].hist(nturns[riv][~np.isnan(nturns[riv])]/25, bins = np.arange(0, 1, .08), histtype = 'step', density = True, align = 'mid', orientation = 'horizontal', 
                ec = 'k', lw = 1.5)
    
    
    lax[a].set_title(abr_rivnames[a])
    nax[a].set_title(abr_rivnames[a])
    
    if a>0:
        lax[a].axis('off')
        nax[a].axis('off')

lax[0].invert_yaxis()    
nax[0].invert_yaxis()   
lax[0].set_ylabel('Length of turnover [yr]')
nax[0].set_ylabel('Turnover Frequency [1/yr]')
    

    # len_violins = sns.violinplot(distribution, ax = ax[0], positions = [a], 
    #                 showmedians = False, showextrema = False, widths=boxwidth, bw_method = 0.3, density_norm = 'area')
     
    # nturn_violins = sns.violinplot(nturns[riv][~np.isnan(nturns[riv])], ax[1], positions = [a], 
    #                  showmedians = False, showextrema = False, widths=boxwidth, bw_method = 0.3, density_norm = 'area')

    # lmin, lq1, lmed, lq3, lmax = np.nanquantile(distribution, [.05, .25, .5, .75, .95])
    # ax[0].axvline([a], ymin = lq1, ymax = lq3, c = 'k', lw  = 5, zorder = 101)
    # ax[0].scatter([a], lmed, c = 'w', marker = 'o', s = 5, zorder = 102)

    # nmin, nq1, nmed, nq3, nmax = np.nanquantile(nturns[riv], [.05, .25, .5, .75, .95])
    # ax[1].axvline([a], ymin = nq1, ymax = nq3, c = 'k', lw  = 5, zorder = 101)
    # ax[1].scatter([a], nmed, c = 'w', marker = 'o', s = 5, zorder = 102)
    
# ax[1].set_xticks(np.arange(a+1)) ## alays set tick range before prescribing labels
# ax[1].set_xticklabels(abr_rivnames);
# ax[0].set_ylabel('Turnover length [yr]')
# ax[1].set_ylabel('Numer of turnovers')
lfig.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/len_violins.svg')    
nfig.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/nturn_violins.svg')    
    
#%% plot lena, irraaddy NDVI stack for appendix

# lenandvi= np.load('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/ndvi_stacks_withoutwater_clipped/lena.npy')[-1, :, :]
# indusndvi= np.load('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/ndvi_stacks_withoutwater_clipped/indus_r2.npy')[-1, :, :]
    
# lenamask = mpimg.imread('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may/lena/mask/1999on/lena_2023_05_01_2023_09_30_mask.tif')    
# indusmask = mpimg.imread('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may/indus_r2/mask/1999on/indus_r2_2023_01_01_2023_12_31_mask.tif')    

# lenamask[lenamask==0] = np.nan
# indusmask[indusmask==0] = np.nan
  
plt.figure(dpi = 600)
plt.imshow(lenamask, cmap = 'grey')
ndvi = plt.imshow(lenandvi, cmap = 'PiYG',vmin = -1, vmax = 1)
plt.axis('off')   
fig.colorbar(ndvi, label = 'NDVI', shrink = 0.5) 
plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/lenandvi.svg')

plt.figure(dpi = 300)
plt.imshow(indusmask, cmap = 'grey')
ndvi = plt.imshow(indusndvi, cmap = 'PiYG',vmin = -1, vmax = 1)
plt.axis('off')   
fig.colorbar(ndvi, label = 'NDVI', shrink = 0.5) 
plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/indusndvi.svg')

    
    
    
    
    
    
    
    
    

