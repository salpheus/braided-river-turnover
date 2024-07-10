#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 07:33:51 2024

@author: safiya
"""
from puller import get_paths
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
# import natsort
import os 
from xml.dom import minidom
import numpy.ma as ma
import matplotlib.colors as mcol
from matplotlib.colors import ListedColormap
'''Script to pull the parts of evan's code that generate csv files of mobility data for all rivers,
    then the csvs need to be sorted by i and plot the average of each i value'''
    
    
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 8}

mpl.rc('font', **font)    
#%% load mellow-rainbow colourmap

def make_cmap(colors, position=None, bit=False):

    if len(position) != len(colors):
        sys.exit('position length must be the same as colors')
    elif position[0] != 0 or position[-1] != 1:
        sys.exit('position must start with 0 and end with 1')
    
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):

        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

        cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    return cmap
    
##### load source xml file
xmldoc = minidom.parse('/Volumes/SAF_Data/Python/colourmaps/mellow-rainbow.xml')
itemlist = xmldoc.getElementsByTagName('Point')
data_vals=[]
color_vals=[]

for s in itemlist:
    
    data_vals.append(float(s.attributes['x'].value))
    color_vals.append((float(s.attributes['r'].value),
    float(s.attributes['g'].value),
    float(s.attributes['b'].value)))

##### construct the colormap

mycmap = make_cmap(color_vals,data_vals)    

#%%

data_folder = input('where is the data stored: ')
#%% mobility csv file will be stored in the same folder as the image and mask files.

# river = input('which river? ')
# base = f'/Volumes/SAF_Data/remote-data/watermasks/{data_folder}/'
base = f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/'
rivers = os.listdir(base)
# rivers = ['congo_lukolela_bolobo', 'lena']
rivers.remove('brahmaputra_yangcun')
rivers.remove('.DS_Store')
rivers.append('agubh2')
# rivers.remove('congo_new')
del rivers[:1]
# rivers = ['congo_lukolela_bolobo', 'kasai', 'brahmaputra_pandu', 'colville']
# rivers = ['southsask']
for river in rivers:
    print(river)    
    root = os.path.join(base, river)
    # root = f'/Volumes/SAF_Data/remote-data/watermasks/C02SR_Batch_WL2_Mar25_masks_clipped_nometadata/{river}'
    # root = f'/Volumes/SAF_Data/remote-data/watermasks/agubh2'
    # root = '/Volumes/SAF_Data/CHAPTER2/greenberg-area-data/full-dataset/Greenberg_AreaMobility_Data/NaturalRiverData/SSaskatchOutlook'
    # paths = get_paths(root)
    
    poly = f"/Volumes/SAF_Data/remote-data/watermasks/gpkgs_fnl/{river}.gpkg"
    # poly = "/Volumes/SAF_Data/CHAPTER2/greenberg-area-data/full-dataset/Greenberg_AreaMobility_Data/NaturalRiverData/SSaskatchOutlook/SSaskatchOutlook.gpkg"
    
    mobility_csvs = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/mobcsvs'
    out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999' # blocked fit_stats outputs
    mobility_out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/mobility'
    figout = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/figs'
    #uncomment line 95 if you want to recalculate mobility stats
    # get_mobility_rivers(poly, paths, river) ## gets mobility yearly, makes a csv in the root folder of mobility data
#%% compille csvs and compute curves
# stop = 30
# for river in rivers:
#     allcsvs = glob.glob(os.path.join(mobility_csvs, '*.csv')) 
#     path_list = natsort.natsorted(allcsvs)
    
    
#     get_stats(os.path.join(mobility_csvs, f'{river}_yearly_mobility.csv'), river, out, 30)
#     get_mobility(pd.read_csv(os.path.join(out, f'{river}_fit_stats.csv'), header = 0),
#                   os.path.join(mobility_out, f'{river}_mobility.csv'))
#%% load csvs and make plots

# river = 'congo_lukolela_bolobo'
# congo = pd.read_csv(os.path.join(out,f'{river}_yearly_mobility.csv'), header = 0, index_col=0)

# avgs = np.array([])
# ivals = congo['i'].unique()
# for i in ivals:
#     avgs = np.append(avgs, np.mean(congo['O_avg'][congo['i']==i])/1e6)

## create raw xy data for eq4
# out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/mobcsvs'

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

#%% calculate constants M and R
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
    aw50 = fits['Aw'][1]/1e6 #[km2] 
    cm50 = fits['CM'][1] #[1/T]
    pm50 = fits['PM'][1]/1e6 #[m2]
    
    channel_params.loc[:, river] = [cm50, pm50, aw50]
    
    dt = np.arange(0, len(full_dataset[full_dataset['river']==river]))
    
    ami = ((aw50-pm50) * np.exp(-1*cm50*dt) + pm50)
    
    Mct = cm50*(1-(pm50/aw50))
    linear_ov_decay.loc['Mct', river] = Mct
    
    plt.plot(dt, ami, zorder = 10000)
    
    # plt.yscale('log')
    plt.ylabel('Channel overlap area, km2');
    plt.xlabel('Duration since baseline');
    plt.title(f'{river}, 3e = {np.round(3/cm50, 1)}, A$_m$* = {np.round(1-(pm50/aw50), 1)}')
    # plt.savefig((f'{out}/figs/choverlap/{river}_choverlap.svg'))

## plot floodplain reworking data

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
    cr50 = fits['CR'][1] #[1/T]
    pr50 = fits['PR'][1]/1e6 #[m2]
    
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
    # plt.savefig(f'{out}/figs/fprework/{river}_fprework.svg')



#%% ## calculate timescales (Tm and Tr) and save csv 

## normalise Rct and Mct by long term average channel area..to do this, load summary dataframes, get the average and normalise then put everything in a new df that will be saved

# efficiency_constants = pd.DataFrame(columns = ['Tm50_timescale', 'Tr50_timescale'], index = rivers)
# for riv in rivers:
#     df = pd.read_csv(os.path.join(mobility_out, f'{riv}_mobility.csv'), header = 0)
#     efficiency_constants.loc[riv, 'Tr50_timescale'] = df.loc[1, 'T_R']
#     efficiency_constants.loc[riv, 'Tm50_timescale'] = df.loc[1, 'T_M']
# efficiency_constants  = pd.concat([linear_ov_decay, linear_reworking], axis = 0).T
# efficiency_constants['Tr_timescale'] = 1/(efficiency_constants['Rct'])
# efficiency_constants['Tm_timescale'] = 1/(efficiency_constants['Mct'])
# efficiency_constants['efficiency'] = efficiency_constants['Tm50_timescale']/efficiency_constants['Tr50_timescale']
# efficiency_constants.to_csv(os.path.join(out, 'efficiency_constants_gr24.csv'))
# resolution = (30 * 30) / 1e6
# area_csvs = glob.glob('/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/C02_1987-2023_allLS_db_csv/*.csv')

# for csv in area_csvs:
#     name = csv.split('/')[-1].split('_')[0]
#     if name == 'congo':
#         name = 'congo_lukolela_bolobo'
    
#     if name=='agubh2':
#         df = pd.read_csv(csv, header = 0)
        
#     else: 
#         df = pd.read_csv(csv, header = 0, skiprows = np.arange(1, 13))
        
#     lt_avg_px = df['wet px'].mean(axis = 0)*resolution
    
#     efficiency_constants.loc[name, 'Tr_timescale'] = efficiency_constants.loc[name, 'Rct']/lt_avg_px                             
#%% compare R/M values 

plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

for r, val in enumerate(rivers):
    plt.plot(linear_ov_decay.loc['Mct', val], linear_reworking.loc['Rct', val], lw = 0, 
             marker = 'o', ms = 15, mec = 'k', label = val)
    plt.annotate(val, (linear_ov_decay.loc['Mct', val], linear_reworking.loc['Rct', val]), 
                 textcoords = 'offset points', xytext = (18,10), ha='center')
plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
plt.legend()    
    
plt.ylabel('Linear Floodplain Reworking')
plt.xlabel('Linear Overlap Decay')
    
plt.savefig(f'{mobility_csvs}/mr-ratio.svg')         

#%% colour by relative mobility ratio (R/M)

# rel_mobility = linear_reworking.loc['Rct', :]/linear_ov_decay.loc['Mct', :]  

# plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)


# relmo = plt.scatter(linear_ov_decay.loc['Mct', :], linear_reworking.loc['Rct', :], lw = 0, s = 180,
#              marker = 'o', c = rel_mobility, ec = 'k', label = val, cmap = mycmap, vmin = rel_mobility[1:8].min(), vmax = rel_mobility[1:8].max())
# plt.colorbar(relmo, label = 'R/M')
# plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
# # plt.legend(labelspacing = 1.2)    
    
# plt.ylabel('Linear Floodplain Reworking')
# plt.xlabel('Linear Overlap Decay')
    
# plt.savefig('/Users/safiya/Desktop/colloquium_local/mr-ratio-mrcolourbar-scaledexcludebadfits.svg')     
                                 
#%% colour by qs                                 
                                 
# # rel_mobility_ratio
# qs_bqart_hydrosheds = [0.471143816, 7.269044905, 33.41815743, 104.3370531, 19.56919841, 35.60403182, 0.111580624, 4.924563359, 6.050316584]
# median_area = fp_params.loc['Aw', :].to_numpy()
# qs_div_medarea = (qs_bqart_hydrosheds)/(median_area) ## MT/yr div km2 -> 

# plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

# qsplot = plt.scatter(linear_ov_decay.loc['Mct', :], linear_reworking.loc['Rct',:], c= qs_bqart_hydrosheds, norm = mcol.LogNorm(),
#              marker = 'o', s = 300, ec = 'k')
# plt.colorbar(qsplot, shrink = 0.5, label = 'Sediment Load, MT/yr')
# plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
# # plt.legend(labelspacing = 1.2)    
    
# plt.ylabel('Linear Floodplain Reworking')
# plt.xlabel('Linear Overlap Decay')

# plt.savefig('/Users/safiya/Desktop/colloquium_local/mr-ratio-lognorm-w-colourbar.svg')                                      

#%% Am* vs Te

# Am_expec_change = np.round(1-(channel_params.loc['Pm', :]/channel_params.loc['Aw', :]), 1)
# Tm_efold = np.round(3/channel_params.loc['Cm', :], 1)

# median_max_ptt = [4, 4, 5, 4.5, 4, 4.5, 4, 5, 4] ## manually taken from med_max_ptt , doing mean oveer two batches, in working-plots-Mar25.py
# mean_max_ptt = [4.33432, 4.66243, 0.5*(5.846+4.72848),
#                 0.5*(5.58356+4.55665),
#                 0.5*(4.63+4.505),
#                 0.5*(5.243+4.205),
#                 0.5*(4.89656+4.42204),
#                 0.5*(5.27361+4.83176),
#                 0.5*(5.27193+3.99222)]

# mean_nswitch = [2.8197, 2.46256, 0.5*(2.48813+2.36686),
#                 0.5*(2.36686+1.87208),
#                 0.5*(3.11992+2.18443),
#                 0.5*(3.50281+2.78882),
#                 0.5*(3.76622+2.63136),
#                 0.5*(2.94386+2.19972),
#                 0.5*(3.46071+2.52574)]
# plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

# plot_nswitch = plt.scatter(Tm_efold, Am_expec_change, c= qs_bqart_hydrosheds, norm = mcol.Normalize(vmin = 1, vmax = 3),
#              marker = 'o', s = 300, ec = 'k')
# plt.colorbar(plot_nswitch, shrink = 0.5, label = 'Mean number of turnovers')
# # plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
# plt.legend(labelspacing = 1.2, loc = 'best')    
# plt.yscale('log')
# plt.xscale('log')    

# plt.ylabel('Normalised wetted area expected to change state')
# plt.xlabel('Overlap decay timeschale')


# plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

# plot_ptt = plt.scatter(Tm_efold[1:8], Am_expec_change[1:8], c= qs_bqart_hydrosheds[1:8],
#                        norm = mcol.LogNorm(),
#              marker = 'o', s = 300, ec = 'k')
# plt.colorbar(plot_ptt, shrink = 0.5, label = 'Qs')
# # plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
# plt.legend(labelspacing = 1.2, loc = 'best')    
# # plt.yscale('log')
# # plt.xscale('log')    

# plt.ylabel('Normalised wetted area expected to change state')
# plt.xlabel('Overlap decay timeschale')

# plt.savefig('/Users/safiya/Desktop/colloquium_local/TmAm-qshydro-w-colourbar-lognorm.svg')                                      

#%% load arrays and find the max wettable area in all timesetps. --start here tomorrow!

max_wettable_area = pd.DataFrame(columns = rivers, index = [1])
root = '/Volumes/SAF_Data/remote-data/'
array_path = 'arrays/C02_1987-2023_allLS_db/full/maskstack/'
scale = 900/1e6 ##convert pixel area resolution to km2
fullstack_path = os.path.join(root, array_path)
for name in rivers:
    sum_arr = np.sum(np.load(os.path.join(fullstack_path, f'{name}_fullstack.npy')), axis = 0)
    
    max_wettable_area[name] = np.count_nonzero(sum_arr)*scale
    
#%% normalise channel overlap

for r, river in enumerate(rivers):
    plt.figure(figsize = (4, 3), dpi = 300, tight_layout = True) ##plot all overlap area data on one axis    

    aw50 = channel_params.iloc[2, r]
    pm50 = channel_params.iloc[1, r]
    cm50 = channel_params.iloc[0, r]
    
    dt = np.arange(0, len(full_dataset[full_dataset['river']==river]))
    ami = ((aw50-pm50) * np.exp(-1*cm50*dt) + pm50)/max_wettable_area.iloc[0, r]
    
    plt.plot(full_dataset[full_dataset['river']==river]['tbase'], 
             full_dataset[full_dataset['river']==river]['median']/max_wettable_area.iloc[0, r],
             lw = 0, ms = 5, label = river)
    
    plt.fill_between(full_dataset[full_dataset['river']==river]['tbase'].astype(int), 
                     full_dataset[full_dataset['river']==river]['Q1']/max_wettable_area.iloc[0, r], 
                     full_dataset[full_dataset['river']==river]['Q3']/max_wettable_area.iloc[0, r],
                     alpha = 0.25, label = river)
   
    plt.plot(dt, ami, zorder = 10000)    
    plt.title(f'{river}, 3e = {np.round(3/cm50, 1)}, A$_m$* = {np.round(1-(pm50/aw50), 1)}')
    plt.xlabel('Duration since baseline')
    plt.ylabel('Channel overlap area (norm.)')   
    plt.ylim(0, 1) 

    plt.savefig(os.path.join(out, 'figs', 'choverlap', f'choverlap-{river}-norm.svg'))          

#%% normalise fp reworking

for r, river in enumerate(rivers):
    plt.figure(figsize = (4, 3), dpi = 300, tight_layout = True) ##plot all overlap area data on one axis    
    
    aw50 = fp_params.iloc[2, r]
    pr50 = fp_params.iloc[1, r]
    cr50 = fp_params.iloc[0, r]
    
    plt.plot(fr_dataset[fr_dataset['river']==river]['tbase'], 
             fr_dataset[fr_dataset['river']==river]['median']/max_wettable_area.iloc[0, r],
             lw = 0, ms = 5, label = river)
    
    plt.fill_between(fr_dataset[fr_dataset['river']==river]['tbase'].astype(int), 
                     fr_dataset[fr_dataset['river']==river]['Q1']/max_wettable_area.iloc[0, r], 
                     fr_dataset[fr_dataset['river']==river]['Q3']/max_wettable_area.iloc[0, r],
                     alpha = 0.25, label = river)
    dt = np.arange(0, len(fr_dataset[fr_dataset['river']==river]))
    ari = ((-1*pr50) * np.exp(-1*cr50*dt) + pr50)/max_wettable_area.iloc[0, r]
    
    plt.plot(dt, ari, zorder = 100)
    
    plt.xlabel('Duration since baseline')
    plt.ylabel('Floodplain reworking area (norm.)')   
    plt.title(f'{river}, 3e = {np.round(3/cr50, 1)}, \n A$_r$* = {np.round(pr50/aw50, 1)}')
    plt.ylim(0, 1) 

    plt.savefig(os.path.join(out, 'figs', 'fprework', f'fprework-{river}-norm.svg'))  

#%% build colourmaps

cmaplist = [mycmap(i) for i in range(mycmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)

mellow_rainbow_cmap = mcol.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, mycmap.N)

years_with_imagery = np.arange(1999, 2024)

bounds = np.arange(0, len(years_with_imagery))#np.linspace(0, len(years_with_imagery), len(years_with_imagery))
norm = mcol.BoundaryNorm(bounds, mellow_rainbow_cmap.N)
#%% plot waterpalette maps for all times

# blues_full = ListedColormap(plt.cm.Blues(np.linspace(0, 1, len(years_with_imagery))))

# # time_cm = ListedColormap(mycmap(np.linspace(0, 1, years_with_imagery)))
# batch_path_root = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack'

# # def extract_year(name):
# #     parts = name.split('_')
# #         parts = name.split('_')
# #         if len(parts) > 2 and parts[-1] == 'mobility.csv':
# #             return '_'.join(parts[:-2])
# # age_by = 13
# rivers = ['congo_destriped_fullstack']

# for river in rivers:
#     # print(river)
#     for nm in glob.glob(os.path.join(batch_path_root, f'{river}*.npy')):
#         # print(nm)
#         riv = np.load(nm)[12:, :, :]
#         water_palette = copy.deepcopy(riv[0, :, :])
        
#         water_palette[water_palette == 0] = -999 
#         water_palette[water_palette==1] = 0
    
#         for ts in range(1, len(riv)):
#              mix = water_palette + riv[ts, :, :] 
#              water_palette[mix==-998] = ts 
             
#         water_palette_mask = ma.masked_equal(water_palette, -999)
    
#         plt.figure(figsize = (5, 5), tight_layout = True, dpi = 400)
#         plt.imshow(water_palette_mask, cmap = mellow_rainbow_cmap, vmin = 0, vmax = len(years_with_imagery))
#         plt.title(river)
#         plt.colorbar(label = 'Time pixel went wet')
#         ax = plt.gca()
#         ax.set_facecolor('xkcd:forest green')
#         plt.savefig(f'/Users/safiya/Desktop/c2_local_exploration/water_palette_may24/{river}_1999_palmap_blue_allLS.svg')       
# #     #%%

# for nm in arrnames:
#     if os.path.exists(f'{batch_path_root}{nm}_fullstack_pal_b1.npy'):
#         b1 = np.load(os.path.join(f'/Volumes/SAF_Data/remote-data/arrays/C02_SR_arrays_wl2_clipped/full/maskstack/{nm}_fullstack.npy'))[12:, :, :]
#         water_palette = copy.deepcopy(b1[0, :, :])

#         water_palette[water_palette == 0] = -999 
#         water_palette[water_palette==1] = 0

#         for ts in range(1, len(b1)):
#              mix = water_palette + b1[ts, :, :] 
#              water_palette[mix==-998] = ts 
             
#         water_palette_mask = ma.masked_equal(water_palette, -999)
    
#         plt.figure(figsize = (10, 10), tight_layout = True, dpi = 400)
#         # plt.imshow(water_palette_mask, cmap = mellow_rainbow_cmap, norm = norm)
#         plt.imshow(water_palette_mask, cmap = blues_full, vmin = 0, vmax = len(years_with_imagery))
        
#         plt.colorbar(label = 'Time pixel wennt wet')
#         plt.savefig(f'/Users/safiya/Desktop/colloquium_local/{nm}_palmap_blue.svg')       
#     else:
#         pal = np.load(f'{batch_path_root}{nm}_fullstack_pal_b2.npy')[-1, :, :]
#         ## we want all ages in batch 2 to increase by 13 (we start at 1999 in batch 1)
#         pal[pal>=0] = pal[pal>=0] + age_by
        
#         palmap_mask = ma.masked_where(pal<0, pal)
#         plt.figure(figsize = (10, 10), tight_layout = True, dpi = 400)
#         # plt.imshow(palmap_mask, cmap = mellow_rainbow_cmap, norm = norm)
#         plt.imshow(palmap_mask, cmap = blues_full, vmin = 0, vmax = len(years_with_imagery))
#         plt.colorbar(label = 'Time pixel wennt wet')
#         plt.savefig(f'/Users/safiya/Desktop/colloquium_local/{nm}_palmap_blue.svg')       
        
 




















              