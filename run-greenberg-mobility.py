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
import natsort
import os
from xml.dom import minidom
import numpy.ma as ma
import matplotlib.colors as mcol
from matplotlib.colors import ListedColormap
'''Script to pull the parts of evan's code that generate csv files of mobility data for all rivers,
    then the csvs need to be sorted by i and plot the average of each i value'''
    
    
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 14}

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
#%%

river = input('which river? ')


root = f'/Volumes/SAF_Data/remote-data/watermasks/{data_folder}/{river}'
# root = f'/Volumes/SAF_Data/remote-data/watermasks/C02SR_Batch_WL2_Mar25_masks_clipped_nometadata/{river}'
# root = f'/Volumes/SAF_Data/remote-data/watermasks/agubh2'
# root = '/Volumes/SAF_Data/CHAPTER2/greenberg-area-data/full-dataset/Greenberg_AreaMobility_Data/NaturalRiverData/SSaskatchOutlook'
paths = get_paths(root)

poly = f"/Volumes/SAF_Data/remote-data/watermasks/gpkgs_fnl/{river}.gpkg"
# poly = "/Volumes/SAF_Data/CHAPTER2/greenberg-area-data/full-dataset/Greenberg_AreaMobility_Data/NaturalRiverData/SSaskatchOutlook/SSaskatchOutlook.gpkg"

mobility_csvs = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/may2024'
out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/may2024/fit_stats' # blocked fit_stats outputs
mobility_out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/may2024/mobility'

get_mobility_rivers(poly, paths, river) ## gets mobility yearly, makes a csv in the root folder of mobility data

stop = 30

allcsvs = glob.glob(os.path.join(mobility_csvs, '*.csv')) 
path_list = natsort.natsorted(allcsvs)



get_stats(os.path.join(mobility_csvs, f'{river}_yearly_mobility.csv'), river, out, 30)
get_mobility(pd.read_csv(os.path.join(out, f'{river}_fit_stats.csv'), header = 0),
              os.path.join(mobility_out, f'{river}_mobility.csv'))
#%% load csvs and make plots

# river = 'congo_lukolela_bolobo'
# congo = pd.read_csv(os.path.join(out,f'{river}_yearly_mobility.csv'), header = 0, index_col=0)

# avgs = np.array([])
# ivals = congo['i'].unique()
# for i in ivals:
#     avgs = np.append(avgs, np.mean(congo['O_avg'][congo['i']==i])/1e6)

## create raw xy data for eq4
out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs'

full_dataset = pd.DataFrame(columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
full_dataset_wick = pd.DataFrame(columns = ['tbase', 'river', 'min', 'median', 'mean', 'max', 'Q1', 'Q3'])
for file in glob.glob(os.path.join(out, '*.csv')):
    riv = (file.split("/")[-1]).split('_')[0]

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
for file in glob.glob(os.path.join(out, '*.csv')):
    riv = (file.split("/")[-1]).split('_')[0]

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

#%%

# def func_m_param(x, m, p, aw):
#     return ((aw - p) * np.exp(-m * x)) + p

# plot wickert normalised data
# ssk = full_dataset_wick.loc[full_dataset['river'] == 'southsask']      
# yuk = full_dataset_wick.loc[full_dataset['river'] == 'yukon']      
# tan = full_dataset_wick.loc[full_dataset['river'] == 'tanana']      
# con = full_dataset_wick.loc[full_dataset['river'] == 'congo']      
# col = full_dataset_wick.loc[full_dataset['river'] == 'colville']      
# ira = full_dataset_wick.loc[full_dataset['river'] == 'irrawaddy']      
# rak = full_dataset_wick.loc[full_dataset['river'] == 'rakaia']      
# kas = full_dataset_wick.loc[full_dataset['river'] == 'kasai']   
# ind = full_dataset_wick.loc[full_dataset['river'] == 'IndusR2'] 

ssk = full_dataset.loc[full_dataset['river'] == 'southsask']      
yuk = full_dataset.loc[full_dataset['river'] == 'yukon']      
tan = full_dataset.loc[full_dataset['river'] == 'tanana']      
con = full_dataset.loc[full_dataset['river'] == 'congo']      
col = full_dataset.loc[full_dataset['river'] == 'colville']      
ira = full_dataset.loc[full_dataset['river'] == 'irrawaddy']      
rak = full_dataset.loc[full_dataset['river'] == 'rakaia']      
kas = full_dataset.loc[full_dataset['river'] == 'kasai']   
ind = full_dataset.loc[full_dataset['river'] == 'IndusR2'] 


frssk = fr_dataset.loc[fr_dataset['river'] == 'southsask']      
fryuk = fr_dataset.loc[fr_dataset['river'] == 'yukon']      
frtan = fr_dataset.loc[fr_dataset['river'] == 'tanana']      
frcon = fr_dataset.loc[fr_dataset['river'] == 'congo']      
frcol = fr_dataset.loc[fr_dataset['river'] == 'colville']      
frira = fr_dataset.loc[fr_dataset['river'] == 'irrawaddy']      
frrak = fr_dataset.loc[fr_dataset['river'] == 'rakaia']      
frkas = fr_dataset.loc[fr_dataset['river'] == 'kasai']   
frind = fr_dataset.loc[fr_dataset['river'] == 'IndusR2']      

df_list = [col, con, ind, ira, kas, rak, ssk, tan, yuk]
df_names = ['colville', 'congo', 'indusr2', 'irrawaddy', 'kasai', 'rakaia', 'southsask', 'tanana', 'yukon']

kg_classes = ['ET', 'Af', 'BWh', 'BSh', 'Aw', 'Cfb', 'Dfb', 'Dfb', 'Dsb']
kg_colours = [(0.702, 0.702, 0.702),
              (0, 0, 1),
              (1, 0, 0),
              (0.961, 0.647, 0), 
              (0.275, 0.667, 0.99), 
              (0.392, 1, 0.196),
              (0.216, 0.784, 1),
              (0.216, 0.784, 1),
              (0.784, 0, 0.784)] ## in order of kg_classes! do not reorder

lss = ['-', '-', '-', '-', '-', '-', '-', '--', '-']
  
plt.figure(1, figsize = (12,10), dpi = 300, tight_layout = True)

params = ['Cm', 'Pm', 'Aw']
channel_params = pd.DataFrame(index = params, columns = df_names)


for d, dframe in enumerate(df_list):
    plt.plot(dframe['tbase'], dframe['median'], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    plt.fill_between(dframe['tbase'].astype(int), dframe['Q1'], dframe['Q3'], color = kg_colours[d], alpha = .25, ec = None) 
    
    fits = pd.read_csv(os.path.join(out, f'fit_stats/{df_names[d]}_fit_stats.csv'), header = 0, index_col = 0 )
    
    aw50 = fits['Aw'][:]/1e6 #[km2]
    cm50 = fits['CM'][1] #[1/T]
    pm50 = fits['PM'][1]/1e6 #[m2]
    
    channel_params.loc[:, df_names[d]] = [cm50, pm50, aw50]
    dt = np.arange(0, len(dframe))
    
    ami = ((aw50-pm50) * np.exp(-1*cm50*dt) + pm50)
    
    # plt.plot(dt, ami, color = kg_colours[d], zorder = 10000)

plt.yscale('log')
plt.legend()
plt.ylabel('Channel overlap area, km2');
plt.xlabel('Duration since baseline');
plt.title('channel overlap area, $A_{Mi}$ = channel area in the ith channel mask \n that did not change state when compared to the baseline channel mask');

plt.figure(2, figsize = (12,10), dpi = 300, tight_layout = True)
fpdf_list = [frcol, frcon, frind, frira, frkas, frrak, frssk, frtan, fryuk]
fr_params = ['Cr', 'Pr', 'Aw']
fp_params = pd.DataFrame(index = fr_params, columns = df_names)

for d, dframe in enumerate(fpdf_list):
    plt.plot(dframe['tbase'], dframe['median'], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    plt.fill_between(dframe['tbase'].astype(int), dframe['Q1'], dframe['Q3'], color = kg_colours[d], alpha = .25, ec = None) 
    
    fits = pd.read_csv(os.path.join(out, f'fit_stats/{df_names[d]}_fit_stats.csv'), header = 0, index_col = 0 )
    
    aw50 = fits['Aw'][1]/1e6 #[m2]
    cr50 = fits['CR'][1] #[1/T]
    pr50 = fits['PR'][1]/1e6 #[m2]
    
    fp_params.loc[:, df_names[d]] = [cr50, pr50, aw50]
    dt = np.arange(0, len(dframe))
    
    ari = (-1*pr50) * np.exp(-1*cr50*dt) + pr50
    
    # plt.plot(dt, ami, color = kg_colours[d], zorder = 10000)

plt.yscale('log')
plt.legend()
plt.ylabel('Channel overlap area, km2');
plt.xlabel('Duration since baseline');
plt.title('channel overlap area, $A_{Ri}$ = comolative reworked fp area')


#%%

fig, ax = plt.subplots(3, 3, figsize = (12,10), dpi = 300, tight_layout = True)

a = ax.ravel()

linear_reworking = pd.DataFrame(columns = df_names, index = ['Rct'])
linear_ov_decay = pd.DataFrame(columns = df_names, index = ['Mct'])

for d, dframe in enumerate(df_list):
    a[d].plot(dframe['tbase'], dframe['median'], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    a[d].fill_between(dframe['tbase'].astype(int), dframe['Q1'], dframe['Q3'], color = kg_colours[d], alpha = .25, ec = None) 
    
    fits = pd.read_csv(os.path.join(out, f'fit_stats/{df_names[d]}_fit_stats.csv'), header = 0, index_col = 0 )
    
    aw50 = fits['Aw'][1]/1e6 #[km2]
    cm50 = fits['CM'][1] #[1/T]
    pm50 = fits['PM'][1]/1e6 #km2
     
    # print(df_names[d], [aw50, cm50, pm50])
    channel_params.loc[:, df_names[d]] = [cm50, pm50, aw50]
    dt = np.arange(0, len(dframe))
    
    ami = ((aw50-pm50) * np.exp(-1*cm50*dt) + pm50)
    
    Mct = cm50*(1-(pm50/aw50))
    linear_ov_decay.loc['Mct', df_names[d]] = Mct
    
    a[d].plot(dt, ami, color = kg_colours[d], zorder = 10000)

# plt.yscale('log')
# plt.legend()
    a[d].set_ylabel('Channel overlap area, km2');
    a[d].set_xlabel('Duration since baseline');
    a[d].set_title(f'{df_names[d]}, 3e = {np.round(3/cm50, 1)} avg across ptiles \n norm long term decay area A$_m$* = {np.round(1-(pm50/aw50), 1)}')
# plt.savefig('/Users/safiya/Desktop/colloquium_local/ch_overlap.svg')            
    
## pplot floodplain reworking    
fig, ax = plt.subplots(3, 3, figsize = (12,10), dpi = 300, tight_layout = True)

a = ax.ravel()

for d, dframe in enumerate(fpdf_list):
    a[d].plot(dframe['tbase'], dframe['median'], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    a[d].fill_between(dframe['tbase'].astype(int), dframe['Q1'], dframe['Q3'], color = kg_colours[d], alpha = .25, ec = None) 
    
    fits = pd.read_csv(os.path.join(out, f'fit_stats/{df_names[d]}_fit_stats.csv'), header = 0, index_col = 0 )
    
    aw50 = fits['Aw'][1]/1e6 #[km2]
    cr50 = fits['CR'][1] #[1/T]
    pr50 = fits['PR'][1]/1e6 #[km2]
       
    fp_params.loc[:, df_names[d]] = [cr50, pr50, aw50]
    dt = np.arange(0, len(dframe))
       
    ari = (-1*pr50) * np.exp(-1*cr50*dt) + pr50
    
    Rct = cr50*pr50/aw50
    linear_reworking.loc['Rct', df_names[d]] = Rct
    
    a[d].plot(dt, ari, color = kg_colours[d], zorder = 10000)

# plt.yscale('log')
# plt.legend()
    a[d].set_ylabel('Floodplain reworking area, km2');
    a[d].set_xlabel('Duration since baseline');
    a[d].set_title(f'{df_names[d]}, 3e = {np.round(3/cr50, 1)}, \n norm fp area A$_r$* = {np.round(pr50/aw50, 1)}')
    
# plt.savefig('/Users/safiya/Desktop/colloquium_local/fp_reworking.svg')        
                                 
#%% compare R/M values 

plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

for r, val in enumerate(df_names):
    plt.plot(linear_ov_decay.loc['Mct', val], linear_reworking.loc['Rct', val], lw = 0, 
             marker = 'o', ms = 15, mfc = kg_colours[r], mec = 'k', label = val)
plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
plt.legend(labelspacing = 1.2)    
    
plt.ylabel('Linear Floodplain Reworking')
plt.xlabel('Linear Overlap Decay')
    
plt.savefig('/Users/safiya/Desktop/colloquium_local/mr-ratio.svg')         

#%%
# get relative mobility ratio (R/M)

rel_mobility = linear_reworking.loc['Rct', :]/linear_ov_decay.loc['Mct', :]  

plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)


relmo = plt.scatter(linear_ov_decay.loc['Mct', :], linear_reworking.loc['Rct', :], lw = 0, s = 180,
             marker = 'o', c = rel_mobility, ec = 'k', label = val, cmap = mycmap, vmin = rel_mobility[1:8].min(), vmax = rel_mobility[1:8].max())
plt.colorbar(relmo, label = 'R/M')
plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
# plt.legend(labelspacing = 1.2)    
    
plt.ylabel('Linear Floodplain Reworking')
plt.xlabel('Linear Overlap Decay')
    
plt.savefig('/Users/safiya/Desktop/colloquium_local/mr-ratio-mrcolourbar-scaledexcludebadfits.svg')     
                                 
#%%                                  
                                 
rel_mobility_ratio
qs_bqart_hydrosheds = [0.471143816, 7.269044905, 33.41815743, 104.3370531, 19.56919841, 35.60403182, 0.111580624, 4.924563359, 6.050316584]
median_area = fp_params.loc['Aw', :].to_numpy()
qs_div_medarea = (qs_bqart_hydrosheds)/(median_area) ## MT/yr div km2 -> 

plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

qsplot = plt.scatter(linear_ov_decay.loc['Mct', :], linear_reworking.loc['Rct',:], c= qs_bqart_hydrosheds, norm = mcol.LogNorm(),
             marker = 'o', s = 300, ec = 'k')
plt.colorbar(qsplot, shrink = 0.5, label = 'Sediment Load, MT/yr')
plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
# plt.legend(labelspacing = 1.2)    
    
plt.ylabel('Linear Floodplain Reworking')
plt.xlabel('Linear Overlap Decay')

plt.savefig('/Users/safiya/Desktop/colloquium_local/mr-ratio-lognorm-w-colourbar.svg')                                      

#%% Am* vs Te

Am_expec_change = np.round(1-(channel_params.loc['Pm', :]/channel_params.loc['Aw', :]), 1)
Tm_efold = np.round(3/channel_params.loc['Cm', :], 1)

median_max_ptt = [4, 4, 5, 4.5, 4, 4.5, 4, 5, 4] ## manually taken from med_max_ptt , doing mean oveer two batches, in working-plots-Mar25.py
mean_max_ptt = [4.33432, 4.66243, 0.5*(5.846+4.72848),
                0.5*(5.58356+4.55665),
                0.5*(4.63+4.505),
                0.5*(5.243+4.205),
                0.5*(4.89656+4.42204),
                0.5*(5.27361+4.83176),
                0.5*(5.27193+3.99222)]

mean_nswitch = [2.8197, 2.46256, 0.5*(2.48813+2.36686),
                0.5*(2.36686+1.87208),
                0.5*(3.11992+2.18443),
                0.5*(3.50281+2.78882),
                0.5*(3.76622+2.63136),
                0.5*(2.94386+2.19972),
                0.5*(3.46071+2.52574)]
plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

plot_nswitch = plt.scatter(Tm_efold, Am_expec_change, c= qs_bqart_hydrosheds, norm = mcol.Normalize(vmin = 1, vmax = 3),
             marker = 'o', s = 300, ec = 'k')
plt.colorbar(plot_nswitch, shrink = 0.5, label = 'Mean number of turnovers')
# plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
plt.legend(labelspacing = 1.2, loc = 'best')    
plt.yscale('log')
plt.xscale('log')    

plt.ylabel('Normalised wetted area expected to change state')
plt.xlabel('Overlap decay timeschale')


plt.figure(figsize = (8, 8), dpi = 300, tight_layout = True)

plot_ptt = plt.scatter(Tm_efold[1:8], Am_expec_change[1:8], c= qs_bqart_hydrosheds[1:8],
                       norm = mcol.LogNorm(),
             marker = 'o', s = 300, ec = 'k')
plt.colorbar(plot_ptt, shrink = 0.5, label = 'Qs')
# plt.axline((0, 0), slope = 1, linewidth = 1, color = 'grey')
plt.legend(labelspacing = 1.2, loc = 'best')    
# plt.yscale('log')
# plt.xscale('log')    

plt.ylabel('Normalised wetted area expected to change state')
plt.xlabel('Overlap decay timeschale')

plt.savefig('/Users/safiya/Desktop/colloquium_local/TmAm-qshydro-w-colourbar-lognorm.svg')                                      

#%% load arrays and find the max wettable area in all timesetps. 


arrnames = ['colville',
 'congo_lukolela_bolobo',
 'indus-r2',
 'irrawaddy',
 'kasai',
 'rakaia',
 'southsask',
 'tanana',
 'yukon']
max_wettable_area = pd.DataFrame(columns = arrnames, index = [1])
root = '/Volumes/SAF_Data/remote-data/'
array_path = 'arrays/C02_SR_arrays_wl2_clipped/full/maskstack/'
scale = 900/1e6 ##convert pixel area resolution to km2
fullstack_path = os.path.join(root, array_path)
for name in arrnames:
    sum_arr = np.sum(np.load(os.path.join(fullstack_path, f'{name}_fullstack.npy')), axis = 0)
    
    max_wettable_area[name] = np.count_nonzero(sum_arr)*scale
    
#%% rescale plots to 1:1

fig, ax = plt.subplots(3, 3, figsize = (18, 15), dpi = 500, tight_layout = True, sharey = True, sharex = True)
ax[0, 0].set_ylim(0, .95)

a = ax.ravel()
for d, dframe in enumerate(df_list):
    
    aw50 = channel_params.iloc[2, d]
    pm50 = channel_params.iloc[1, d]
    cm50 = channel_params.iloc[0, d]
    
    dt = np.arange(0, len(dframe))
    ami = ((aw50-pm50) * np.exp(-1*cm50*dt) + pm50)/max_wettable_area.iloc[0, d]
   
    a[d].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    a[d].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None) 
    a[d].plot(dt, ami, color = kg_colours[d], zorder = 10000)    
    
    # if df_names[d] == 'rakaia' or df_names[d] == 'indusr2'or df_names[d] == 'irrawaddy' or df_names[d] == 'tanana'or df_names[d] == 'kasai': 
    #     ax[1].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    #     ax[1].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None)     
    #     ax[1].plot(dt, ami, color = kg_colours[d], zorder = 10000)
    # # elif  or df_names[d] == 'tanana'or df_names[d] == 'kasai': 
    # #     ax[1].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    # #     ax[1].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None)     
    # #     ax[1].plot(dt, ami, color = kg_colours[d], zorder = 10000)
    # else:
    
    #     ax[0].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    #     ax[0].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None)     
    #     ax[0].plot(dt, ami, color = kg_colours[d], zorder = 10000)


    a[d].set_xlabel('Duration since baseline')

    a[d].set_ylabel('Channel overlap area (norm.)')   
# a[d].ylim(0, 1) 

plt.savefig('/Users/safiya/Desktop/colloquium_local/ch_ovlap_norm-subplots.svg')            

#%% 

fig, ax = plt.subplots(3, 3, figsize = (18, 15), dpi = 500, tight_layout = True, sharey = True, sharex = True)
ax[0, 0].set_ylim(0, .95)
a = ax.ravel()

for d, dframe in enumerate(fpdf_list):
    
    aw50 = fp_params.iloc[2, d]
    pr50 = fp_params.iloc[1, d]
    cr50 = fp_params.iloc[0, d]
    
    dt = np.arange(0, len(dframe))
   
    a[d].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    a[d].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None) 
        
    
    # if df_names[d] == 'rakaia' or df_names[d] == 'indusr2'or df_names[d] == 'irrawaddy' or df_names[d] == 'tanana'or df_names[d] == 'kasai': 
    #     ax[1].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    #     ax[1].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None)     
    #     ax[1].plot(dt, ami, color = kg_colours[d], zorder = 10000)
    # # elif  or df_names[d] == 'tanana'or df_names[d] == 'kasai': 
    # #     ax[1].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    # #     ax[1].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None)     
    # #     ax[1].plot(dt, ami, color = kg_colours[d], zorder = 10000)
    # else:
    
    #     ax[0].plot(dframe['tbase'], dframe['median']/max_wettable_area.iloc[0, d], marker = 'o', mfc = kg_colours[d], mec = 'k', ls = lss[d], lw = 2, label = df_names[d], c = kg_colours[d])
    #     ax[0].fill_between(dframe['tbase'].astype(int), dframe['Q1']/max_wettable_area.iloc[0, d], dframe['Q3']/max_wettable_area.iloc[0, d], color = kg_colours[d], alpha = .25, ec = None)     
    #     ax[0].plot(dt, ami, color = kg_colours[d], zorder = 10000)

    ari = ((-1*pr50) * np.exp(-1*cr50*dt) + pr50)/max_wettable_area.iloc[0, d]
    
    a[d].plot(dt, ari, color = kg_colours[d], zorder = 10000)
    a[d].set_xlabel('Duration since baseline')

    a[d].set_ylabel('Floodplain reworking area (norm.)')   
# a[d].ylim(0, 1) 

plt.savefig('/Users/safiya/Desktop/colloquium_local/fprework_norm-subplots.svg')  

#%%

cmaplist = [mycmap(i) for i in range(mycmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)

mellow_rainbow_cmap = mcol.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, mycmap.N)

years_with_imagery = np.arange(1999, 2023)

bounds = np.arange(0, len(years_with_imagery))#np.linspace(0, len(years_with_imagery), len(years_with_imagery))
norm = mcol.BoundaryNorm(bounds, mellow_rainbow_cmap.N)
#%% plot waterpalette maps for all times

blues_full = ListedColormap(plt.cm.PuBu_r(np.linspace(0, 1, len(years_with_imagery))))

# time_cm = ListedColormap(mycmap(np.linspace(0, 1, years_with_imagery)))
batch_path_root = os.path.join(f'/Volumes/SAF_Data/remote-data/arrays/C02_SR_arrays_wl2_clipped/full/maskstack/batches/')
age_by = 13

 
for nm in arrnames:
    if os.path.exists(f'{batch_path_root}{nm}_fullstack_pal_b1.npy'):
        b1 = np.load(os.path.join(f'/Volumes/SAF_Data/remote-data/arrays/C02_SR_arrays_wl2_clipped/full/maskstack/{nm}_fullstack.npy'))[12:, :, :]
        water_palette = copy.deepcopy(b1[0, :, :])

        water_palette[water_palette == 0] = -999 
        water_palette[water_palette==1] = 0

        for ts in range(1, len(b1)):
             mix = water_palette + b1[ts, :, :] 
             water_palette[mix==-998] = ts 
             
        water_palette_mask = ma.masked_equal(water_palette, -999)
    
        plt.figure(figsize = (10, 10), tight_layout = True, dpi = 400)
        # plt.imshow(water_palette_mask, cmap = mellow_rainbow_cmap, norm = norm)
        plt.imshow(water_palette_mask, cmap = blues_full, vmin = 0, vmax = len(years_with_imagery))
        
        plt.colorbar(label = 'Time pixel wennt wet')
        plt.savefig(f'/Users/safiya/Desktop/colloquium_local/{nm}_palmap_blue.svg')       
    else:
        pal = np.load(f'{batch_path_root}{nm}_fullstack_pal_b2.npy')[-1, :, :]
        ## we want all ages in batch 2 to increase by 13 (we start at 1999 in batch 1)
        pal[pal>=0] = pal[pal>=0] + age_by
        
        palmap_mask = ma.masked_where(pal<0, pal)
        plt.figure(figsize = (10, 10), tight_layout = True, dpi = 400)
        # plt.imshow(palmap_mask, cmap = mellow_rainbow_cmap, norm = norm)
        plt.imshow(palmap_mask, cmap = blues_full, vmin = 0, vmax = len(years_with_imagery))
        plt.colorbar(label = 'Time pixel wennt wet')
        plt.savefig(f'/Users/safiya/Desktop/colloquium_local/{nm}_palmap_blue.svg')       
        
        





























              