#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:38:47 2024

@author: safiya
"""
import numpy as np
import pandas as pd
import copy
import os
import whackamole
import matplotlib as mpl
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)

#%% imports

root = '/Volumes/SAF_Data/remote-data/'
dataframe_data = 'watermasks/admin/mask_database_csv/C02_SR_wl2_clipped_db_csv/'

masterdf = pd.read_excel(os.path.join(root, dataframe_data, 'c02sr_batch_wl2_clipped_testdataset_master.xlsx'), header = 1, skiprows = 1, index_col = 0)

kginuse = pd.read_excel(os.path.join(root, dataframe_data, 'c02sr_batch_wl2_clipped_testdataset_master.xlsx'), 
                        nrows = 1, header = 0).T
## cant figure out how to get the KG tuples in
# kginuse.drop(kginuse.loc[0, :], axis = 0)

# kginuse.set_index('Code')

## remove kasai_down
river_names = masterdf.columns.drop(['kasai_down', 'agubh2']) 
river_names
masterdf = masterdf.drop(['kasai_down'], axis = 1)

colours = pd.read_csv('/Volumes/SAF_Data/remote-data/kgcolours.csv', header = 0, index_col=0) ## load library of all KG colours

# kg_plot_colours = kginuse.merge(colours, how = 'left', on = 'Code', copy = False) ## filter for only the kG colours for the rivers being in this group

t_init = 1987
t_start = 2013
t_end = 2022
years = np.arange(t_start, t_end+1)
allyears = np.arange(t_init, t_end+1)
levels = len(years) ## define number of levels for the colormap
full_levels = len(allyears)
colors = plt.cm.rainbow(np.linspace(0, 1, levels))
rainbow_discrete = ListedColormap(colors)
rainbow_full = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, full_levels)))

#%% in a loop, find the maximum pixel area permissible and put it in a dataframe?

'''making a master dataframe that has the maximum potential area if we changed the baseline to be each year in the timeseries, 
for each river in the dataset'''

# areaframe = pd.DataFrame(index = river_names, columns = allyears)
array_path = 'arrays/C02_SR_arrays_wl2_clipped/full/maskstack/'
# stacklist = glob.glob(os.path.join(root, array_path, '*.npy'))


# for item in stacklist:
#     stack = item.split('/')[-1] #get file name from the path

#     if stack != 'kasai_down_fullstack.npy' or stack != 'agubh2_fullstack.npy':

#         riv = stack.split('_')[0]
#         print(riv)
#         arr = np.load(os.path.join(root, array_path, stack))
                
#         for yr in reversed(range(arr.shape[0])):  ## looking backwards because we want to know the effects of increasing the temporal resolutio of the dataset BACK in time
        
#             collapse = np.sum(arr[yr:, :, :], axis = 0) ## collapse all wet pixels into a photo, we just want to count everything
#             wet_px = np.count_nonzero(collapse) ## count the number of nonzero pixels in every collapsed iteration
    
            
#             areaframe.loc[riv, allyears[yr]] = wet_px
# areaframe.loc['congo_lukolela_bolobo'] = areaframe.loc['congo']        
# areaframe = areaframe.drop(['congo'], axis = 0)
# areaframe.to_csv(os.path.join(root, dataframe_data, 'max_wetted_area.csv'))


## if its not loaded:
areaframe = pd.read_csv(os.path.join(root, dataframe_data, 'max_wetted_area.csv'), header = 0, index_col = 0) 

#%%         
''' really the max permissible area should be the main corridor of the river --the maximum width of the main corridor but I can't think of
    how to calculate that right now'''
plt.figure(figsize=(10, 5), dpi = 200)
ax = plt.gca()
areaframeT = areaframe.T 
areaframeT.plot(ax = ax)
        
#%%
''' ok lets try the idea of wworking in two batches: pre 2012 (1999-2011, and 2013-2021). 
With the exception of the Congo (Af), indus (BWh) and Colville (ET) the rest of systems provide enough resolution for both
we have another A (Kasai), another B (Irrawaddy, Yukon technically). Unfortunately no ET :('''
                                                                                           
b1_years = np.arange(1999, 2012)
b2_years = np.arange(2013, t_end+1)

batch1 = river_names[2:]
batch2 = copy.deepcopy(river_names)

## uncomment if you want to calculate water_palette for new batches, if else, np.load the batched arrays from within the full/maskstack/batches folder

# def water_palette(batch, batch_time, batch_flag):
#     '''build the arrays that show the time to wet of each pixel in the mask (water palette maps) and generate a dataframe of the number of
#     pixels newly wet in each timestep'''
#     indices = np.where(np.isin(allyears, batch_time))[0] ## find the indices of the allyears array where we need to pull the planes from the fullstack
    
#     print(indices)
#     ttw = pd.DataFrame(index = batch, columns = batch_time) ##dataframe to house the time to wet data
    
#     for riv in batch:
#         print(riv)
#         loadpath = os.path.join(root, array_path)
#         # print(loadpath)
#         arr = np.load(os.path.join(loadpath,f'{riv}_fullstack.npy'))
#         # indices_neg = indices-len(allyears)    ## quick fix for now, should work well going backwards from the end year) 
#         #crop the arr
#         arr = arr[indices, :, :]
#         # print(arr.shape)
        
#         palette = copy.deepcopy(arr[0, :, :]) # copy the initial plane
#         palette[palette == 0] = -999
#         palette[palette == 1] = 0 # give the initial water px at time t = 0 an age = 0
        
#         ttw.loc[riv, batch_time[0]] = np.count_nonzero(palette>=0)
#         # print('initial wet px count: ', np.count_nonzero(palette==0))
        
#         for ts in range(1, len(arr)):
#             mix = palette + arr[ts, :, :]
#             palette[mix == -998] = ts 
            
#             area = np.count_nonzero(palette >= 0) ## find the area of the image at that time that is wet pixels
#             time = batch_time[0]+ts
#             ttw.loc[riv, time] = area
            
#             # print(f'px area for {riv} at {ts} timestep, i.e. {time} = {area}')
#         pal = np.expand_dims(palette, axis = 0)
#         ttwarr = np.concatenate((arr, pal), axis = 0)
#         if batch_flag == 1:
#             np.save(os.path.join(loadpath, 'batches', f'{riv}_fullstack_pal_b1.npy'), ttwarr)
#         elif batch_flag == 2:
#             np.save(os.path.join(loadpath, 'batches', f'{riv}_fullstack_pal_b2.npy'), ttwarr)
#         else:
#             classifier = input('add ID flag for batch')
#             np.save(os.path.join(loadpath, 'batches',f'{riv}_fullstack_pal_{classifier}.npy'), ttwarr)
#     return ttw
        
# b2ttw = water_palette(batch2, b2_years, batch_flag = 2)
# # b2ttw = b2ttw.T
# b1ttw = water_palette(batch1, b1_years, batch_flag = 1)    
# # b1ttw = b1ttw.T
        
#%% 
# ## using batch 2 max areas and plotting for batch 2:
    
# batch2_ttw_norm = copy.deepcopy(b2ttw)
# batch2_ttw_norm['MaxArea_2013on'] = areaframe['2013']

# ## normalise to max permissible area with a 2013 baseline
# batch2_ttw_norm.iloc[:,:-1] = batch2_ttw_norm.iloc[:,:-1].div(batch2_ttw_norm.MaxArea_2013on, axis=0)   
# batch2_ttw_norm = batch2_ttw_norm.T ## because pandas is stupid and you have to reorient the df so each river is in a different column

# plt.figure(figsize = (10, 6), dpi = 150, tight_layout = True)
# ax = plt.gca()
# batch2_ttw_norm.iloc[:-1, :].plot(ax = ax)
# ax.set_ylabel('New wet px area/max potential wet px')
# ax.set_xlabel('Time')    
        
# ##plot batch 1--this is wrong bc it takes the available area from 2013-2021 into account too 
# batch1_ttw_norm = copy.deepcopy(b1ttw)
# batch1_ttw_norm['MaxArea_1999on'] = areaframe['1999'] 

# ## normalise to max permissible area with a 2013 baseline
# batch1_ttw_norm.iloc[:,:-1] = batch1_ttw_norm.iloc[:,:-1].div(batch1_ttw_norm.MaxArea_1999on, axis=0)   
# batch1_ttw_norm = batch1_ttw_norm.T ## because pandas is stupid and you have to reorient the df so each river is in a different column

# plt.figure(figsize = (10, 6), dpi = 150, tight_layout = True)
# ax = plt.gca()
# ax.set_ylabel('New wet px area/max potential wet px')
# batch1_ttw_norm.iloc[:-1, :].plot(ax = ax)   
# ax.set_xlabel('Time')        

# ######## trying to plot both on one plot??
# batch1_2021norm = copy.deepcopy(b1ttw)
# batch1_2021norm['MaxArea_1999on'] = areaframe['1999']

# # normalise to max permissible area with a 2013 baseline
# batch1_2021norm.iloc[:,:-1] = batch1_2021norm.iloc[:,:-1].div(batch1_2021norm.MaxArea_1999on, axis=0)   
# batch1_2021norm = batch1_2021norm.T ## because pandas is stupid and you have to reorient the df so each river is in a different column

# fig, ax = plt.subplots(1, 2, figsize = (15, 5), dpi = 150, tight_layout = True)
# batch2_ttw_norm.iloc[:-1, :].plot(ax = ax[1], marker = 'o', ls = '-')
# batch1_2021norm.iloc[:-1, :].plot(ax = ax[0], marker = '*', ls = '--')
# ax[0].set_ylabel('New wet px area/max potential wet px')
# ax[1].set_title('2013-2021 with 2013 to now as baseline')
# ax[0].set_title('1999-2013 with 1999 to now as baseline, prob unsmart')
# ax[1].set_xlabel('Time')    
# ax[0].set_xlabel('Time')  

#%% compute and save ptt arrays

### load maskstack arrays
load_path = os.path.join(root, array_path)
col = np.load(os.path.join(load_path,f'colville_fullstack.npy'))
con = np.load(os.path.join(load_path,f'congo_lukolela_bolobo_fullstack.npy'))
ind = np.load(os.path.join(load_path,f'indus-r2_fullstack.npy'))
ira = np.load(os.path.join(load_path,f'irrawaddy_fullstack.npy'))
kas = np.load(os.path.join(load_path,f'kasai_fullstack.npy'))
rak = np.load(os.path.join(load_path,f'rakaia_fullstack.npy'))
ssk = np.load(os.path.join(load_path,f'southsask_fullstack.npy'))
tan = np.load(os.path.join(load_path,f'tanana_fullstack.npy'))
yuk = np.load(os.path.join(load_path,f'yukon_fullstack.npy'))

## define indices of interest for the ptt calculations

b1_indices = np.where(np.isin(allyears, b1_years))[0]
b2_indices = np.where(np.isin(allyears, b2_years))[0]

b1_ind_d = b1_indices[:-1]
b2_ind_d = b2_indices[:-1]

##make dataframes to store the raveled ptt and flag values for each arrray (this is for ploting bulk distributions only! no spatial context)
# batch1_max_ptt = pd.DataFrame()
# batch2_max_ptt = pd.DataFrame()

# batch1_nswitch = pd.DataFrame()
# batch2_nswitch = pd.DataFrame()

# # batch1_flags = pd.DataFrame()
# # batch2_flags = pd.DataFrame()

ms_list = [col, con, ind, ira, kas, rak, ssk, tan, yuk] ##list of arrays to iterate over, should be same order as batch1_ptt columns!

diff_path = os.path.join(root, 'arrays/C02_SR_arrays_wl2_clipped/full/diffs/')
# for idx, rivname in enumerate(river_names):
#     print(rivname)
    
#     dif = np.load(os.path.join(diff_path,f'{rivname}_diffs_fullstack.npy')) ## load the difference matrix once so you dont have to do it twice
    
#     if len(np.where(np.isin(batch1, rivname))[0])>0:
#         print(f'{rivname} in batch1')
#         msb1 = ms_list[idx][b1_indices, :, :]
#         difb1 = dif[b1_ind_d, :, :]
        
#         print('b1', msb1.shape, difb1.shape)
        
#         flagsb1, pttb1 = pixel_turn_time(msb1, difb1)
#         nswitchb1 = pd.DataFrame(np.count_nonzero(~np.isnan(pttb1), axis = 0).ravel(), columns = [rivname])
#         max_pttb1 = pd.DataFrame(np.nanmax(pttb1, axis = 0).ravel(), columns = [rivname])
#         # flagsb1 = pd.DataFrame(np.nanmax(pttb1, axis = 0).ravel(), columns = [rivname])
        
        
#         batch1_max_ptt = pd.concat((batch1_max_ptt, max_pttb1), axis = 1, ignore_index = False, names = [rivname])
#         batch1_nswitch = pd.concat((batch1_nswitch, nswitchb1), axis = 1, ignore_index = False, names = [rivname])

        

#         # batch1_max_ptt[rivname] = max_pttb1
#         # batch1_nswitch[rivname] = nswitchb1
        
#     if len(np.where(np.isin(batch2, rivname))[0])>0:
#         print(f'{rivname} in batch2')        
#         msb2 = ms_list[idx][b2_indices, :, :]
#         difb2 = dif[b2_ind_d, :, :]
#         print('b2', msb2.shape, difb2.shape)
        
#         flagsb2, pttb2 = pixel_turn_time(msb2, difb2)
        
#         nswitchb2 = pd.DataFrame(np.count_nonzero(~np.isnan(pttb2), axis = 0).ravel(), columns = [rivname])
#         max_pttb2 = pd.DataFrame(np.nanmax(pttb2, axis = 0).ravel(), columns = [rivname])
        
#         batch2_max_ptt = pd.concat((batch2_max_ptt, max_pttb2), axis = 1, ignore_index = False, names = [rivname])
#         batch2_nswitch = pd.concat((batch2_nswitch, nswitchb2), axis = 1, ignore_index = False, names = [rivname])
    

        
#%%

# remove zeros from dataframe as they take up space

# batch1_max_ptt_nz = copy.deepcopy(batch1_max_ptt) ##nz = no zeros
# batch1_nswitch_nz = copy.deepcopy(batch1_nswitch)

# batch2_max_ptt_nz = copy.deepcopy(batch2_max_ptt)
# batch2_nswitch_nz = copy.deepcopy(batch2_nswitch)

# for col in batch2_nswitch.columns:
#     # Filter out the rows where value is not 0
#     batch2_max_ptt_nz[col] = batch2_max_ptt[col][batch2_max_ptt[col] != 0]
#     batch2_nswitch_nz[col] = batch2_nswitch[col][batch2_nswitch[col] != 0]
    
# for col in batch1_nswitch.columns:
#     batch1_max_ptt_nz[col] = batch1_max_ptt[col][batch1_max_ptt[col] != 0]
#     batch1_nswitch_nz[col] = batch1_nswitch[col][batch1_nswitch[col] != 0]
        
# new_b1_columns = ['indus-r2_b1', 'irrawaddy_b1', 'kasai_b1', 'rakaia_b1', 'southsask_b1', 'tanana_b1', 'yukon_b1']
# new_b2_columns = ['colville_b2', 'congo_lukolela_bolobo_b2', 'indus-r2_b2', 'irrawaddy_b2', 'kasai_b2', 'rakaia_b2', 'southsask_b2', 'tanana_b2', 'yukon_b2']


# # oldb1 = ['indus-r2', 'irrawaddy', 'kasai', 'rakaia', 'southsask', 'tanana',
# #        'yukon']
# # oldb2 = ['colville', 'congo_lukolela_bolobo', 'indus-r2', 'irrawaddy', 'kasai',
# #        'rakaia', 'southsask', 'tanana', 'yukon']
# # batch2_nswitch.columns = oldb2
# # batch1_nswitch.columns = oldb1

# ## rename columns of max_ptt and nswitch dfs
# batch1_max_ptt_nz.columns = new_b1_columns
# batch2_max_ptt_nz.columns = new_b2_columns

# batch1_nswitch_nz.columns = new_b1_columns
# batch2_nswitch_nz.columns = new_b2_columns

# max_ptt_merge = batch1_max_ptt_nz.merge(batch2_max_ptt_nz, left_index=True, right_index=True)
# nswitch_merge = batch1_nswitch_nz.merge(batch2_nswitch_nz, left_index=True, right_index=True)

# ## drop nans where it exists in all rows

# max_ptt_merge_dn = max_ptt_merge.dropna(axis = 0, how = 'all')
# nswitch_merge_dn = nswitch_merge.dropna(axis = 0, how = 'all')        

## save these dataframes as csv files

# max_ptt_merge_dn.to_csv(os.path.join(root, dataframe_data, 'max_ptt_b1b2_nonans_nozero.csv'))       
# nswitch_merge_dn.to_csv(os.path.join(root, dataframe_data, 'nswitch_b1b2_nonans_nozero.csv'))               

# ## if pulling just the csv files 
max_ptt_merge_dn = pd.read_csv(os.path.join(root, dataframe_data, 'max_ptt_b1b2_nonans_nozero.csv'))
nswitch_merge_dn = pd.read_csv(os.path.join(root, dataframe_data, 'nswitch_b1b2_nonans_nozero.csv'))       

max_ptt_merge_dn = max_ptt_merge_dn.drop(['Unnamed: 0'], axis = 1) 
nswitch_merge_dn = nswitch_merge_dn.drop(['Unnamed: 0'], axis = 1)
#%% 

# compare_batches_names = ['indus-r2_b1', 'irrawaddy_b1', 'kasai_b1', 'rakaia_b1',
#                 'southsask_b1', 'tanana_b1', 'yukon_b1', 'colville_b2',
#                 'congo_lukolela_bolobo_b2', 'indus-r2_b2', 'irrawaddy_b2',
#                 'kasai_b2', 'rakaia_b2', 'southsask_b2', 'tanana_b2', 'yukon_b2']

# violin_maxptt = max_ptt_merge_dn.drop(['Unnamed: 0', 'colville_b2', 'congo_lukolela_bolobo_b2'], axis = 1)

# in_both = ['indus-r2', 'irrawaddy', 'kasai', 'rakaia', 'southsask', 'tanana', 'yukon']

# violin_maxptt['River'] = [name.split('_')[0] for name in violin_maxptt.columns]
# violin_maxptt['Type'] = [name.split('_')[1] for name in violin_maxptt.columns]

# melted_violinptt = violin_maxptt.melt(id_vars=['River', 'Type'], var_name='Column', value_name='Value')

meanprops = dict(marker = 'o', markerfacecolor = 'blue', ms = 5, mec = 'k', mew = 0)
flierprops = dict(marker='o', markerfacecolor='xkcd:gray', markersize=10,  markeredgecolor='xkcd:gray')


fig = plt.figure('maxptt', figsize = (10, 5), tight_layout = True, dpi = 400)
ax = plt.gca()
sns.boxplot(max_ptt_merge_dn, width = .5, whis = (10, 90), notch = False, showfliers = False, showmeans = True, meanprops = meanprops, flierprops = flierprops)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60);
ax.set_ylabel('Lenght of longest turnover per pixel')
plt.savefig('/Users/safiya/Desktop/colloquium_local/maxptt_box.svg')
        
fig = plt.figure('nswitches', figsize = (10, 5), tight_layout = True, dpi = 400)
ax = plt.gca()
sns.boxplot(nswitch_merge_dn, width = .5, whis = (10, 90), notch = False, showfliers = False, showmeans = True, meanprops = meanprops, flierprops = flierprops)
ax.set_ylabel('Number of turnovers per pixel')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60);
plt.savefig('/Users/safiya/Desktop/colloquium_local/nswitches_box.svg') 

# fig = plt.figure('maxptt', figsize = (10, 5), tight_layout = True, dpi = 400)
# ax = plt.gca()
# # sns.stripplot(max_ptt_merge_dn, width = .5, whis = (5, 95), notch = False, showfliers = False, showmeans = True, meanprops = meanprops, flierprops = flierprops)
# sns.violinplot(max_ptt_merge_dn, inner = 'box', width = 0.75, linewidth = 1, bw_method = 'silverman', bw_adjust = 1.5)
# # sns.boxplot(max_ptt_merge_dn, fill = False, showmeans = True, meanprops = meanprops, showfliers = False, width = 0.25, whis = (5, 95), linecolor = 'r', linewdth = 1)

# ax.set_xticklabels(ax.get_xticklabels(), rotation=60);
# plt.savefig('/Users/safiya/Desktop/colloquium_local/maxptt_vio.svg')
        
# fig = plt.figure('nswitches', figsize = (10, 5), tight_layout = True, dpi = 400)
# ax = plt.gca()
# sns.boxplot(nswitch_merge_dn, width = .5, whis = (5, 95), notch = False, showfliers = False, showmeans = True, meanprops = meanprops, flierprops = flierprops)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=60);
# plt.savefig('/Users/safiya/Desktop/colloquium_local/nswitches_vio.svg')        
        
#%% 

qs_bqart_hydrosheds = [4.003670996,
58.60795027,
378.5154589,
941.6220194,
232.2508518,
42.13947861,
0.681072279,
82.44863155,
42.03729241] ## kg/yr, values i calculated using hydrosheds and bqart, in order of riv names
        
med_max_ptt = max_ptt_merge_dn.median(axis = 0)
mean_max_ptt = max_ptt_merge_dn.mean(axis = 0)

median_nswitch = nswitch_merge_dn.median(axis = 0)
mean_nswitch = nswitch_merge_dn.mean(axis = 0)
        
        
                         

