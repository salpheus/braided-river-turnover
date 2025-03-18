#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:34:31 2024

@author: safiya
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)

# %% get ebi descriptive statistics
main_fol = '/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated'
rivlist = ['amudaryadown','amudaryanew','betsiboka','bhareli_wide', 'brahmaputra_pandu_allyr',
           'colville', 'congo_lukolela_bolobo','indus_r2','irrawaddy','irrawaddy_up',
           'kasai','lena', 'mangoky','ob_down', 'ob_up', 'rakaia', 'southsask',
           'tanana', 'yukon', 'yukon_eagle']

for river in rivlist:
    
    csvs = glob.glob(os.path.join(main_fol, river, 'form-master*.csv'))
    dataframes = [pd.read_csv(f, header = 0) for f in csvs]
    
    ebi_master = pd.DataFrame(columns = ['FID', 'ebi', 'wetted_width', 'thread_count'])
    
    thread_width_master = pd.DataFrame(columns = ['FID', 'min', 'max', 'mean', 'sd', '50'])
    
    for yr, df in enumerate(dataframes):
        year = int(csvs[yr].split('/')[-1].split('.csv')[0].split('_')[-1])
        df['year'] = np.ones_like(len(df))*year
        
        ebi_master = pd.concat((ebi_master, df.loc[:, ['FID', 'ebi', 'wetted_width', 'thread_count']]), axis = 0).dropna(axis = 0, how = 'any')
        thread_width_master = pd.concat((thread_width_master, df.loc[:, ['FID', 'min', 'max', 'mean', 'sd', '50']]), axis = 0).dropna(axis = 0, how = 'any')
    
    ebi_master.to_csv(os.path.join(main_fol, '00_form_masters', f'{river}_master_ebi-bi.csv'))
    thread_width_master.to_csv(os.path.join(main_fol, '00_form_masters', f'{river}_master_width.csv'))
    
#%%
var = 'ebi'
ebi_descrip = pd.DataFrame(columns = rivlist)
ebicsvs = glob.glob(os.path.join(main_fol, '0_ebi_masters', '*.csv'))

masters = [pd.read_csv(f, header = 0) for f in ebicsvs]

fig, ax = plt.subplots(3, 7, figsize = (20, 12), dpi = 300, sharex = True)
ax = ax.ravel()

for i, df in enumerate(masters):
    df[var][df[var]<=0] = np.nan
    stats = df[var].describe()
    
    ebi_descrip[rivlist[i]] = stats
    ax[i].hist(df[var], bins = np.arange(0, 10, .5), ec = 'k', fc = 'xkcd:sage green')
    ax[i].set_title(rivlist[i])
    ax[i].set_xlabel(var) 
epi_descrip = ebi_descrip.T





