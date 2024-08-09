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

#%% get ebi descriptive statistics
main_fol = '/Volumes/SAF_Data/remote-data/rivgraph_transects_curated'
rivlist = os.listdir(main_fol)
rivlist.remove('.DS_Store')
rivlist.remove('0_ebi_masters')

# for river in rivlist[1:]:
#     csvs = glob.glob(os.path.join(main_fol, river, '*.csv'))
#     dataframes = [pd.read_csv(f, header = 0) for f in csvs]
    
#     master = pd.DataFrame(columns = ['FID', 'ebi', 'wetted_width', 'year'])
    
#     for yr, df in enumerate(dataframes):
#         year = int(csvs[yr].split('/')[-1].split('.csv')[0].split('_')[-1])
#         df['year'] = np.ones_like(len(df))*year
        
#         master = pd.concat((master, df), axis = 0)
#     master.to_csv(os.path.join(main_fol, river, f'{river}_master_ebi.csv'))
    
#%%

ebi_descrip = pd.DataFrame(columns = rivlist)
ebicsvs = glob.glob(os.path.join(main_fol, '0_ebi_masters', '*.csv'))

masters = [pd.read_csv(f, header = 0) for f in ebicsvs]

fig, ax = plt.subplots(3, 7, figsize = (20, 12), dpi = 300, sharex = True)
ax = ax.ravel()

for i, df in enumerate(masters):
    df['ebi'][df['ebi']==0] = np.nan
    stats = df['ebi'].describe()
    
    ebi_descrip[rivlist[i]] = stats
    ax[i].hist(df['ebi'], bins = np.arange(1, 10, 0.5), ec = 'k', fc = 'xkcd:sage green')
    ax[i].set_title(rivlist[i])
    ax[i].set_xlabel('ebi') 
epi_descrip = ebi_descrip.T





