#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 23:49:09 2024
plot histograms of form things sfor supplement
@author: safiya
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob

from matplotlib.ticker import MultipleLocator

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)
#%% load dfs and plot things
thread_widths_dfs = glob.glob('/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated/000_threadwidths/*.txt')
tws = [pd.read_csv(f) for f in thread_widths_dfs]

rivlist = ['ADD', 'ADU', 'BET', 'BHA', 'BRA',
           'COL', 'CON', 'IND', 'IRA', 'IRU',
           'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK',
           'TAN', 'YUK', 'YUE']

#%% plot histograms

fig, ax = plt.subplots(4, 5, figsize = (15, 15), dpi = 300, tight_layout = True, sharex=True)
ax = ax.ravel()
for a, tw in enumerate(tws):
    ax[a].hist(tw, bins = np.arange(30, 4000, 150), fc = 'xkcd:light grey', ec = 'k', lw = 0.5)
    ax[a].set_title(f'{rivlist[a]}: Avg width = {np.round(np.mean(tw), 1)} m')
    ax[a].set_xlabel('Thread width, m')