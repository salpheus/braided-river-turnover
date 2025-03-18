#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:41:45 2024
make thread count_mabbab violinplot for chapter 4

@author: safiya
"""
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

ancient = pd.read_excel('/Users/safiya/Desktop/dissertation_wtf/BAR-INVENTORY_local.xlsx', sheet_name = 'MAB-BAB_plot')
modern = [pd.read_csv(f) for f in glob.glob('/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated/00_form_masters/*ebi-bi.csv')]
model = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated/model_ebi_bi_plot.xlsx', sheet_name = 'remapped-22oct')
abr_rivnames = ['MOD', 'ADD', 'ADU', 'BET', 'BHA', 'BRA', 'COL', 'CON', 'IND', 'IRA', 'IRU', 'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK', 'TAN', 'YUE', 'YUK']
allts = pd.read_excel('/Volumes/SAF_Data/SAF_Data/bar-manuscript_sum22/data-interp/BI-EBI.xlsx', sheet_name = 'bi-edit', skiprows = 15, index_col = 0).to_numpy()

#%%

flierprops = dict(marker='o', markerfacecolor='xkcd:gray', markersize=2,  markeredgecolor='xkcd:gray')
meanprops = dict(marker = 'o', markerfacecolor = 'blue', ms = 0, mec = 'k', mew = 0, linestyle = '--', linewidth = 1.5, color = 'k')
meanlineprops = dict(linestyle = '-', lc = 'k', lw = 2)
boxprops = dict(color = 'k', linewidth = 1.5)
capprops = dict(color = 'k', linewidth = 1.5)
whiskerprops = dict(color = 'k', linecolor = 'k')
boxwidth = 0.7
linewidth = 1.5


plt.figure(figsize = (13, 3), dpi = 300, tight_layout = True)

sns.stripplot(ancient, size = 8)
sns.stripplot(model['MCB:BAB'], size = 8,native_scale = True)

# modelvio = plt.violinplot(model['MCB:BAB'], positions = [6], 
#                 showmedians = False, showextrema = True, widths=boxwidth, bw_method = 0.3)

modelbi = plt.violinplot(allts.ravel(), positions = [7], 
                showmedians = False, showextrema = True, widths=boxwidth, bw_method = 0.3)

for i, dist in enumerate(modern):
    if i == 6 or i == 4:
        violins = plt.violinplot(dist['thread_count'][dist['thread_count']<np.quantile(dist['thread_count'], .99)], positions = [i+8], 
                        showmedians = False, showextrema = True, widths=boxwidth, bw_method = 0.3)
    else:
        violins = plt.violinplot(dist['thread_count'][dist['thread_count']>=1], positions = [i+8], ## have to clean up, there is no possible way for tc to be less than 0
                        showmedians = False, showextrema = True, widths=boxwidth, bw_method = 0.3)
    
    for pc in violins['bodies']:
        pc.set_facecolor(None)
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
    
    emin, q1, med, q3, emax = np.nanquantile(dist['thread_count'][dist['thread_count']>=1], [.05, .25, .5, .75, .95])
    
    # plt.vlines(idx+1, ymin = emin, ymax = emax, ec = 'k', zorder = 100)
    plt.vlines(i+8, ymin = q1, ymax = q3, ec = 'k', lw  = 5, zorder = 101)
    # plt.scatter(i+8, med, c = 'w', marker = 'o', s = 5, zorder = 102)
    # plt.scatter(i+8, dist['thread_count'].mean(), c = 'w', marker = 'o', s = 5, zorder = 102)
for vio in [modelvio, modelbi]:    
    for pc in vio['bodies']:
        pc.set_facecolor(None)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        pc.set_linewidth(.5)
    # Change the color of the extrema lines
    # for line in :  # Min line
        vio['cmins'].set_color('k')
        vio['cmins'].set_linewidth(1)
        vio['cmaxes'].set_color('k')
        vio['cmaxes'].set_linewidth(1)
        vio['cbars'].set_color('k')
        vio['cbars'].set_linewidth(1)

vmin, q1, med, q3, emax = np.nanquantile(model['MCB:BAB'], [.05, .25, .5, .75, .95])
plt.vlines(6, ymin = q1, ymax = q3, ec = 'k', lw  = 5, zorder = 101)
plt.scatter(6, med, c = 'w', marker = 'o', s = 5, zorder = 102)

vmin, q1, med, q3, emax = np.nanquantile(allts, [.05, .25, .5, .75, .95])
plt.vlines(7, ymin = q1, ymax = q3, ec = 'k', lw  = 5, zorder = 101)
plt.scatter(7, med, c = 'w', marker = 'o', s = 5, zorder = 102)
# plt.yscale('log')
plt.ylabel('MCB:BAB')
plt.xticks(rotation = 15);
ax = plt.gca()
ax.set_xticks(np.arange(28)) ## alays set tick range before prescribing labels
ax.set_xticklabels(['HF', 'MO', 'CV', 'BF', 'JV', 'PC', 'MOD'] + abr_rivnames);

plt.savefig('/Volumes/SAF_Data/SAF_Data/CHAPTER3/manu_figs/mab-bab_mean.svg')
