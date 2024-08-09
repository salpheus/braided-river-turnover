#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:07:47 2024

@author: safiya
"""
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import seaborn as sns
from sklearn.decomposition import PCA
import scipy.stats as stats
from matplotlib.ticker import MultipleLocator

# from functions import *
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 6}

mpl.rc('font', **font)

#%% load megamerge

megamerge = pd.read_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/megadf_pca/megamerge_pca.csv')
megamerge = megamerge.drop(columns = ['Unnamed: 0.1', 'Unnamed: 0'], axis = 1)
megamerge.set_index('river')

print(megamerge.columns)
to_log = ['mean_annu_qw_sc', 'bed-ssc_qw_m3yr', 'part_size_mm', 'bed_prop_of_total', 'mean_slope', '2yr/mean',
          'Tm_timescale', 'Tr_timescale'] ## this is all visual and manually done for anything in the previous plot that looks like an exponential

inventory = pd.read_excel('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', sheet_name='inventory_pca')

#%% take log of params you know arent notmally dis
for val in to_log:
    megamerge[val] = np.log(megamerge[val])
    
#%% test for normalite with a kstest

# pca_params = ['river', 'ndvi', 'meantt', 'nturns', 
#        'mean_annu_qw_sc', 
#        'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'part_size_mm', 'Tm_timescale', 'Tr_timescale',
#        'terrain_slope', 'mean_ebi', 'med_ebi',
#        'max_ebi', 
#        'mean_slope', 'mean_qw_annu_m3s', '2yr_ret_qwflood_m3s', '2yr/mean']
## updated params to remove obvious axes where hey show the most variance 
pca_params = ['river', 'ndvi', 'meantt', 'nturns', 
       'mean_annu_qw_sc', 
       'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'part_size_mm', 'Tm_timescale', 'Tr_timescale',
       'terrain_slope', 'mean_ebi', 'med_ebi',
       'max_ebi', 
       'mean_slope', 'mean_qw_annu_m3s', '2yr_ret_qwflood_m3s', '2yr/mean']

pvals_newdata = ['ndvi', 'meantt', 'nturns']

# for var in pvals_newdata:

#     anderson = stats.anderson(megamerge[var], 'norm')
#     print(f"Anderson-Darling Test: Statistic={anderson.statistic}")
#     print("Critical values:", anderson.critical_values)
#     print("Significance levels:", anderson.significance_level)

## make histos of big distributions

# #% make climate data numeric
# basic_climate = ['B', 'B', 'A', 'C', 'C', 'D', 'A', 'B', 'B', 'A', 'A', 'D', 'B', 'D', 'D', 'C', 'B', 'D', 'D', 'D']

to_standardize = megamerge[pca_params].copy()
## use MAT and aridity idx instead
rivs, counts = np.unique(megamerge['river'], return_counts = True)

to_standardize['MAT'] = np.repeat(inventory.loc[1:, 'MAT'], counts).reset_index(drop=True)
to_standardize['aridity_idx'] = np.repeat(inventory.loc[1:, 'aridity_idx'], counts).reset_index(drop=True)
# ## cant do needs to be binary

to_standardize.set_index('river')

for i in pvals_newdata:
    to_standardize[i] = np.log(to_standardize[i])

means = to_standardize.mean(numeric_only = True)
stds = to_standardize.std(numeric_only = True)

for val in to_standardize.columns:
    to_standardize[val] = (to_standardize[val]-means[val])/stds[val]

print('means: ', to_standardize.mean(numeric_only = True))
print('stdev: ', to_standardize.std(numeric_only = True))

fig, ax = plt.subplots(2, 3, figsize = (12, 8), dpi = 300)
ax[0, 0].hist(megamerge['ndvi'], bins = np.arange(-1, 1, 0.05))
ax[0, 1].hist(megamerge['meantt'], bins = np.arange(1, 25, 1))
ax[0, 2].hist(megamerge['nturns'], bins = np.arange(1, 25, 1))

ax[0, 0].set_title('ndvi')
ax[0, 1].set_title('mean tt')
ax[0, 2].set_title('numturns')

ax[1, 0].hist(to_standardize['ndvi']);
ax[1, 1].hist(to_standardize['meantt'], 20)
ax[1, 2].hist(to_standardize['nturns'], 20);

#%% do PCA
# 1. perform dimensionality reduction

pca = PCA(n_components=19, svd_solver = 'auto')

pca_data = to_standardize.values[:, 1:]  ## separate out the values into as high-dimensional array (should be shape [21, 32])
principalComponents = pca.fit_transform(pca_data)   ## perform the transform (calc cov matrix transforming the data to 2x2)     
        
variance_ratio = (pca.explained_variance_ratio_) #array([0.87920411, 0.0684717 ])
cutoff = np.where(np.cumsum(variance_ratio) > .95)[0][0] ## cutoff PCs whwen cumulative sum of variance exceeds 95%

#%% make pca plots 

plt.figure(figsize = (5,5), dpi = 150)
plt.plot(np.cumsum(variance_ratio))
plt.axvline(cutoff)
plt.xlabel('principal components')
plt.ylabel('Cumulative expected variance')

# get component scores-- principal axes in feature space representing the direction of max variance in the data

plt.figure(figsize = (10, 4), dpi = 300, tight_layout = True)
for comp in range (0, cutoff+1):
    plt.plot(pca.components_[comp], label = comp)
# plt.xlim(0, len(to_standardize.columns))
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xticklabels(to_standardize.columns);
ax.xaxis.set_tick_params(rotation=70)

ax.set_xlabel('pc variance in each variable')
ax.set_ylabel('variance')

#%%
pcs_cutoff = principalComponents[:, :cutoff+1]
pcs_cutoff_df = pd.DataFrame(pcs_cutoff, index = to_standardize.index.values)
pcs_cutoff_df = pd.concat((pcs_cutoff_df, to_standardize), axis = 1) ## concatenate w real data so you can colour by value

# Assuming `data_x` and `data_y` are your 4M data points
sample_rows = np.random.choice(len(to_standardize), size=100000, replace=False)

sampledf = pcs_cutoff_df.loc[sample_rows, :]

plt.figure(figsize = (10, 10), dpi = 300)
sns.scatterplot(data = pcs_cutoff_df.loc[sample_rows, :], 
                x = 0, y = 1, hue = 'river', marker = '.', s = 5, edgecolors = None)


plt.figure(figsize = (18, 18), dpi = 300)
sns.pairplot(data = sampledf, 
             vars = [0, 1, 2, 3, 4, 5, 6, 7, 8], 
             height = 1.5, aspect = 1, hue = 'river', diag_kind = 'kde', corner = True,
             plot_kws=dict(marker=".", s=5, edgecolors = None))






    