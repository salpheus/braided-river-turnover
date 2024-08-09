#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:10:33 2024

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
# from functions import *
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 6}

mpl.rc('font', **font)

#%% load files
inventory = pd.read_excel('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx')
inventory=inventory.drop([0], axis = 0)
## climate zones starting from amudarya down!
basic_climate = ['B', 'B', 'A', 'C', 'C', 'D', 'A', 'B', 'B', 'A', 'A', 'D', 'B', 'D', 'D', 'C', 'B', 'D', 'D', 'D']
# define parameters for pca

 ## redoing pca with ndvi and less variables [8 AUG]
pca_params = ['mean_annu_qw_sc','bed-ssc_qw_m3yr', 'part_size_mm', 'bed_prop_of_total',
              'Tm_timescale', 'Tr_timescale', 'terrain_slope',
              'med_ebi', 'r_len_shape', 'r_ebi_shape', 'r_nturn_shape', 'r_meantt_shape',
              'mean_tt_length', 'max_tt_mean', 'num_turns_mean',
              'mean_slope', 'mean_arid_idx_catch',
              '2yr/mean', 'ndvi_med']
#pca_params [7 AUG!]
# pca_params = ['mean_annu_qw_sc', 'bed-ssc_qw_m3yr',
#        'part_size_mm', '3e_overlap', '3e_reworking', 
#        'Tm_timescale', 'Tr_timescale', 'efficiency', 'terrain_slope',
#        'gradient', 'MAT', 'aridity_idx', 'sand_frac', 'clay_frac', 'mean_ebi',
#        'std_ebi', 'med_ebi', 'max_ebi', 'cv_ebi',
#        'r_len_shape', 'entropy_len',
#        'r_ebi_shape', 'entropy_ebi',
#        'r_nturn_shape','entropy_nturn',
#        'r_meantt_shape', 'entropy_meantt',
#        'r_meantt_dry_shape','entropy_meantt_dry',
#        'mean_tt_length', 'max_tt_mean', 'num_turns_mean']

#%% plot probplots! [DONT USE] plots to look at the unnormalised distributions

fig, ax = plt.subplots(5, 7, dpi = 300, tight_layout = True, figsize = (20, 15))
ax = ax.ravel()
for a, col in enumerate(pca_params):
    stats.probplot(inventory[col].to_numpy(), dist = 'norm', fit = True, rvalue = True, plot = ax[a])
    ax[a].set_title(col)
    
#%% plot probplots! [DONT USE]to look at the unnormalised, logged distributions

to_log = ['mean_annu_qw_sc', 'bed-ssc_qw_m3yr', 'part_size_mm', '3e_overlap', '3e_reworking', 
          'Tm_timescale', 'Tr_timescale'] ## this is all visual and manually done for anything in the previous plot that looks like an exponential
fig, ax = plt.subplots(5, 7, dpi = 300, tight_layout = True, figsize = (20, 15))
ax = ax.ravel()
for a, col in enumerate(pca_params):
    if col in to_log: 
        stats.probplot(np.log(inventory[col].to_numpy()), dist = 'norm', fit = True, rvalue = True, plot = ax[a])
        ax[a].set_title(f'log of {col}')
    else:
        stats.probplot(inventory[col].to_numpy(), dist = 'norm', fit = True, rvalue = True, plot = ax[a])
        ax[a].set_title(col)
    

#%% plot QQ plots to look at raw distributions

fig, ax = plt.subplots(4, 5, dpi = 300, tight_layout = True, figsize = (18, 15))
ax = ax.ravel()
for a, col in enumerate(pca_params):
    samples = pd.to_numeric(inventory[col]).dropna()
    sm.qqplot(samples, dist = stats.norm, fit = True, line = 's', ax = ax[a]) 
    ax[a].set_title(col)
#%% plot QQ plots to look at the unnormalised, logged distributions using statsmodels!

## for pca_params [8 AUG!]
to_log = ['mean_annu_qw_sc', 'bed-ssc_qw_m3yr', 'part_size_mm', 'bed_prop_of_total', 'mean_slope', '2yr/mean',
          'Tm_timescale', 'Tr_timescale'] ## this is all visual and manually done for anything in the previous plot that looks like an exponential


## using pca_params [7AUG]
# to_log = ['mean_annu_qw_sc', 'bed-ssc_qw_m3yr', 'part_size_mm', '3e_overlap', '3e_reworking', 
#           'Tm_timescale', 'Tr_timescale'] ## this is all visual and manually done for anything in the previous plot that looks like an exponential
# fig, ax = plt.subplots(5, 7, dpi = 300, tight_layout = True, figsize = (20, 15))
fig, ax = plt.subplots(4, 5, dpi = 300, tight_layout = True, figsize = (18, 15))
ax = ax.ravel()
for a, col in enumerate(pca_params):
    samples = pd.to_numeric(inventory[col]).dropna()
    if col in to_log: 
        sm.qqplot(np.log(samples), dist = stats.norm, fit = True, line = 's', ax = ax[a])
        ax[a].set_title(f'log of {col}')
    else:
        sm.qqplot(samples, dist = stats.norm, fit = True, line = 's', ax = ax[a]) 
        ax[a].set_title(col)
    
#%%% create dataframe of pca parameters and normalise
inventory_gauss = inventory[inventory.columns.intersection(pca_params)]
inventory_gauss = inventory_gauss.set_index(inventory['river'])

for col in to_log:
    inventory_gauss[col] = np.log(inventory_gauss[col])
        
inventory_std = inventory_gauss.copy(deep=True)
for col in inventory_std.columns:
    inventory_std[col] = (inventory_gauss[col]-np.mean(inventory_gauss[col]))/np.std(inventory_gauss[col])
        
print(inventory_std.mean(axis = 0))
print(inventory_std.std(axis = 0))        
        
#%% first testing using sklearn.decomposition
## from https://builtin.com/machine-learning/pca-in-python

# 1. perform dimensionality reduction

pca = PCA(n_components=15, svd_solver = 'auto')

inv_gauss_data = inventory_std.values  ## separate out the values into as high-dimensional array (should be shape [21, 32])
principalComponents = pca.fit_transform(inv_gauss_data)   ## perform the transform (calc cov matrix transforming the data to 2x2)     
        
## make a dataframe with the prinvipal components
# principalDf = pd.DataFrame(data = principalComponents, 
#                            columns = ['PC1' , 'PC2'], index = inventory_gauss.index.values)
# principalDf = pd.concat((principalDf, pd.Series(basic_climate, index = inventory['river'], name = 'basic_clim')), axis = 1)
## visualise the projections
# plt.figure(figsize = (6, 6), tight_layout = True, dpi = 300) 
# sns.scatterplot(data = principalDf, x = 'PC1', y = 'PC2', hue = 'basic_clim')
# plt.xlabel('PC1')
# plt.ylabel('PC2')

variance_ratio = (pca.explained_variance_ratio_) #array([0.87920411, 0.0684717 ])
cutoff = np.where(np.cumsum(variance_ratio) > .95)[0][0] ## cutoff PCs whwen cumulative sum of variance exceeds 95%

plt.figure(figsize = (5,5), dpi = 150)
plt.plot(np.cumsum(variance_ratio))
plt.axvline(cutoff)
plt.xlabel('principal components')
plt.ylabel('Cumulative expected variance')
# get component scores-- principal axes in feature space representing the direction of max variance in the data

#these each have one score for each variable in the dataset
pc1 = pca.components_[0]
pc2 = pca.components_[1] 

#%% display circles function from online https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/tree/master
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            plt.show(block=False)
#%% interpret the results

plt.figure(figsize = (10, 4), dpi = 300, tight_layout = True)
for comp in range (0, cutoff+1):
    plt.plot(pca.components_[comp], label = comp)
plt.xlim(0, len(inventory_std.columns))
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xticklabels(inventory_std.columns);
ax.xaxis.set_tick_params(rotation=70)

ax.set_xlabel('pc variance in each variable')
ax.set_ylabel('variance')


# Generate a correlation circle
# pcs = pca.components_ 
# display_circles(pcs, n_comp=2, pca, axis_ranks = [0,1], labels = inventory_std.columns, label_rotation = 70)

#%% 
pcs_cutoff = principalComponents[:, :cutoff+1]
pcs_cutoff_df = pd.DataFrame(pcs_cutoff, index = inventory_std.index.values)
pcs_cutoff_df = pd.concat((pcs_cutoff_df, pd.Series(basic_climate, index = inventory['river'], name = 'basic_clim')), axis = 1)
sns.pairplot(data = pcs_cutoff_df, height = 1.5, aspect = 1, dropna = True, hue = 'basic_clim', diag_kind = 'hist', corner = True)




















