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
import matplotlib.colors as mcol

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
plt.rc('legend',fontsize=8, title_fontsize=8)


#%% load megamerge

megamerge = pd.read_csv('/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/megadf_pca/clipped_ndvi/megamerge_pca.csv')
megamerge = megamerge.set_index('river')
megamerge = megamerge.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis = 1)
print(megamerge.columns)

inventory = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', sheet_name='inventory_py')
inventory.drop([0], axis = 0)
inventory = inventory.set_index('river')
#%% test for normalite with a kstest

# pca_params = ['river', 'ndvi', 'meantt', 'nturns', 
#        'mean_annu_qw_sc', 
#        'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'part_size_mm', 'Tm_timescale', 'Tr_timescale',
#        'terrain_slope', 'mean_ebi', 'med_ebi',
#        'max_ebi', 
#        'mean_slope', 'mean_qw_annu_m3s', '2yr_ret_qwflood_m3s', '2yr/mean']
## updated params to remove obvious axes where hey show the most variance 
# pca_params = ['kg_clim', 
#        'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'Tm_timescale',
#        'Tr_timescale', 'efficiency', 'terrain_slope', 'sand_frac',
#        'clay_frac', 'med_ebi','r_len_shape',
#        'r_ebi_shape', 'r_nturn_shape', 
#        'mean_slope', 
#        'mean_arid_idx_catch', '2yr/mean', 
#        'qw_norm_area'] ##IT1

# ## Iteration 2: removed the shape parameters for the distributions
# pca_params = ['kg_clim', 
#        'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'Tm_timescale',
#        'Tr_timescale', 'efficiency', 'terrain_slope', 'sand_frac',
#        'clay_frac', 'med_ebi', 
#        'mean_slope', 
#        'mean_arid_idx_catch', '2yr/mean', 
#        'qw_norm_area'] ##IT2

## Iteration 3: removing sand/clay fractions because this data is from 0-5cm layer of soul and thats really small
## and Tr and Tm because they are captured in 'efficiency'
# pca_params = ['kg_clim', 
#         'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'efficiency', 'terrain_slope', 'med_ebi', 
#         'mean_slope', 
#         'mean_arid_idx_catch', '2yr/mean', 
#         'qw_norm_area']

## iteration 4: with insights form inventory pca
# pca_params = ['kg_clim', 
#         'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'efficiency',# 'terrain_slope',
#         'med_ebi','r_len_shape',
#         'r_ebi_shape',
#         'mean_slope', 
#         'mean_arid_idx_catch', '2yr/mean', 
#         'qw_norm_area'] ##IT


# iteration5-sunday 18 aug without turnover information
pca_params = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', #'num_turns_mean', 
                 'mean_slope', 'dvia_med',  #'max_tt_mean',
                'efficiency', 'med_ebi']
# [ ### list fof inventory parameters to pair plot by or colour by
#         'bed_prop_of_total', 'bed-ssc_qw_m3yr', 'efficiency', 'med_ebi',
#         'r_ebi_shape', 
#         'mean_slope', 
#         'mean_arid_idx_catch', 'mean_DVIa', 
#         'mean_qw_norm']#, 'mean_tt_length', 'max_tt_mean', 'num_turns_mean']
# inv_pca =  inventory.set_index('river').loc[:, feature_list]
inventory_pca = inventory.loc[:, pca_params]

pvals_newdata =  []#['meantt', 'nturns']

megamerge = pd.merge(megamerge, inventory_pca, on='river', how='left')

#%% take logs we know we have to take logs og

# to_log = ['bed-ssc_qw_m3yr', 'bed_prop_of_total', 'mean_slope',
#           'Tm_timescale', 'Tr_timescale'] ## this is all visual and manually done for anything in the previous plot that looks like an exponential
## itertion sunday 18 aug
to_log = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', 'part_size_mm', 
                  'mean_slope']
# ['mean_qw_norm', 'bed-ssc_qw_m3yr', 'bed_prop_of_total', 'mean_slope', 'ndvi_med', 'mean_arid_idx_catch', 'med_ebi']#,

for val in to_log:
    if val in megamerge.columns:
        print(val)
        megamerge[val] = np.log(megamerge[val])
    
# megamerge = megamerge.drop('qw_norm_area_x', axis = 1)
# megamerge.rename(columns={'qw_norm_area_x': 'qw_norm_area'})
# for var in pvals_newdata:

#     anderson = stats.anderson(megamerge[var], 'norm')
#     print(f"Anderson-Darling Test: Statistic={anderson.statistic}")
#     print("Critical values:", anderson.critical_values)
#     print("Significance levels:", anderson.significance_level)

## make histos of big distributions

# #% make climate data numeric
# basic_climate = ['B', 'B', 'A', 'C', 'C', 'D', 'A', 'B', 'B', 'A', 'A', 'D', 'B', 'D', 'D', 'C', 'B', 'D', 'D', 'D']#%% take log of params you know arent notmally dis

#%% standardize the df
 
to_standardize = megamerge.copy().dropna(axis = 0, how = 'any') 
to_standardize = to_standardize.drop('medtt', axis = 1)
unstandardized_data = to_standardize.copy()
# to_standardize = to_standardize.drop('meantt', axis = 1)
#%% dont run this if leaving turnover behavious out of the pca
for i in pvals_newdata:
    if i in to_standardize.columns:
        
        to_standardize[i] = np.log(to_standardize[i])
    
#%%

means = to_standardize.mean(numeric_only = True)
stds = to_standardize.std(numeric_only = True)

for val in to_standardize.columns:
    if val in ['kg_clim', 'meantt', 'nturns']:
        continue
    print(val)
    to_standardize[val] = (to_standardize[val]-means[val])/stds[val]

print('means: ', to_standardize.mean(numeric_only = True))
print('stdev: ', to_standardize.std(numeric_only = True))

# fig, ax = plt.subplots(2, 3, figsize = (12, 8), dpi = 300)
# ax[0, 0].hist(megamerge['ndvi'], bins = np.arange(-1, 1, 0.05))
# ax[0, 1].hist(megamerge['meantt'], bins = np.arange(1, 25, 1))
# ax[0, 2].hist(megamerge['nturns'], bins = np.arange(1, 25, 1))

# ax[0, 0].set_title('ndvi')
# ax[0, 1].set_title('mean tt')
# ax[0, 2].set_title('numturns')

# ax[1, 0].hist(to_standardize['ndvi']);
# ax[1, 1].hist(to_standardize['meantt'], 20)
# ax[1, 2].hist(to_standardize['nturns'], 20);

#%% do PCA
# 1. perform dimensionality reduction

pca = PCA(n_components=8, svd_solver = 'auto')

## drop nan rows (ndvi outliers), strings (kgclim) for final pca dataset
# pca_data = to_standardize.drop('kg_clim', axis = 1).values  ## separate out the values into as high-dimensional array (should be shape [:, 19])
pca_data = to_standardize.drop(columns = ['meantt', 'nturns']).values  ## separate out the values into as high-dimensional array (should be shape [:, 19])

principalComponents = pca.fit_transform(pca_data)   ## perform the transform (calc cov matrix transforming the data to 2x2)     
        
variance_ratio = (pca.explained_variance_ratio_) #array([0.87920411, 0.0684717 ])
cutoff = np.where(np.cumsum(variance_ratio) > .75)[0][0] ## cutoff PCs whwen cumulative sum of variance exceeds 95%

#%% make pca plots 

plt.figure(figsize = (5,5), dpi = 150)
plt.plot(np.cumsum(variance_ratio))
plt.axvline(cutoff)
plt.xlabel('principal components')
plt.ylabel('Cumulative expected variance')

# get component scores-- principal axes in feature space representing the direction of max variance in the data

xaxlabels = to_standardize.drop(columns = ['meantt', 'nturns']).columns
plt.figure(figsize = (10, 4), dpi = 300, tight_layout = True)
for comp in range (0, cutoff+1):
    plt.plot(pca.components_[comp], label = comp)
# plt.xlim(0, len(to_standardize.columns))
plt.legend()
ax = plt.gca()
ax.set_xticks(range(len(xaxlabels))) ## alays set tick range before prescribing labels
ax.set_xticklabels(xaxlabels);
ax.xaxis.set_tick_params(rotation=70)

ax.set_xlabel('pc variance in each variable')
ax.set_ylabel('variance')


#%%
pcs_cutoff = principalComponents[:, :cutoff+1] ### array of position in PC spaace for each pixel
pcs_cutoff_df = pd.DataFrame(pcs_cutoff, index = to_standardize.index.values)
pcs_cutoff_df = pd.concat((pcs_cutoff_df, unstandardized_data), axis = 1) ## concatenate w real data so you can colour by value

# Assuming `data_x` and `data_y` are your 4M data points
sample_rows = np.random.choice(len(unstandardized_data), size=100000, replace=False)

sampledf = pcs_cutoff_df.iloc[sample_rows, :].reset_index()
sampledf = sampledf.rename(columns={'index':'river'})
sampledf['meantt'] = np.round(sampledf['meantt'], 1)
# plt.figure(figsize = (10, 10), dpi = 300)
# sns.scatterplot(data = sampledf, 
#                 x = 0, y = 1, hue = 'river', marker = '.', s = 5, edgecolors = None)


# plt.figure(figsize = (12, 12), dpi = 300)
# sns.pairplot(data = sampledf, 
#              vars = [0, 1, 2, 3, 4], 
#               aspect = 1, hue = 'nturns', diag_kind = 'kde', height = 5, palette = 'PuRd',
#              plot_kws=dict(marker=".", s=100, edgecolors = None))
# sns.pairplot(data = sampledf, 
#              vars = [0, 1, 2, 3, 4], 
#               aspect = 1, hue = 'meantt', diag_kind = 'kde', height = 5, palette = 'spring',
#              plot_kws=dict(marker=".", s=100, edgecolors = None))



# prin = sns.pairplot(data = pcs_cutoff_df,
#              vars = [0, 1, 2, 3, 4], height = 1.5, aspect = 1, dropna = True, hue = 'efficiency',
#              palette = sns.color_palette("seismic", 20),
#              diag_kind = 'hist', size = 10, plot_kws={'s':500})

#%% Make a biplot with the first two principal components

colby = 'nturns'
nt_norm = mcol.Normalize(vmin = np.floor(sampledf[colby].min()), 
                         vmax = np.ceil(sampledf[colby].max()))
scaler = 3

plt.figure(figsize = (5, 5,), dpi = 200)
ax = plt.gca()

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis='both', which='minor', labelcolor = 'k')
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))
ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))

plt.grid(True, 'major')

biplot = plt.scatter(sampledf[1], sampledf[2], c = sampledf[colby], ec = None, s = 20,
              cmap = 'inferno', norm = nt_norm)
plt.xlabel('PC1')
plt.ylabel('PC2')

## loadiings for arrows on biplot
loadings_pca1 = scaler*pca.components_[1]
loadings_pca2 = scaler*pca.components_[2]

names = ['NDVI', '$Q_{bfrac}$', '$uQ_{s}$', 'u$Q_{w}$', 'eff.', 'eBI$_{med}$', 'slope', '$DVIa_{med}$']

for n, name in enumerate(names):
    plt.arrow(0, 0, loadings_pca1[n], loadings_pca2[n], length_includes_head = False,
              head_width = .1, head_length = .2, fc = 'k')
    plt.annotate(name, (loadings_pca1[n], loadings_pca2[n]), color = 'g')
    
# for i in range(len(abr_rivnames)):
#     plt.annotate(abr_rivnames[i], (pca_results['cPC1'][i]+0.25, pca_results['cPC2'][i]+0.1), fontsize = 6, color = 'm')
plt.colorbar(biplot, label = 'mean number of turns', shrink = 0.75)




    