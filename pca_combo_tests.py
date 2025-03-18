#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:35:14 2024
Final code to do PCA and export results
@author: safiya
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcol
from sklearn.decomposition import PCA
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans

# from functions import *
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('legend',fontsize=8, title_fontsize=8)
mpl.rc('font', **font)

#%% code for continuous cmaps
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcol.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


#%%

inventory = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/admin/inventory-offline_bhpnew.xlsx', 
                          sheet_name='inventory_py', index_col = 0)
column_names = inventory.columns.tolist()

simplify_with = ['qw_norm_catch_area', 'bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', 'part_size_mm',
                 'efficiency', 'MAT', 'med_ebi', 'r_len_shape', 'std_ebi', 'mean_tt_length',
                 'num_turns_mean', 'mean_slope', 'mean_arid_idx_catch', 'ndvi_mean', 'turn_freq',
                 'dvia_med', 'DVIc', 'ww_50_m', 'max_tt_mean', 'Tm_timescale', 'Tr_timescale', 'corridor_prop_turned', 'stream_pow', 'stat_mean_turn']

inv_simple = inventory[simplify_with].copy()
#%% define colormapas
colby = 'stat_mean_turn'
# colby = entropies_unlog
# colby = turnover_props.idxmax(axis = 0).to_numpy()
## purp to orange with pink
# turn_cols =['#5a34ef', '#8a4ec7', '#ab61ac', '#c87193', '#e4817c', '#fea950', '#fc994f', '#fa894d', '#f9784b', '#f76549'] 
## a new blue sun colours
# turn_cols = ['#625841', '#726147', '#826b4d', '#5d7ab0', '#6587ba', '#7294b9', '#8ba292', '#a2af70', '#b7bb50', '#cbc631']
# turn_cols = ['#372808', '#41736c', '#82bbc0', '#ffffff', '#ffd164', '#ff9e51', '#ff603b']#['#372808', '#3d5544', '#438280', '#69aeb4', '#b4d6d9', '#ffe476', '#ffc760', '#ffa955', '#ff8749', '#ff603b']
# turn_cols =['#0014a5', '#0c399b', '#145195', '#1c688f', '#237d89', '#2f9283', '#4fa67a', '#6fb871', '#deb54e', '#ffc353']
turn_cols =['#000079', '#3f137a', '#681f7b', '#8d2a7b', '#b2357c', '#cb4c84', '#e3638b', '#fb7b92', '#ffa43d', '#ffc635', '#ffe556']



# nt_norm = mcol.Normalize(vmin = np.round(inv_simple[colby].mean()-inv_simple[colby].std(), 1),
#                          vmax = np.round(inv_simple[colby].mean()+inv_simple[colby].std(), 1))

# nt_norm = mcol.Normalize(0.1, 0.25)                                                             
nt_norm = mcol.Normalize(1, 6)                                                             
# nt_norm = mcol.Normalize(3, 15)                                                             
# nt_norm = mcol.Normalize(vmin = np.floor(inv_simple[colby].min()), 
#                           vmax = np.round(inv_simple[colby].max(), 1))

turnover_cm = get_continuous_cmap(turn_cols)
#%% get correlation coefficients for each variable, pick the strongest ones in each category type (e.g. climate, discharge etc)

# corrcoeffs = pd.DataFrame(columns = ['Kendall Tau', 'Spearman', 'Pearson'], index=inv_simple.columns)

# for col in inv_simple.columns:
#     corrcoeffs.loc[col, 'Kendall Tau'] = inv_simple['num_turns_mean'].corr(inv_simple[col], method = 'kendall')
#     corrcoeffs.loc[col, 'Spearman'] = inv_simple['num_turns_mean'].corr(inv_simple[col], method = 'spearman')
#     corrcoeffs.loc[col, 'Pearson'] = inv_simple['num_turns_mean'].corr(inv_simple[col], method = 'pearson')

# corrcoeffs = corrcoeffs.drop(index = ['mean_tt_length', 'num_turns_mean'])
# plt.figure(figsize = (6, 3), dpi = 300)
# plt.stem(corrcoeffs['Kendall Tau'], markerfmt = 'black', linefmt = 'k', label = 'Kendall')
# plt.stem(corrcoeffs['Pearson'], markerfmt = 'red', linefmt = 'r', label = 'Pearson')
# plt.stem(corrcoeffs['Spearman'], markerfmt = 'blue', linefmt = 'b', label = 'Spearman')

# plt.legend()
# ax = plt.gca()
# ax.set_xticks(range(len(corrcoeffs))) ## alays set tick range before prescribing labels
# ax.set_title('Correlation between mean num. turnovers and inventory variables')
# ax.set_xticklabels(corrcoeffs.index.values);

# ax.xaxis.set_tick_params(rotation=70)
# ax.set_ylabel('Correlation Coefficient')

#%% see QQ plots of distributions

'''Key Variables from the correlation analyses:
    [B] = boundary variable, [F] = form & mobility variable, [T] = turnover variable
    X[B] -- Q norm to catchment area
    [B] -- bedload proportion
    [B] -- Unit discharge
    [B] -- Unit sed flux
    [B] -- Grain size
    [B] -- Slope
    [B] -- NDVI
    [B] -- Aridity index 
    X[B] -- DVIc (weaker than ARI)
    [B] -- DVIa (seasonality index, might be useful)
    
    [F] -- Efficiency
    [F] -- EBI
    [F] -- Length turnovevr shape parameter
    [F] -- Std. Dev. of EBI
    
    [T] -- mean number of turnovers
    [T] -- mean length of turnover
'''
### define parmeters to use in each PCA run
boundary_params = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr',  #'num_turns_mean',
                   'mean_slope', 'DVIc', 'ndvi_mean']#, 'turn_freq'] ## NDVI here cannot be mapped for AGUBH2, try with nan to see

form_mob_params = ['efficiency', 'med_ebi', 
                 'mean_slope', ]#'turn_freq']#, 'corridor_prop_turned'] ## can include AGUBH2 no data manipulation

# combo_params = ['unit_discharge_m2s', 'unit_sedflux_m2yr', #, 
#                 'DVIc', 'ndvi_mean',  #'max_tt_mean',
#                 'efficiency', 'med_ebi','mean_slope']
## initial params
combo_params = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', #, 
                  'mean_slope', 'DVIc', 'ndvi_mean',  #'max_tt_mean',
                'efficiency', 'med_ebi', ]#'turn_freq']

# visualise QQ plots to see the distributions and the distributions of log data


fig, ax = plt.subplots(3, 5, figsize = (11, 8), tight_layout = True, dpi = 150)
ax = ax.ravel()

for a, col in enumerate(combo_params):
    samples = pd.to_numeric(inv_simple[col]).dropna()
    sm.qqplot(samples, dist = stats.norm, fit = True, line = 's', ax = ax[a], markerfacecolor = 'k') 
    ax[a].set_title(col)

    sm.qqplot(np.log10(samples), dist = stats.norm, fit = True, line = None, ax = ax[a],  markerfacecolor = 'r')
    
#%% take logs to make distributions more normal, and standardise data

params_to_log = ['bed_prop_of_total', 'unit_discharge_m2s', 'unit_sedflux_m2yr', 'part_size_mm', 
                  'mean_slope', 'Tm_timescale', 'Tr_timescale', 'stream_pow']#,after viewing the qq plot declare the vars to take logs of

inv_boundary = inv_simple[inv_simple.columns.intersection(boundary_params)].copy(deep=True)
inv_form_mob = inv_simple[inv_simple.columns.intersection(form_mob_params)].copy(deep=True)
inv_combo = inv_simple[inv_simple.columns.intersection(combo_params)].copy(deep=True)
 
for col in params_to_log:
    if col in inv_boundary.columns:
        inv_boundary[col] = np.log10(inv_simple[col])

    if col in inv_form_mob.columns:
        inv_form_mob[col] = np.log10(inv_simple[col])

    if col in inv_combo.columns:
        inv_combo[col] = np.log10(inv_simple[col])
        
invb_std = inv_boundary.copy(deep=True)
invf_std = inv_form_mob.copy(deep=True)
invc_std = inv_combo.copy(deep=True)

invb_std = (invb_std-invb_std.mean(axis = 0))/invb_std.std(axis = 0)  
invf_std = (invf_std-invf_std.mean(axis = 0))/invf_std.std(axis = 0)              
invc_std = (invc_std-invc_std.mean(axis = 0))/invc_std.std(axis = 0)       
 
## ensure standardisation worked
print(invb_std.mean(axis = 0))
print(invb_std.std(axis = 0))        

print(invf_std.mean(axis = 0))
print(invf_std.std(axis = 0))        

print(invc_std.mean(axis = 0))
print(invc_std.std(axis = 0))        


ncomp_b = len(invb_std.columns)
ncomp_f = len(invf_std.columns)
ncomp_c = len(invc_std.columns)
#%% Perform PCA includes turnover in form and combo

pca_boundary = PCA(n_components=ncomp_b, svd_solver = 'auto')
pca_form = PCA(n_components=ncomp_f, svd_solver = 'auto')
pca_combo = PCA(n_components=ncomp_c, svd_solver = 'auto')

## boundary data
pCs_bound = pca_boundary.fit_transform(invb_std)   ## perform the transform (calc cov matrix transforming the data to 2x2) get principal components  
evr_bound = (pca_boundary.explained_variance_ratio_) 
cutoff_bound = np.where(np.cumsum(evr_bound) > .95)[0][0] ## cutoff PCs whwen cumulative sum of variance exceeds 95%
bound_load = pca_boundary.components_ ## component loadings of the PCA

## form data
pCs_form = pca_form.fit_transform(invf_std)   ## perform the transform (calc cov matrix transforming the data to 2x2)  get principal components
evr_form = (pca_form.explained_variance_ratio_) 
cutoff_form = np.where(np.cumsum(evr_form) > .95)[0][0] ## cutoff PCs whwen cumulative sum of variance exceeds 95%
form_load = pca_form.components_ ## component loadings of the PCA (how much variables weigh on the PCs)

## combo data
pCs_combo = pca_combo.fit_transform(invc_std)   ## perform the transform (calc cov matrix transforming the data to 2x2) get principal components 
evr_combo = (pca_combo.explained_variance_ratio_) 
cutoff_combo = np.where(np.cumsum(evr_combo) > .95)[0][0] ## cutoff PCs whwen cumulative sum of variance exceeds 95%
combo_load = pca_combo.components_ ## component loadings of the PCA

## make the principal component a GIANT dataframe with turnover information for plotting
#%% mke DF of pca results for plotting, but only with first two PCs for each test
pca_results = pd.concat((pd.DataFrame(pCs_bound[:, :2], index = inv_simple.index.values, columns = ['bPC1', 'bPC2']),
                         pd.DataFrame(pCs_form[:, :2], index = inv_simple.index.values, columns = ['fPC1', 'fPC2']),
                         pd.DataFrame(pCs_combo[:, :3], index = inv_simple.index.values, columns = ['cPC1', 'cPC2', 'cPC3']),
                         inv_simple), axis = 1)

#%% VISUALISE!!!

## first, the pca loadings for up to the cutoff amount of PCs for each df

fig, ax = plt.subplots(3, 1, figsize = (6, 6), dpi = 300, tight_layout = True)
cutoffs = [cutoff_bound, cutoff_form, cutoff_combo]
loads = [bound_load, form_load, combo_load]
load_labels = [invb_std.columns, invf_std.columns, invc_std.columns]
evrs = [evr_bound, evr_form, evr_combo]
titles = ['Boundary', 'Form & Mobility', 'Combination']
lws = [2, 1.75, 1.5, 1.25, .75, .5, .5, .5, .25, .25, .25, .25, .25, .25]
for a, ax in enumerate(ax.ravel()):
    for comp in range (0, cutoffs[a]+1):
        ax.plot(loads[a][comp], label = comp, lw = lws[comp])
        ax.legend(bbox_to_anchor=(1.1, 1))    
    ax.set_title(titles[a])
    ax.set_xticks(range(len(load_labels[a]))) ## alays set tick range before prescribing labels
    ax.set_xticklabels(load_labels[a]);


    ax.xaxis.set_tick_params(rotation=70)

fig, ax = plt.subplots(1, 3, figsize = (10, 4), dpi = 300, tight_layout = True, sharey = True)
for a, ax in enumerate(ax.ravel()):
    ax.plot(np.cumsum(evrs[a]), marker = 'o', c = 'k')
    ax.axvline(cutoffs[a]+1, c = 'r', ls = '--')
    ax.axhline(.75, c = 'k', ls = '--', alpha = 0.5)
    ax.set_xlabel('principal components')
    ax.set_ylabel('Cumulative expected variance for components')
    ax.set_title(titles[a])

# colby = 'stat_mean_turn'

abr_rivnames = ['AGU', 'ADD', 'ADN', 'BET', 'BHA',  'BHP', 'COL', 'CON', 'IND', 'IRA', 'IRU', 'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK', 'TAN', 'YUE', 'YUK']
# fig, ax = plt.subplots(1, 3, figsize = (10, 4), dpi = 300, tight_layout = True, sharey = True)
# ax[0].scatter(pca_results['bPC1'], pca_results['bPC2'], c = pca_results[colby], ec = 'k', s = 100,
#               cmap = 'inferno', norm = nt_norm)
# for i in range(len(abr_rivnames)):
#     ax[0].annotate(abr_rivnames[i], (pca_results['bPC1'][i]+0.25, pca_results['bPC2'][i]+0.1), fontsize = 6, color = 'm')
# ax[0].set_xlabel('PC0')
# ax[0].set_ylabel('PC1')
# ax[0].set_title('Boundary variables')

# ax[1].scatter(pca_results['fPC1'], pca_results['fPC2'], c = pca_results[colby], ec = 'k', s = 100,
#               cmap = 'inferno', norm = nt_norm)
# for i in range(len(abr_rivnames)):
#     ax[1].annotate(abr_rivnames[i], (pca_results['fPC1'][i]+0.25, pca_results['fPC2'][i]+0.1), fontsize = 6, color = 'm')
# ax[1].set_xlabel('PC0')
# ax[1].set_ylabel('PC1')
# ax[1].set_title('Form & Mobility variables')

# leg = ax[2].scatter(pca_results['cPC1'], pca_results['cPC2'], c = pca_results[colby], ec = 'k', s = 100,
#               cmap = 'inferno', norm = nt_norm)
# for i in range(len(abr_rivnames)):
#     ax[2].annotate(abr_rivnames[i], (pca_results['cPC1'][i]+0.25, pca_results['cPC2'][i]+0.1), fontsize = 6, color = 'm')
# ax[2].set_xlabel('PC0')
# ax[2].set_ylabel('PC1')
# ax[2].set_title('Boundary and form variables') 
# plt.colorbar(leg, ax = ax[2], label = colby)

#%% k-means clustering -- no good
# '''https://medium.com/more-python-less-problems/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2'''

# plt.figure(figsize = (6, 5), dpi = 150)
# ks = range(1, 10)
# inertias = []
# for k in ks:
#     # Create a KMeans instance with k clusters: model
#     model = KMeans(n_clusters=k)
    
#     # Fit model to samples
#     model.fit(pca_results.loc[:, ['cPC1', 'cPC2']])
    
#     # Append the inertia to the list of inertias
#     inertias.append(model.inertia_)
    
# plt.plot(ks, inertias, '-o', color='black')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()

# ## given an elbow location on the kmeans plot, plot the clusters (inertia points)
# # https://www.datacamp.com/datalab/templates/recipe-python-k-means
# elbow = 3
# kmeans = KMeans(n_clusters = elbow)
# pred_y = kmeans.fit_predict(pca_results.loc[:, ['cPC1', 'cPC2']])

# plt.figure(figsize = (5, 5), dpi = 150)
# plt.scatter(pca_results[:, 'cPC1'], pca_results[:, 'cPC2'], marker = 'o', c = pca_results[colby], ec = 'k', s = 100,
#               cmap = 'inferno', norm = nt_norm)
# plt.scatter(kmeans.cluster_centers_[:, 0], 
#             kmeans.cluster_centers_[:, 1], 
#             s=200,                             # Set centroid size
#             c='red') 
# abr_rivnames = ['ADD', 'ADN', 'BET', 'BHA', 'BHP', 'COL', 'CON', 'IND', 'IRA', 'IRU', 'KAS', 'LEN', 'MAN', 'OBD', 'OBU', 'RAK', 'SSK', 'TAN', 'YUE', 'YUK']
# for i, label in enumerate(abr_rivnames):
#     plt.text(pcs_cutoff[:, 0][i], pcs_cutoff[:, 1][i], label)
# plt.xlabel('PC0')
# plt.ylabel('PC1')
# plt.title(f'if elbow = {elbow}')

#%% plot loadings on plot: COMBO

#propt = ([0.6, 0.5, 0.4, 0.7, 0.7, 0.6, 0.5, 0.1, 0.5, 0.5, 0.5, 0.3, 0.2,
#       0.6, 0.1, 0.1, 0.8, 0.4, 0.3, 0.1, 0.2]) ## proportion of channel corridor turned over, rounded, making 5 sizes 0-.2, -.4, -.6, -.8, -1

sizeby_propt = np.array([0.6, 0.4, 0.4, 0.6, 0.6, 0.6, 0.4, 0.1, 0.4, 0.4, 0.4, 0.2, 0.2, 0.6, 0.1, 0.1, 0.8, 0.4, 0.2, 0.1, 0.2])
scaler = 3

plt.figure(figsize = (5, 5,), dpi = 300)
ax = plt.gca()

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis='both', which='minor', labelcolor = 'k')
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))
ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))

plt.grid(True, 'major')
# cs = pca_results[colby]*prop_turned
# cs = inventory[colby]*prop_turned
# csnorm = mcol.Normalize(cs.min(), cs.max())
# propt_norm = mcol.Normalize(prop_turned.min(), prop_turned.max())

## PLOT HERE
biplot = plt.scatter(pca_results['cPC1'], pca_results['cPC2'], c = pca_results[colby], ec = 'k', s = 350*sizeby_propt,
# biplot = plt.scatter(pca_results['cPC1'], pca_results['cPC2'], c = [colby], ec = 'k', s = 300,#*sizeby_propt,
              cmap = turnover_cm, norm = nt_norm,) 
plt.xlabel('PC1')
plt.ylabel('PC2')

## loadiings for arrows on biplot
loadings_pca1 = scaler*pca_combo.components_[0]
loadings_pca2 = scaler*pca_combo.components_[1]

names = ['$Q_{bfrac}$', '$uQ_{w}$', 'u$Q_{s}$', 'Eff.', 'eBI$_{med}$', 'slope', 'NDVI', '$DVI_{c}$',  ]
# names = ['Qw', '$uQ_{s}$', 'eBI$_{med}$', 'Eff', 'Slope', 'NDVI', '$DVI_{c}$']#,  'stream_pow']
# names = inv_combo.columns
for n, name in enumerate(names):
    plt.arrow(0, 0, loadings_pca1[n], loadings_pca2[n], length_includes_head = False,
              head_width = .1, head_length = .2, fc = 'k')
    plt.annotate(name, (loadings_pca1[n], loadings_pca2[n]), color = 'g')
    
for i in range(len(abr_rivnames)):
    plt.annotate(abr_rivnames[i], (pca_results['cPC1'][i]+0, pca_results['cPC2'][i]+0), fontsize = 6, color = 'm')

# fig.colorbar(biplot, label = 'mean number of Turn', shrink = 0.75)
fig.colorbar(biplot, label = 'mean n turnovers', shrink = 0.75)
plt.title('PC1 and PC2 for all variables')
# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/fnl_log10PCA_combo_statmeanT.svg')
# plt.savefig('/Users/safiya/Desktop/dissertation_wtf/defense_slide_figs/pca_combo_bhp2.png')
#%% Plot for boundary conditions only
plt.figure(figsize = (5, 5,), dpi = 300)
ax = plt.gca()
scaler = 2
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis='both', which='minor', labelcolor = 'k')
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))
ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))

plt.grid(True, 'major')

biplot = plt.scatter(pca_results['bPC1'], pca_results['bPC2'], c = pca_results[colby], ec = 'k', s = 300*sizeby_propt,
                     cmap = turnover_cm, norm = nt_norm)
plt.xlabel('PC1')
plt.ylabel('PC2')

## loadiings for arrows on biplot
loadings_pca1 = scaler*pca_boundary.components_[0]
loadings_pca2 = scaler*pca_boundary.components_[1]

names = ['$Q_{bfrac}$', '$uQ_{w}$', 'u$Q_{s}$', 'slope', 'NDVI',  '$DVI_{c}$']


for n, name in enumerate(names):
    plt.arrow(0, 0, loadings_pca1[n], loadings_pca2[n], length_includes_head = False,
              head_width = .1, head_length = .2, fc = 'k')
    plt.annotate(name, (loadings_pca1[n], loadings_pca2[n]), color = 'g')
    
for i in range(len(abr_rivnames)):
    plt.annotate(abr_rivnames[i], (pca_results['bPC1'][i]+0.25, pca_results['bPC2'][i]+0.1), fontsize = 6, color = 'm')

fig.colorbar(biplot, label = 'mean number of turns', shrink = 0.75)
plt.title('PC1 and PC2 for boundary conditions')
# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/log10PCA_boundary.svg')
plt.savefig('/Users/safiya/Desktop/dissertation_wtf/defense_slide_figs/pca_boundary_bhp2.png')

#%% Plot for form conditions only
plt.figure(figsize = (5, 5,), dpi = 300)
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis='both', which='minor', labelcolor = 'k')
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))
ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))

plt.grid(True, 'major')


biplot = plt.scatter(pca_results['fPC1'], pca_results['fPC2'], c = pca_results[colby], ec = 'k', s = 300*sizeby_propt,
              cmap = turnover_cm, norm = nt_norm)
plt.xlabel('PC1')
plt.ylabel('PC2')

## loadiings for arrows on biplot
loadings_pca1 = scaler*pca_form.components_[0]
loadings_pca2 = scaler*pca_form.components_[1]

names = [ 'Eff.', 'eBI$_{med}$', '$slope_{mean}$']

for n, name in enumerate(names):
    plt.arrow(0, 0, loadings_pca1[n], loadings_pca2[n], length_includes_head = False,
              head_width = .1, head_length = .2, fc = 'k')
    plt.annotate(name, (loadings_pca1[n], loadings_pca2[n]), color = 'g')
    
for i in range(len(abr_rivnames)):
    plt.annotate(abr_rivnames[i], (pca_results['fPC1'][i]+0.25, pca_results['fPC2'][i]+0.1), fontsize = 6, color = 'm')
plt.title('PC1 and PC2 for form conditions')
fig.colorbar(biplot, label = 'mean number of turns', shrink = 0.75)

# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/log10PCA_form.svg')
plt.savefig('/Users/safiya/Desktop/dissertation_wtf/defense_slide_figs/pca_form_bph2.png')

#%% plot PC2 and PC3 loadings on plot: COMBO
scaler = 3

plt.figure(figsize = (5, 5,), dpi = 300)
ax = plt.gca()

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis='both', which='minor', labelcolor = 'k')
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))
ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%0.1f'))

plt.grid(True, 'major')

biplot = plt.scatter(pca_results['cPC2'], pca_results['cPC3'], c = pca_results[colby], ec = 'k', s = 300*sizeby_propt,
              cmap = turnover_cm, norm = nt_norm) 
plt.xlabel('PC2')
plt.ylabel('PC3')

## loadiings for arrows on biplot
loadings_pca1 = scaler*pca_combo.components_[1]
loadings_pca2 = scaler*pca_combo.components_[2]

names = ['$Q_{bfrac}$', '$uQ_{s}$', 'u$Q_{w}$', 'Eff.', 'eBI$_{med}$', 'slope', 'NDVI', '$DVI_{c}$']

for n, name in enumerate(names):
    plt.arrow(0, 0, loadings_pca1[n], loadings_pca2[n], length_includes_head = False,
              head_width = .1, head_length = .2, fc = 'k')
    plt.annotate(name, (loadings_pca1[n], loadings_pca2[n]), color = 'g')
    
for i in range(len(abr_rivnames)):
    plt.annotate(abr_rivnames[i], (pca_results['cPC2'][i]+0.25, pca_results['cPC3'][i]+0.1), fontsize = 6, color = 'm')

fig.colorbar(biplot, label = 'mean number of turns', shrink = 0.75)
plt.title('PC1 and PC2 for all variables')
# plt.savefig('/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/log10PCA_combo_pc23.svg')
plt.savefig('/Users/safiya/Desktop/dissertation_wtf/defense_slide_figs/pca_combo_pc2_pc3_bhp2.png')






