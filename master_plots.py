#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:11:16 2024

@author: safiya
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

num_systems = int(input('how many rivers being plotted here? '))

#%% Import test case stacks, using skip datasets

tan_stack, tan_diffs = maskstack_diffs_skips()
ind_stack, ind_diffs = maskstack_diffs_skips()
ira_stack, ira_diffs = maskstack_diffs_skips()
amd_stack, amd_diffs = maskstack_diffs_skips()

tan_stackw2d, tan_stackd2w, tancomp = pixel_composite(tan_stack, tan_diffs)
ind_stackw2d, ind_stackd2w, indcomp = pixel_composite(ind_stack, ind_diffs)
ira_stackw2d, ira_stackd2w, iracomp = pixel_composite(ira_stack, ira_diffs)
amd_stackw2d, amd_stackd2w, amdcomp = pixel_composite(amd_stack, amd_diffs)


tan_wet_dry_time_flags, tan_px_tt = pixel_turn_time(tan_stack, tan_diffs)
ind_wet_dry_flags, ind_px_tt = pixel_turn_time(ind_stack, ind_diffs)
ira_wet_dry_flags, ira_px_tt = pixel_turn_time(ira_stack, ira_diffs)
amd_wet_dry_flags, amd_px_tt = pixel_turn_time(amd_stack, amd_diffs)

#%% Figure List
'''
1. H/gram of the length of pixel turnover times globally for each system
2. H/gram of the number of switching events per location
3. Composite map of the area (see hotspots of change)

'''
## fig 1: ptt hist
ptt_hist, ax = plt.subplots(2, num_systems, figsize = (10, 10), tight_layout = True, sharex = True, sharey = True, dpi = 400)
yearlykwargs = dict(density = True, bins = np.arange(0, 36), alpha = 0.5, fc = 'xkcd:stone', ec = 'k')

ax[0, 0].hist(tan_px_tt.ravel(), **yearlykwargs)
ax[0, 0].set_title('Tanana');
ax[0, 0].set_xlabel('Length of Px. Turn Time');

ax[0, 1].hist(ind_px_tt.ravel(), **yearlykwargs)
ax[0, 1].set_title('Indus');
ax[0, 1].set_xlabel('Length of Px. Turn Time');

ax[0, 0].set_ylabel('Density');

ax[1, 0].hist(ira_px_tt.ravel(), **yearlykwargs)
ax[1, 0].set_title('Irrawaddy');
ax[1, 0].set_xlabel('Length of Px. Turn Time');

ax[1, 1].hist(amd_px_tt.ravel(), **yearlykwargs)
ax[1, 1].set_title('Amu Darya');
ax[1, 1].set_xlabel('Length of Px. Turn Time');

ax[1, 0].set_ylabel('Density');
ax[0, 0].set_xlim(0, 36);

#%%
## fig 2: num switches hist

num_switches_tan = np.count_nonzero(~np.isnan(tan_px_tt), axis = 0)
num_switches_tan = num_switches_tan[num_switches_tan > 0]

num_switches_ira = np.count_nonzero(~np.isnan(ira_px_tt), axis = 0)
num_switches_ira = num_switches_ira[num_switches_ira > 0]

num_switches_ind = np.count_nonzero(~np.isnan(ind_px_tt), axis = 0)
num_switches_ind = num_switches_ind[num_switches_ind > 0]

num_switches_amd = np.count_nonzero(~np.isnan(amd_px_tt), axis = 0)
num_switches_amd = num_switches_amd[num_switches_amd > 0]

switch_hist, ax = plt.subplots(2, num_systems, figsize = (10, 10), tight_layout = True, sharex = True, sharey = True, dpi = 400)
yearlykwargs = dict(density = True, alpha = 0.5, bins = np.arange(0, 36), fc = 'xkcd:stone', ec = 'k')

ax[0, 0].hist(num_switches_tan.ravel(), **yearlykwargs)
ax[0, 0].set_title('Tanana');
ax[0, 0].set_xlabel('Number of switches per px loc');

ax[0, 1].hist(num_switches_ind.ravel(), **yearlykwargs)
ax[0, 1].set_title('Indus');
ax[0, 1].set_xlabel('Number of switches per px loc');

ax[0, 0].set_ylabel('Density');

ax[1, 0].hist(num_switches_ira.ravel(), **yearlykwargs)
ax[1, 0].set_title('Irrawaddy');
ax[1, 0].set_xlabel('Number of switches per px loc');

ax[1, 1].hist(num_switches_amd.ravel(), **yearlykwargs)
ax[1, 1].set_title('Amu Darya');
ax[1, 1].set_xlabel('Number of switches per px loc');

ax[1, 0].set_ylabel('Density');
ax[0, 0].set_xlim(0, 36);

#%%
## figure 3: composite time
vmax = 36
vmin = 0
comp, ax = plt.subplots(2, 2, figsize=(10, 10), tight_layout = True, dpi = 400)
tanana_comp = ax[0, 0].imshow(tancomp, cmap = 'BrBG', vmin = vmin, vmax = vmax)
ax[0, 0].set_title('Tanana');
comp.colorbar(tanana_comp, ax = ax[0, 0], label = 'Hotspots of channel activity');

indus_comp = ax[0, 1].imshow(indcomp, cmap = 'BrBG', vmin = vmin, vmax = vmax)
ax[0, 1].set_title('Indus');
comp.colorbar(indus_comp, ax = ax[0, 1], label = 'Hotspots of channel activity');

irrawaddy_comp = ax[1, 0].imshow(iracomp, cmap = 'BrBG', vmin = vmin, vmax = vmax)
ax[1, 0].set_title('Irrawaddy');
comp.colorbar(irrawaddy_comp, ax = ax[1, 0], label = 'Hotspots of channel activity');

amudarya_comp = ax[1, 1].imshow(amdcomp, cmap = 'BrBG', vmin = vmin, vmax = vmax)
ax[1, 1].set_title('Amu Darya');
comp.colorbar(amudarya_comp, ax = ax[1, 1], label = 'Hotspots of channel activity');
