#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:23:59 2024

@author: safiya
"""

import numpy as np
import matplotlib.pyplot as plt
import stackmasks
import whackamole
import remove_land

#%%
maskpath = "/Volumes/SAF_Data/CHAPTER2/greenberg-area-data/full-dataset/greenberg_AreaMobility_Data/NaturalRiverData/SSaskatchOutlook/mask/"
ssk, diffs, comp = maskstack_diffs(maskpath)

floodplain_reworking = fp_rework_ari(ssk)
ch_overlap = ch_overlap_area_ami(ssk, diffs)
t_end = ssk.shape[0]
end_year = 2021

years = np.arange(end_year-t_end, end_year)

## convert pixels to area
resolution = 900 #m2 -> km2, 1m2 = 1e-6 km2, 1 landsat px = 30m x 30m = 900m2
unit = 1e6
floodplain_reworking = (floodplain_reworking*resolution)/unit ##in km2
ch_overlap = (ch_overlap*resolution)/unit ##in km2

time = np.arange(1, len(ssk))

#%% comput channel medians and normalise

wetted_area = np.apply_over_axes(np.sum, ssk, [1, 2]).flatten()
plt.plot(years, wetted_area)
plt.axhline(np.median(wetted_area), c = 'r', ls = '-', label = 'median')
plt.title('wetted area through time');
plt.legend();

idx = np.where(wetted_area==0)[0]
plt.figure(figsize = (4, 8), dpi = 200)
questionable = np.squeeze(ssk[idx, :, :], axis = 0)
plt.imshow(questionable, cmap = 'Grays')

#%% make figure, evan metrics
fig2, ax = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True, dpi = 200)

amit = ax[0].scatter(time, ch_overlap, marker = 'o', c = 'g', ec = 'k')
arit = ax[1].scatter(time, floodplain_reworking, marker = 'o', c = 'r', ec = 'k')

ax[0].set_ylabel('Channel Overlap Area, A$_mi$')
ax[1].set_ylabel('Floodplain Reworking Area, A$_ri$')

ax[0].set_xlabel('Duration since baseline');
ax[1].set_xlabel('Duration since baseline');

#%% 

sskw2d, sskd2w = pixel_composite(ssk, diffs)


## inital metric figures, hotspot map and evan figures 

fig1, ax = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True, dpi = 200)
wd = ax[0].imshow(sskw2d, cmap = 'pink_r', vmin = 0, vmax = np.nanmax(sskw2d))
dw = ax[1].imshow(sskd2w, cmap = 'Blues', vmin = 0, vmax = np.nanmax(sskd2w))

ax[0].set_title('stacked wet to dry px events')
ax[1].set_title('stacked dry to wet px events')

fig1.colorbar(wd, ax = ax[0])
fig1.colorbar(dw, ax = ax[1])

ssk_flags, ssk_turn_time = pixel_turn_time(ssk, diffs)

#%%

summap = plt.figure(figsize = (4, 8), dpi = 400, tight_layout = True)
comp = plt.imshow(np.sum(ssk, axis = 0), cmap = 'Blues')
summap.colorbar(comp, aspect = 25, shrink = .9)
plt.title('Composite watermask of South Saskatchewan \n with Greenberg Masks');

num_switches = np.count_nonzero(~np.isnan(ssk_turn_time), axis = 0)
num_switches = num_switches[num_switches > 0]

plt.figure(figsize = (5,5), dpi =200)
plt.hist(num_switches.flatten(), fc = 'grey', ec = 'k', bins = np.arange(0, t_end))
plt.title('SSK: number of switching events at each pixel location')
plt.ylabel('Count')

## distribution of lendths pf pixel turnover times

turntime_dist = plt.figure(figsize = (5, 5), tight_layout = True)
plt.hist(ssk_turn_time.flatten(), bins = np.arange(0, t_end), fc = 'grey', ec = 'k');
plt.title('Distribution of all the stasis period lengths wet AND dry');

turnwet_dry, ax = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True, sharex = True, sharey = True)
w2d = ax[0].hist(ssk_turn_time[ssk_flags[:, :, :]==-1].flatten(), fc = 'xkcd:azure', ec = 'k',  bins = np.arange(0, t_end), density = True);
d2w = ax[1].hist(ssk_turn_time[ssk_flags[:, :, :]==1].flatten(), fc = 'xkcd:shit brown', ec = 'k', bins = np.arange(0, t_end), density = True);
    
ax[0].set_title('Length of wet spells (wet cells ended by drying events)');
ax[1].set_title('Length of dry spells (dry cells ended by wetting events)');



