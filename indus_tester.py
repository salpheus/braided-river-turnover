#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:11:51 2024

@author: safiya
"""
import numpy as np
import matplotlib.pyplot as plt
import stackmasks
import whackamole
import remove_land

#%% PULL THE DATA

##make the indus stack and the differences array
indus_stack, indus_diffs = maskstack_diffs('/Volumes/SAF_Data/remote-data/watermasks/indusforever/indus/mask/')

## make the staked wet to dry and dry to wet pixel composites
indusw2d, indusd2w = pixel_composite(indus_stack, indus_diffs)

## get the published floodplain and channel overlap metrics
floodplain_reworking = fp_rework_ari(indus_stack)
ch_overlap = ch_overlap_area_ami(indus_stack, indus_diffs)

## get the pixel turnover
indus_flags, indus_turn_time = pixel_turn_time(indus_stack, indus_diffs)

#%% Make some plots of things

## inital metric figures, hotspot map and evan figures 

fig1, ax = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True, dpi = 200)
wd = ax[0].imshow(indusw2d, cmap = 'pink_r', vmin = 0, vmax = np.nanmax(indusw2d))
dw = ax[1].imshow(indusd2w, cmap = 'Blues', vmin = 0, vmax = np.nanmax(indusw2d))

ax[0].set_title('stacked wet to dry px events')
ax[1].set_title('stacked dry to wet px events')

fig1.colorbar(wd, ax = ax[0])
fig1.colorbar(dw, ax = ax[1])

## evan and andy mobility metrics from paper

fig2, ax = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True, dpi = 200)

amit = ax[0].plot(ch_overlap, 'g')
arit = ax[1].plot(floodplain_reworking, 'r')

ax[0].set_ylabel('Channel Overlap Area, A$_mi$')
ax[1].set_ylabel('Floodplain Reworking Area, A$_ri$')

ax[0].set_xlabel('Duration since baseline');
ax[1].set_xlabel('Duration since baseline');

## distribution of lendths pf pixel turnover times

turntime_dist = plt.figure(figsize = (5, 5), tight_layout = True)
plt.hist(indus_turn_time.flatten());
plt.title('Distribution of all the stasis period lengths wet AND dry');

turnwet_dry, ax = plt.subplots(1, 2, figsize = (10, 5), tight_layout = True, sharex = True, sharey = True)
w2d = ax[0].hist(indus_turn_time[indus_flags[:, :, :]==-1].flatten(), fc = 'xkcd:shit brown', bins = np.arange(0, 36), density = True);
d2w = ax[1].hist(indus_turn_time[indus_flags[:, :, :]==1].flatten(), fc = 'blue', bins = np.arange(0, 36), density = True);
    
ax[0].set_title('start time of dry to wet turnover');
ax[1].set_title('start time of wet to dry turnover');

#%%
## 
indus_nan = make_land_nan(indus_stack)
overlapmap = plt.figure(figsize = (10, 10), dpi = 200) 
plt.imshow(indus_nan[0, :, :], alpha = .5)
plt.imshow(indus_nan[1, :, :], alpha = .5)

totalwet = plt.figure(figsize = (10, 10), dpi = 300)
totwet = plt.imshow(np.nansum(indus_stack, axis = 0), cmap = 'Blues')
totalwet.colorbar(totwet)

totmin = np.nanmin(np.nansum(indus_stack, axis = 0))
totmax = np.nanmax(np.nansum(indus_stack, axis = 0))

xs, ys = np.where(np.nansum(indus_stack, axis =0)==np.nanmax(np.nansum(indus_stack, axis = 0)))
plt.plot(ys, xs, 'r*')

#%%

for x, y in zip(xs, ys):
    print(indus_stack[:, x, y])
    print()
    
#%%

plt.figure(figsize = (10,10), dpi = 500)
plt.imshow(indus_stack[0, :, :])
plt.plot(ys, xs, 'r*')

#%%
num_switches = np.count_nonzero(~np.isnan(indus_turn_time), axis = 0)
num_switches = num_switches[num_switches > 0]

plt.figure(figsize = (5,5), dpi =200)
plt.hist(num_switches.flatten(), fc = 'grey', ec = 'k', bins = np.arange(0, 15))
plt.title('number of switching events at each pixel location')
plt.ylabel('Count')

