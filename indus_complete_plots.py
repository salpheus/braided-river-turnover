#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 07:21:02 2024

@author: safiya
"""
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import copy


import whackamole

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 10}

mpl.rc('font', **font)

#%%

root = '/Volumes/SAF_Data/remote-data/' 
riv = 'indus_r2''
indus_stack_path = os.path.join(root, 'arrays/indus_complete/full/maskstack/{riv}_fullstack.npy')
indus_diffs_path = os.path.join(root, 'arrays/indus_complete/full/diffs/{riv}_diffs_fullstack.npy')

indus_stack = np.load(indus_stack_path)
indus_diffs = np.load(indus_diffs_path)

ind_flags, ind_ptt = pixel_turn_time(indus_stack, indus_diffs)

ind_w2d, ind_d2w, composite = pixel_composite(indus_stack, indus_diffs)

start_t = 1987
end_t = 2022
years = np.arange(start_t, end_t+1)

# compute water palette
water_palette = copy.deepcopy(indus_stack[0, :, :])#.astype('float')

## reassign the first timestep water pixels to 0 meaning they are water at time = 0
water_palette[water_palette == 0] = -999 ##turn t=0 land to -999
water_palette[water_palette==1] = 0 ## give water px at t=0 a time to wet ==0

for ts in range(1, len(indus_stack)):
     mix = water_palette + indus_stack[ts, :, :] ## adding [0, -999] of [0, 1]s together to get mix of [0, 1, -999, -998]s or [0, 1, 2]s
    
     water_palette[mix==-998] = ts ## because we need to differentiate between 0+1 (not new water pixels) and -999+1 (new water px)
 
# max_area = np.count_nonzero(water_palette>=0) ## calculate number of pixels that are water
water_palette_mask = ma.masked_equal(water_palette, -999)
 
# freq, bins = np.histogram(water_palette_mask, bins = np.arange(0, len(years)+1)) ## get the frequency of each number in the tally then I will plot the cum sum
# csum_wet_px_area = np.cumsum(freq) ## get the running total of wetted pixel area

max_wetted_area = np.count_nonzero(composite > 0) ## in px
conversion_times = np.unique(water_palette_mask)

num_converted_px = []  ## will house the numper of pixels converted per timestep
for c in conversion_times.compressed():
    ncp = len(np.where(water_palette==c)[0])
    num_converted_px.append(ncp)

num_converted_px_norm = np.divide(num_converted_px, max_wetted_area) # number pf pixels converted per ts, normalised by max wetted area

num_switches = np.count_nonzero(~np.isnan(ind_ptt), axis = 0) ## number of switches at each pixel location

max_ptt = np.nanmax(ind_ptt, axis = 0)

## make array of only wet ptts
ind_ptt_wet = copy.deepcopy(ind_ptt)
ind_ptt_wet[ind_flags==1] = np.nan ## keep only wet pixel times
ind_ptt_wet = ma.masked_equal(ind_ptt_wet, np.nan)

## make array of only dry ptts
ind_ptt_dry = copy.deepcopy(ind_ptt)
ind_ptt_dry[ind_flags==-1] = np.nan ## keep only wet pixel times
ind_ptt_dry = ma.masked_equal(ind_ptt_dry, np.nan)

#%% plot some thangs

## aesthetics setup

switchmap_levels = num_switches.max() ## max number of switches for switchmap
full_levels = len(years)
colors = plt.cm.rainbow(np.linspace(0, 1, full_levels)) #raimbow cmap for water palette map for now
rainbow_discrete = ListedColormap(colors)
rainbow_full = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, full_levels)))
blues_full = ListedColormap(plt.cm.PuBu_r(np.linspace(0, 1, full_levels)))
diffs_levels = np.nanmax(ind_w2d)
w2d_cm = ListedColormap(plt.cm.Oranges(np.linspace(0, 1, diffs_levels)))
d2w_cm = ListedColormap(plt.cm.Blues(np.linspace(0, 1, diffs_levels)))
composite_cm = ListedColormap(plt.cm.Greys(np.linspace(0, 1, full_levels)))
numswitches_cm = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, switchmap_levels)))

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"]) make colormaps from colours

# hotspot maps
fig, ax = plt.subplots(1, 3, figsize = (15, 6), tight_layout = True, dpi = 300)
w2d = ax[0].imshow(ind_w2d, cmap = w2d_cm, alpha = 0.7, vmin = 0, vmax = np.nanmax(ind_w2d))
ax[0].set_title('Indus 1987-2022, Wet to Dry Hotspot')
plt.colorbar(w2d, ax = ax[0])

d2w = ax[1].imshow(ind_d2w, cmap = d2w_cm, alpha = 0.7, vmin = 0, vmax = np.nanmax(ind_d2w))
ax[1].set_title('Indus 1987-2022, Dry to Wet Hotspot')
plt.colorbar(d2w, ax = ax[1])

comp = ax[2].imshow(composite, cmap = composite_cm, vmin = 0, vmax = np.nanmax(composite))
ax[2].set_title('Indus 1987-2022, Activity Hotspot Map (absolute vals)')
plt.colorbar(comp, ax = ax[2])

## water palette
plt.figure(figsize = (6, 8), dpi = 300, tight_layout = True)
palmap = plt.imshow(water_palette_mask, cmap = blues_full, vmin = 0, vmax = full_levels)
plt.title('Time each Indus pixel went wet from 1987-2022')
plt.colorbar(palmap, shrink = .75, aspect = 35)
plt.savefig('/Users/safiya/Desktop/c2_local_exploration/for-colloquium/indus_waterpal.svg')
## time series graph of it
plt.figure(figsize = (10, 4), tight_layout = True, dpi = 150)
plt.plot(years, num_converted_px_norm, 'k-', marker = 'o', ms = 5)
plt.title('Trend in pixel conversion within the active channel domain')
plt.ylabel('Number of pixels converted normalised \n to max wetted pixel area')
plt.xlabel('Year')

## num_switches through time
plt.figure(figsize = (4, 6), dpi = 300, tight_layout = True)
palmap = plt.imshow(num_switches, cmap = numswitches_cm, vmin = 0, vmax = switchmap_levels)
plt.title('Amount of turnover at each \n pixel location from 1987-2022')
plt.colorbar(palmap, shrink = .75, aspect = 35, label = 'Number of switches')

## pixel turn time histogram
## 3 panel subplots
# 1. len of turnover
# 2. len of wet turnover
# 3. len of dry turnover

fig, ax = plt.subplots(1, 3, figsize = (15, 6), tight_layout = True, dpi = 300, sharex = True, sharey = True)
ax[0].hist(ind_ptt.ravel(), ec = 'k', fc = 'xkcd:stone', bins = np.arange(full_levels), density = True)
ax[0].set_title('Indus 1987-2022 all turnover')
ax[0].set_xlabel('Length of turnover time')

ax[1].hist(ind_ptt_wet.ravel(), ec = 'k', fc = 'xkcd:sky blue', bins = np.arange(full_levels), density = True)
ax[1].set_title('Indus 1987-2022 wet turnover')
ax[1].set_xlabel('Length of turnover time')

ax[2].hist(ind_ptt_dry.ravel(), ec = 'k', fc = 'xkcd:light brown', bins = np.arange(full_levels), density = True)
ax[2].set_title('Indus 1987-2022 dry turnover')
ax[2].set_xlabel('Length of turnover time')
ax[0].set_ylabel('Density')
##histogram of length of max turnover times spatially

fig, ax = plt.subplots(1, 1, figsize = (4, 4), tight_layout = True, dpi = 300, sharex = True, sharey = True)
ax.hist(max_ptt.ravel(), ec = 'k', fc = 'xkcd:off white', bins = np.arange(full_levels))
ax.set_title('Indus 1987-2022 \n length of max turnover at each pixel')
ax.set_xlabel('Turnover time length')
ax.set_ylabel('Count')
## add max turnovver histogram wet and dry

# map of max turnover time spatially


