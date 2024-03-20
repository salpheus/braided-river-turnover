#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:55:28 2024

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

base_dataset = ['amguema', 'amudarya', 'Congo', 'congo_bumba_sambo', 'indus', 'irrawaddy', 'jamuna', 'kasai', 
                'kasaidown', 'rakaia', 'southsask', 'tanana', 'ubangi', 'waitaki', 'yukon']

base_dataset_colours = ['xkcd:light blue', 'xkcd:baby pink', 'xkcd:evergreen', 'xkcd:jungle green', 'xkcd:sage green', 'xkcd:dirty green', 'xkcd:muddy green', 'xkcd:shit', 
                'xkcd:dirt', 'xkcd:steel', 'xkcd:greyblue', 'xkcd:dark grey blue', 'xkcd:camouflage green', 'xkcd:bluegrey', 'xkcd:flat blue']
t_init = 1987
t_start = 2013
t_end = 2021
years = np.arange(t_start, t_end+1)
allyears = np.arange(t_init, t_end+1)
levels = 10 ## define number of levels for the colormap
full_levels = len(allyears)
colors = plt.cm.rainbow(np.linspace(0, 1, levels))
rainbow_discrete = ListedColormap(colors)
rainbow_full = ListedColormap(plt.cm.rainbow(np.linspace(0, 1, full_levels)))

#%% plot pristine grids

root = '/Volumes/SAF_Data/remote-data'
pristine_stack_path = os.path.join(root, 'arrays/pristine/maskstack/')
pristine_diffs_path = os.path.join(root, 'arrays/pristine/diffs/')
full_stack_path = os.path.join(root, 'arrays/full/maskstack/')

'''
1. "pri"--Poster of the masks with time nrivers x nyears
2. "pal--Palette maps of time to wet maps for each river [defined geometry, 3 x 5]
3. "ptt"--H/gram of the length of pixel turnover times globally for each system [defined geometry, 3 x 5]
4. "hsw"--H/gram of the number of switching events per location [defined geometry, 3 x 5]
5. "wet"--composite map of the wet areaa (see hotspots of change) [defined geometry, 3 x 5]
6. "dry"--composite map of all the dry areas [defined geometry, 3 x 5]
7. longest time in wet per pixel location (normalised for something to compare across rivers) [defined geometry, 3 x 5]
8. longest time in dry per pixel location (normalised for something to compare across rivers)? maybe not useful bc floodplain & bars [defined geometry, 3 x 5]
9. "nsw"--map number of state switches per px location (to show where in channels are most active) [defined geometry, 3 x 5]

'''

normdf = pd.DataFrame(index = years, columns = base_dataset) ## will store the data for the weird norm metric cumulative annual conv count/ (max polygon-median wet px area)
occupationdf = pd.DataFrame(index = years, columns = base_dataset) ## will store the data for the normalised values of how much of each channel belt area is occupied through time
pri, ax = plt.subplots(len(base_dataset), len(years), dpi = 400, figsize = (24, 36), squeeze= True)
pal, axs = plt.subplots(3, 5, dpi = 400, figsize = (20, 15), tight_layout = True, squeeze= True)
# nsw, nax = plt.subplots(3, 5, dpi = 400, figsize = (20, 15), tight_layout = True, squeeze= True)

ptt, pax = plt.subplots(3, 5, dpi = 400, figsize = (15, 7), tight_layout = True, squeeze= True, sharex = True, sharey = True)
hsw, hax = plt.subplots(3, 5, dpi = 400, figsize = (15, 7), tight_layout = True, squeeze= True, sharex = True, sharey = True)

# wet, wax = plt.subplots(3, 5, dpi = 400, figsize = (15, 5), tight_layout = True, squeeze= True)

yearlykwargs = dict(density = True, bins = np.arange(0, len(years)), ec = 'k')

for row, river in enumerate(base_dataset):
    #ravel all axes
    a = axs.ravel() #
    p = pax.ravel() #
    h = hax.ravel() #
    # w = wax.ravel()
    # n = nax.ravel() #
    
    
    stack = np.load(os.path.join(pristine_stack_path, f'{river}_pstack.npy'))
    diffs = np.load(os.path.join(pristine_diffs_path, f'{river}_diffs_pstack.npy'))
    
    flags, px_turn_time = pixel_turn_time(stack, diffs) 
    
    # plot number of switches per px location [fig 4]
    n_switches = np.count_nonzero(~np.isnan(px_turn_time), axis = 0)
    n_switches = n_switches[n_switches>0]
    h[row].hist(n_switches.ravel(), **yearlykwargs, fc = base_dataset_colours[row])
    h[row].set_title(base_dataset[row])
    h[row].set_xlabel('N switches per x loc')
    h[row].set_ylabel('Density')
    h[row].set_xlim(1, len(years))
    # plt.savefig('/Users/safiya/Desktop/c2_local_exploration/num_switches_hist.png')
    # map number of switches per px loc [fig 9]
    # n[row].imshow(n_switches, cmap = rainbow_discrete, vmin = 0, vmax = 9) ## doesnt work bc n_switches is a list! fix next week
    
    # histogram of ptt [fig 3]
    p[row].hist(px_turn_time.ravel(), **yearlykwargs, fc = base_dataset_colours[row])
    p[row].set_ylabel('Density')
    p[row].set_xlabel('Length of pixel turnover time')
    p[row].set_title(base_dataset[row])
    p[row].set_xlim(1, len(years))
    # plt.savefig('/Users/safiya/Desktop/c2_local_exploration/ptt_length_hist.png')
    #
    
    water_palette = copy.deepcopy(stack[0, :, :])#.astype('float')
    
    ## reassign the first timestep water pixels to 0 meaning they are water at time = 0
    water_palette[water_palette == 0] = -999 ##turn t=0 land to -999
    water_palette[water_palette==1] = 0 ## give water px at t=0 a time to wet ==0

    

    for t, yr in enumerate(years):
        ax[row, t].imshow(stack[t, :, :], cmap = 'gray')
        ax[row, t].set_title(f'{river} {yr}')
        
    for ts in range(1, len(stack)):
        mix = water_palette + stack[ts, :, :] ## adding [0, -999] of [0, 1]s together to get mix of [0, 1, -999, -998]s or [0, 1, 2]s
        '''if mixing at the first timestep you can combine numbers in the following ways: 
          0->0 (water to land) = 0
          0->1 (water to water) = 1
          -999 -> 0 (land to land) = -999
          -999 -> 1 (land to water) = -998
          
          at t = 2 you can combine like:
              1-> 0 water to land = 1 (BUT WE DON'T WANT THIS IN PALETTE)
              1-> 1 water to water = 2 (we also don't want this)
             -999 -> 0 (land to land) = -999
             -999 -> 1 (land to water) = -998 (again this is what we want!)
         --- trying the approach where we just flag this array for every timestep where there is a -998 flag and replacing that with the timestep in question?
        '''
        water_palette[mix==-998] = ts ## because we need to differentiate between 0+1 (not new water pixels) and -999+1 (new water px)
    
    max_area = np.count_nonzero(water_palette>=0) ## calculate number of pixels that are water
    median_area = np.median(np.apply_over_axes(np.sum, stack, [1, 2])) ## median of the pixel sum across all axes (pixel area)
    
    

    #water_palette[np.logical_and(stack[0, :, :]==0, stack[1, :, :]==1)] = 1
    water_palette_mask = ma.masked_equal(water_palette, -999)
    
    freq, bins = np.histogram(water_palette_mask, bins = np.arange(0, len(years)+1)) ## get the frequency of each number in the tally then I will plot the cum sum
    csum_wet_px_area = np.cumsum(freq) ## get the running total of wetted pixel area
    
    # norm_for_df = (max_area-csum_wet_px_area)/median_area
    norm_for_df = csum_wet_px_area/max_area
    weirdnorm = freq/(max_area-median_area)
    occupationdf[river] = norm_for_df
    normdf[river] = weirdnorm
    
    palmap = a[row].imshow(water_palette_mask, cmap = rainbow_discrete, vmin = 0, vmax = 9)
    a[row].set_title(f'Time to wet {river}')    
    plt.colorbar(palmap, ax = a[row], shrink = 0.75)
    
    plt.savefig('/Users/safiya/Desktop/c2_local_exploration/time_to_wet-saturday.svg')

#%% Repeat for fullstack data

full_occupationdf = pd.DataFrame(index = allyears, columns = base_dataset) ## will store the data for the normalised values of how much of each channel belt area is occupied through time
pal, axs = plt.subplots(3, 5, dpi = 400, figsize = (20, 15), tight_layout = True, squeeze= True)


for row, river in enumerate(base_dataset):
    full_stack = np.load(os.path.join(full_stack_path, f'{river}_fullstack.npy'))
    fwater_palette = copy.deepcopy(full_stack[0, :, :])#.astype('float')
    
    ## reassign the first timestep water pixels to 0 meaning they are water at time = 0
    fwater_palette[fwater_palette == 0] = -999 ##turn t=0 land to -999
    fwater_palette[fwater_palette==1] = 0 ## give water px at t=0 a time to wet ==0

    a = axs.ravel()

    for ts in range(1, len(full_stack)):
        mix = fwater_palette + full_stack[ts, :, :] ## adding [0, -999] of [0, 1]s together to get mix of [0, 1, -999, -998]s or [0, 1, 2]s

        fwater_palette[mix==-998] = ts ## because we need to differentiate between 0+1 (not new water pixels) and -999+1 (new water px)
    
    max_area = np.count_nonzero(fwater_palette>=0) ## calculate number of pixels that are water
    median_area = np.median(np.apply_over_axes(np.sum, full_stack, [1, 2])) ## median of the pixel sum across all axes (pixel area)

    #water_palette[np.logical_and(stack[0, :, :]==0, stack[1, :, :]==1)] = 1
    fwater_palette_mask = ma.masked_equal(fwater_palette, -999)
    
    freq, bins = np.histogram(fwater_palette_mask, bins = np.arange(0, len(allyears)+1)) ## get the frequency of each number in the tally then I will plot the cum sum
    csum_wet_px_area = np.cumsum(freq) ## get the running total of wetted pixel area
    
    # norm_for_df = (max_area-csum_wet_px_area)/median_area
    norm_for_df = csum_wet_px_area/max_area
    full_occupationdf[river] = norm_for_df
    
    palmap = a[row].imshow(fwater_palette_mask, cmap = rainbow_full, vmin = 0, vmax = len(allyears))
    a[row].set_title(f'Time to wet {river}')    
    plt.colorbar(palmap, ax = a[row], shrink = 0.75)
    
    # plt.savefig('/Users/safiya/Desktop/c2_local_exploration/time_to_wet_allyears.svg')
    
#%% plot the normalisation statsitics 

plt.figure('pristine-norms', figsize = (10, 6), dpi = 300)
ax = plt.gca()
occupationdf.plot(lw = 2, marker = 'o', ms = 3, color = base_dataset_colours, ax = ax)
plt.legend(labels = occupationdf.columns, bbox_to_anchor = (1.01, 1))
ax.set_ylabel('time to wet/max wet px area')
ax.set_xlabel('year');

plt.figure('pristine-norms-weird', figsize = (10, 6), dpi = 300)
ax = plt.gca()
normdf.plot(lw = 2, marker = 'o', ms = 3, color = base_dataset_colours, ax = ax)
plt.legend(labels = occupationdf.columns, bbox_to_anchor = (1.01, 1))
ax.set_ylabel('time to wet/(max-med wet px area)')
ax.set_xlabel('year');


plt.figure('full-norms', figsize = (10, 6), dpi = 300)
ax = plt.gca()
full_occupationdf.plot(lw = 2, marker = 'o', ms = 3, color = base_dataset_colours, ax = ax)
plt.legend(labels = full_occupationdf.columns, bbox_to_anchor = (1.01, 1))
ax.set_ylabel('time to wet/max wet px area')
ax.set_xlabel('year');

#%% 

