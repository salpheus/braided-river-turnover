#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:00:14 2024

@author: safiya
"""
import numpy as np 
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import stackmasks_skip
import whackamole
import copy




timeseries = pd.read_csv('/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/tanana_summary.csv', header = 0)


plt.plot(timeseries['year'], timeseries['wet px'])

tan_stack, tan_diffs = maskstack_diffs_skips()

## for the congo only 
tan_stack = tan_stack[-9:, :, :]
tan_diffs = tan_diffs[-8:, :, :]
counter = np.arange(0, len(tan_stack))

time_stack = [counter[t] * tan_stack[t, :, :] for t in range(len(tan_stack))] ## make stack like tan_stack but with the time the water pixel is replaced

n_times = len(tan_stack)
water_palette = copy.deepcopy(tan_stack[0, :, :]) ## first timestep of the watermask, will be filled in with the water pixels at each time
for t in range(1, n_times):
    mix = water_palette + tan_stack[t, :, :]
    
    # where mix == 2 = water that stayed water, do nothing; mix == 1 = new water, inherit new time; where mix == 0, still dry, do nothing
    water_palette[mix == 1] = t

    # water_palette[mix==1] = t
    # print(np.unique(mix))
water_palette_mask = ma.masked_equal(water_palette, 0)
plt.figure(figsize = (10, 10), dpi = 300)
plt.title('Time at which each pixel went wet for the first time')    
plt.imshow(water_palette_mask, cmap = 'rainbow')
plt.colorbar()

freq, bins = np.histogram(water_palette_mask, bins = np.arange(1, n_times+1)) ## get the frequency of each number in the tally then I will plot the cum sum
csum_tan = np.cumsum(freq)

plt.figure('cumsum', figsize = (4,4), dpi = 300, tight_layout = True)
plt.plot(np.arange(1, n_times), csum_tan, marker = 'o', ls = '-', c = 'k')
plt.ylabel('cumulative wet pixel count per time step');
plt.xlabel('Timestep');

tan_flags, tan_ptt = pixel_turn_time(tan_stack, tan_diffs)
tan_ptt_water = copy.deepcopy(tan_ptt)
tan_ptt_dry = copy.deepcopy(tan_ptt)
tan_ptt_water[tan_flags==-1] = np.nan ## pull only the wet turnover events
tan_ptt_dry[tan_flags==1] = np.nan ## pull only the dry turnover events

longest_sp_wet = np.nanmax(tan_ptt_water, axis = 0)
longest_sp_dry = np.nanmax(tan_ptt_dry, axis = 0)

plt.figure('sp_wet', figsize = (5, 5), dpi = 300)
plt.hist(longest_sp_wet.ravel(), bins = np.arange(1, n_times), fc = 'xkcd:light grey blue', ec = 'k', label = 'wet')
plt.hist(longest_sp_dry.ravel(), bins = np.arange(1, n_times), ec = 'xkcd:shit green', histtype = 'step', lw = 2, label = 'dry')
plt.ylabel('Longest state persistence')
plt.legend()
plt.xlabel('Time, years');


sum_switches = np.sum(abs(tan_flags), axis = 0)
plt.figure(dpi = 400, tight_layout = True)
plt.imshow(sum_switches, cmap = 'copper')
plt.title('Hotspot map sum_switches')
plt.colorbar()

plt.figure(dpi = 400, tight_layout = True)
plt.hist(sum_switches[sum_switches>0].ravel(), bins = np.arange(1, 18), ec = 'k')
plt.ylabel('sum_switches');