#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:06:32 2024

@author: safiya
"""
import numpy as np
import copy

def pixel_composite(maskstack, diffs):
    '''import stack of masked imagery and return a time composite of 
       hotspots of wet/dry pixels through the time period, and a total composite of wet pixels through time'''
    
    wet2dry = copy.deepcopy(diffs)
    dry2wet = copy.deepcopy(diffs)

    wet2dry[wet2dry != -1] = 0
    dry2wet[dry2wet != 1] = 0
    
    ## stack wet2dry and dry2wet cubes by finding the sum along the time axis (i.e. axis = 0)
    
    stackw2d = abs(np.sum(wet2dry, axis = 0))
    stackd2w = abs(np.sum(dry2wet, axis = 0))
    
    composite = np.sum(maskstack, axis = 0)
    return stackw2d, stackd2w, composite

def fp_rework_ari(maskstack, baseline = 0):
    '''Calculate floodplain reworking area, A_ri from Wickert et al 2013 and Greenberg et al 2023
       using total area of dry px in baseline channel mask, Ad, and the dry px area yet to be visited
       by the channe in the first i masks, Ai'''
       
    ari = np.array([]) 
    ad = np.count_nonzero(maskstack[baseline, :, :]==0)
    sumstack = copy.deepcopy(maskstack[baseline, :, :])
    
    for t in range(baseline+1, maskstack.shape[0]):
        sumstack = sumstack + maskstack[t, :, :]
        unvisited_fp = np.count_nonzero(sumstack==0)
        ari = np.append(ari, [ad-unvisited_fp])
        
    return ari

def ch_overlap_area_ami(maskstack, diffs, baseline = 0):
    '''Calculate the channel ovelap area, companion metric to fp reworking area (Greenberg et al 2023 and Wickert et al, 2013)
       calculated using the wetted channel area in baseline channel mask, Aw, area that has changed state when comparing the ith
       channelmask to the baseline mask Di (= [Did2w+ Diw2d]/2)'''
    
    aw = np.count_nonzero(maskstack[baseline, :, :]==1)
    ami = np.array([])
    
    wet_grid = copy.deepcopy(maskstack)
    
    # wet_grid = np.zeros_like(maskstack)
    # wet_grid[baseline, :, :][maskstack[baseline, :, :]==1] = 1 ##assign baseline of the wetgrid to be water values like maskstack
    
    for t in range(baseline+1, maskstack.shape[0]):
        
        ##inherit the baseline wetted areas into the timestep being evaluated
        
        wet_grid[t, :, :][wet_grid[t-1, :, :]==-1] = -1
        wet_grid[t, :, :][wet_grid[t-1, :, :]==1] = 1
            
        wet_grid[t, :,:][diffs[t-1, :, :]==-1] ## reassign water pixels as wet to dry pixels
        wet_grid[t, :,:][diffs[t-1, :, :]==1]  ## reassign pixels as dry to wet pixels
        
        diwd = np.count_nonzero(wet_grid[t, :, :]==-1)
        didw = np.count_nonzero(wet_grid[t, :, :]==1)
        
        di = 0.5*(didw+diwd)
        
        ami = np.append(ami, [aw-di])
        # print(ami)
        
    return ami

def pixel_turn_time(maskstack, diffs):
    '''calculate the turover time (the length of time a pixel spends wet or dry in each wet/dry cycle)'''
    base_diffs = np.append(np.zeros([1, diffs.shape[1], diffs.shape[2]]), diffs, axis = 0) ##pad diffs array with plane of zeros in first plane
    wet_dry_flags = np.zeros_like(maskstack) ## make array of zeros with shape of mask stack to house wet dry flags, everything dry will replace with wet flags
    px_turn_time = np.zeros_like(maskstack, dtype = 'float')  # length of pixel turnover

    px_turn_time[:] = np.nan ##make all the turnover times nan
    t_end = maskstack.shape[0] ## find the end time for this mask stack to act as an avulsion proxy
    for lat in range(maskstack.shape[1]):
        for long in range(maskstack.shape[2]):
            
            ## find where the differences between timesteps are non zero, this would ignore all land pixels on the channel periphery that are inactive
            ptt = np.nonzero(base_diffs[:, lat, long])[0].tolist() ## this pulls out the time INDICES from the base diffs array where the pixels change.
                                                                    # the [0] just pulls the first dimension of the return bc its 2D by default and makes it a list
            
            if len(ptt) > 0: ## len(ptt) = 0 for all unchanged land cells in the channel periphery so we want to focus on the coordinates that are active
                wet_dry_flags[ptt, lat, long] = base_diffs[ptt, lat, long] ## pull the difference values and add them into the wet dry flags matrix--isnt this effectively the same aas base_diffs?
                # print(wet_dry_flags[ptt, lat, long], ptt, np.diff(ptt))
                # px_turn_time[ptt, lat, long] = np.diff(ptt) ## need a better way to put in ptt of locations that meet the t_end condition before the if statement
                
                if ptt[-1] != t_end:
                    ptt_av = np.append(ptt, t_end) ## because we want to have the end time as a cap on the turnover times too?        
                    px_turn_time[ptt, lat, long] = np.diff(ptt_av)
                else: 
                    px_turn_time[ptt, lat, long] = np.diff(ptt)
    return wet_dry_flags, px_turn_time 




















            