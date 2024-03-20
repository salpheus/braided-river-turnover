#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:31:01 2024

@author: safiya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# get folder
# get list of files in folder/for mask in folder
# pull first file, get size, store dimensions. We want m dimen to write to the n direction and n dimen to write to k direction

def maskstack(img = False, save = False):
    '''make a stack of masks from the  puller imagers output. For this to work masks must be exactly the same size and binary images'''
    
    river = input('river, case-sensitive!:') 
    maskpath = input('path:')
    
    resolution = (30*30)/1e-6 ##landsat resolution in km2
    mask_names = os.listdir(maskpath)
    summary = pd.DataFrame(columns = ['year', 'wet px', 'wet area'])
    ## get dimensions from the first mask in the folder, should be the same for all files
    dims = np.shape(Image.open(os.path.join(maskpath, mask_names[0])))
    # dims = np.shape(Image.open(f'{maskpath}{mask_names[0]}'))
    lat = dims[0]
    long = dims[1]
    
    maskstack = np.empty([len(mask_names), lat, long], dtype = int)
    
    compilation, ax = plt.subplots(6, 6, sharex = True, sharey = True, tight_layout = True, dpi = 400, figsize = (18, 18))
    
    ax = ax.ravel()
    
    for t, file in enumerate(mask_names):
        
        year = file.split(f'{river}_')[1].split("_", 1)[0].strip()
        
        im = Image.open(f'{maskpath}{file}')
        maskstack[t, :, :] = im
        
        nwet = np.sum(maskstack[t, :, :])
        area_wet = nwet*resolution
        
        ## build up summary dataframe for the system
        df = pd.DataFrame([[year, nwet, area_wet]], columns = summary.columns)
        summary = pd.concat([summary, df], axis = 0, ignore_index = True)
        
        if img:
            yr = int(year)
            ax[t].imshow(im, cmap = 'gray')
            ax[t].set_title(f'{river} {yr}')
    
    to_delete = summary.index[summary['wet px']==0].tolist()
    print('Missing years ', summary.loc[summary['wet px'] == 0, 'year'])
    
    diffs = np.diff(maskstack, axis = 0)
    
    skipstack = np.delete(maskstack, to_delete, axis = 0)
    skipdiffs = np.diff(skipstack, axis = 0)
    
    summary.set_index('year')    
    summary.to_csv(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/{river}_summary_skip.csv')
    
    pristine = maskstack[-9:, :, :] ## 2013-2021 pristine record for all rivers on file
    pristine_diffs = np.diff(pristine, axis = 0)
    
    if save:  ## save all versions of the array and compilation image
        compilation.savefig(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_compilation/{river}_compilation_stackmasks_skip.png', dpi = 400)    
        
        np.save(f'/Volumes/SAF_Data/remote-data/arrays/full/maskstack/{river}_fullstack.npy', maskstack, allow_pickle = True) ##save maskstack
        np.save(f'/Volumes/SAF_Data/remote-data/arrays/full/diffs/{river}_diffs_fullstack.npy', diffs, allow_pickle = True) ## save maskstack diffs 
        
        np.save(f'/Volumes/SAF_Data/remote-data/arrays/skip/maskstack/{river}_skipstack.npy', skipstack, allow_pickle = True) ##save maskstack
        np.save(f'/Volumes/SAF_Data/remote-data/arrays/skip/diffs/{river}_diffs_skipstack.npy', skipdiffs, allow_pickle = True) ## save maskstack diffs 
        
        np.save(f'/Volumes/SAF_Data/remote-data/arrays/pristine/maskstack/{river}_pstack.npy', pristine, allow_pickle = True) ##save maskstack
        np.save(f'/Volumes/SAF_Data/remote-data/arrays/pristine/diffs/{river}_diffs_pstack.npy', pristine_diffs, allow_pickle = True) ## save maskstack diffs
    
    return 