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

def maskstack_diffs():
    '''make a stack of masks from the  puller imagers output. For this to work masks must be exactly the same size and binary images'''
    
    river = input('river:') 
    maskpath = input('path:')
    
    resolution = (30*30)/1e-6 ##landsat resolution in km2
    mask_names = os.listdir(maskpath)
    df = pd.DataFrame(columns = ['year', 'wet px', 'wet area'])
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
        yr = int(year)
        im = Image.open(f'{maskpath}{file}')
        maskstack[t, :, :] = im
        
        ax[t].imshow(im, cmap = 'gray')
        ax[t].set_title(f'{river} {yr}')
        
        nwet = np.sum(maskstack[t, :, :])
        area_wet = nwet*resolution
        
        ## build up summary dataframe for the system
        summary = pd.DataFrame([[year, nwet, area_wet]], columns = df.columns)
        df = pd.concat([df, summary], axis = 0)
        
    compilation.savefig(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_compilation/{river}_compilation_stackmasks.png', dpi = 400)    
    
    df.set_index('year')    
    diffs = np.diff(maskstack, axis = 0)
    df.to_csv(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/{river}_summary.csv')
    
    return maskstack, diffs
