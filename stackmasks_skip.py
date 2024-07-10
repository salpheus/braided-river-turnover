#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:31:01 2024

@author: safiya
"""
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.plot import adjust_band
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from PIL import Image

# get folder
# get list of files in folder/for mask in folder
# pull first file, get size, store dimensions. We want m dimen to write to the n direction and n dimen to write to k direction
# Normalize bands into 0.0 - 1.0 scale
# def normalize(array):

#     array_min, array_max = np.nanmin(array), np.nanmax(array)
#     return ((array - array_min) / (array_max - array_min))

def normalize(array):
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    if array_max - array_min == 0:
        return array  # Avoid division by zero if all elements are the same
    return (array - array_min) / (array_max - array_min)

def maskstack(img = False, save = False):
    '''make a stack of masks from the  puller imagers output. For this to work masks must be exactly the same size and binary images'''

    src = input('ENTER Parent folder: ')
    rivlist = os.listdir(f'/Volumes/SAF_Data/remote-data/watermasks/{src}/')
    print(rivlist)
    
    for l, river in enumerate(rivlist): 
        print(f'processing {river}. Its number {l+1} out of {len(rivlist)}')
        # river = input('river, case-sensitive!:') 
        # parentfolder= input('folder containing the masks:')
        
        # maskpath = os.path.join(parentpath, 'mask')
    
        resolution = (30*30)/1e-6 ##landsat resolution in km2
        mask_names = glob.glob(f'/Volumes/SAF_Data/remote-data/watermasks/{src}/{river}/mask/*.tif') ##pull paths to all masks
        image_names = glob.glob(f'/Volumes/SAF_Data/remote-data/watermasks/{src}/{river}/image/*.tif') ##pull paths to all images
        
        summary = pd.DataFrame(columns = ['year', 'wet px', 'wet area'])
        # print(mask_names)
        
        ## get dimensions from the first mask in the folder, should be the same for all files
        dims = np.shape(Image.open(mask_names[0]))
        lat = dims[0]
        long = dims[1]
        
        maskstack = np.empty([len(mask_names), lat, long], dtype = int)
        
        compilation, ax = plt.subplots(6, 6, sharex = True, sharey = True, tight_layout = True, dpi = 400, figsize = (18, 18))
        
        ax = ax.ravel()
        
        for t, info in enumerate(zip(mask_names, image_names)):
            mpath = info[0] 
            impath = info[1]
            file = mpath.split('/')[-1]
            # fullimg = impath.split('/')[-1]
            
            year = file.split(f'{river}_')[1].split("_", 1)[0].strip()
            
            ##load images for maskstack array
            im = Image.open(mpath)
            maskstack[t, :, :] = im
            
            nwet = np.sum(maskstack[t, :, :])
            area_wet = nwet*resolution
            
            # build plot image
            mask = (rasterio.open(mpath).read(1))*1.0
            mask[mask<1.0] = np.nan ## make everything but channel transparent (this is a fudge could make this more refined)
            
            composite = rasterio.open(impath)
            
            # if np.isnan(composite).any() or np.isinf(composite).any():
            #     print("Data contains NaNs or Infinities")
            #     # Replace NaNs and Infinities with a specific value, e.g., zero
            #     composite = np.nan_to_num(composite, nan=0.0, posinf=0.0, neginf=0.0)
            
            
            red = composite.read(4) ## Band 6
            green = composite.read(3) ## Band 5
            blue = composite.read(2) ## Band 4, python indexing
        
            ##  alt rasterio.adjust_band
            red_norm = normalize(red)
            green_norm = normalize(green)
            blue_norm = normalize(blue)
            # 
            fnl = np.dstack((blue_norm, green_norm, red_norm))
            
            ## build up summary dataframe for the system
            df = pd.DataFrame([[year, nwet, area_wet]], columns = summary.columns)
            summary = pd.concat([summary, df], axis = 0, ignore_index = True)
            
            if img:
                yr = int(year)
                ax[t].imshow(fnl)
                ax[t].imshow(mask, cmap = 'gray')
                ax[t].set_title(f'{river} {yr}')
        
        to_delete = summary.index[summary['wet px']==0].tolist()
        print('Missing years ', summary.loc[summary['wet px'] == 0, 'year'])
        
        diffs = np.diff(maskstack, axis = 0)
        
        skipstack = np.delete(maskstack, to_delete, axis = 0)
        skipdiffs = np.diff(skipstack, axis = 0)
        
        summary.set_index('year')    
        summary.to_csv(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/C02_fromGaleazzi_db_csv/{river}_sum_summary_skip.csv')
        
        pristine = maskstack[-9:, :, :] ## 2013-2021 pristine record for all rivers on file
        pristine_diffs = np.diff(pristine, axis = 0)
        
        if save:  ## save all versions of the array and compilation image
        
            saveroot = '/Volumes/SAF_Data/remote-data/arrays/C02_fromGaleazzi_db/'
            compilation.savefig(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_compilation/C02_fromGaleazzi_db_poster/{river}_compilation_sum_stackmasks_skip.png', dpi = 400)    
            
            np.save(os.path.join(saveroot, 'full/maskstack/', f'{river}_fullstack.npy'), maskstack, allow_pickle = True) ##save maskstack
            np.save(os.path.join(saveroot, 'full/diffs/', f'{river}_diffs_fullstack.npy'), diffs, allow_pickle = True) ## save maskstack diffs 
            
            np.save(os.path.join(saveroot, 'skip/maskstack/', f'{river}_skipstack.npy'), skipstack, allow_pickle = True) ##save maskstack
            np.save(os.path.join(saveroot, 'skip/diffs/', f'{river}_diffs_skipstack.npy'), skipdiffs, allow_pickle = True) ## save maskstack diffs 
            
            np.save(os.path.join(saveroot, 'pristine/maskstack/', f'{river}_pstack.npy'), pristine, allow_pickle = True) ##save maskstack
            np.save(os.path.join(saveroot, 'pristine/diffs/', f'{river}_diffs_pstack.npy'), pristine_diffs, allow_pickle = True) ## save maskstack diffs
        
        return 