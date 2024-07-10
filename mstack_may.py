#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:13:12 2024

@author: safiya
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

def normalize(array):
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    if array_max - array_min == 0:
        return array  # Avoid division by zero if all elements are the same
    return (array - array_min) / (array_max - array_min)

def concatenate_dataframes(summary, df):
    if not df.empty and not df.isna().all().all():
        summary = pd.concat([summary, df], axis=0, ignore_index=True)
    return summary

def process_river(river, src, img, save, resolution):
    mask_names = glob.glob(f'/Volumes/SAF_Data/remote-data/watermasks/{src}/{river}/mask/*.tif')
    image_names = glob.glob(f'/Volumes/SAF_Data/remote-data/watermasks/{src}/{river}/image/*.tif')
    
    summary = pd.DataFrame(columns=['year', 'wet px', 'wet area'])
    
    dims = np.shape(Image.open(mask_names[0]))
    lat, long = dims[0], dims[1]
    
    maskstack = np.empty([len(mask_names), lat, long], dtype=int)
    if img:
        compilation, ax = plt.subplots(5, 8, sharex=True, sharey=True, tight_layout=True, dpi=400, figsize=(16, 10))
        ax = ax.ravel()

    for t, (mpath, impath) in enumerate(zip(mask_names, image_names)):
        file = os.path.basename(mpath)
        year = file.split(f'{river}_')[1].split("_", 1)[0].strip()
        
        im = Image.open(mpath)
        maskstack[t, :, :] = im
        
        nwet = np.sum(maskstack[t, :, :])
        area_wet = nwet * resolution
        
        mask = (rasterio.open(mpath).read(1)) * 1.0
        mask[mask < 1.0] = np.nan
        
        composite = rasterio.open(impath)
        
        red = composite.read(4)
        green = composite.read(3)
        blue = composite.read(2)
        
        red_norm = normalize(red)
        green_norm = normalize(green)
        blue_norm = normalize(blue)
        
        fnl = np.dstack((blue_norm, green_norm, red_norm))
        
        df = pd.DataFrame([[year, nwet, area_wet]], columns=summary.columns)
        summary = concatenate_dataframes(summary, df)
        
        if img:
            yr = int(year)
            ax[t].imshow(fnl)
            ax[t].imshow(mask, cmap='gray')
            ax[t].set_title(f'{river} {yr}')
    
    to_delete = summary.index[summary['wet px'] == 0].tolist()
    print('Missing years ', summary.loc[summary['wet px'] == 0, 'year'])
    
    diffs = np.diff(maskstack, axis=0)
    skipstack = np.delete(maskstack, to_delete, axis=0)
    skipdiffs = np.diff(skipstack, axis=0)
    
    summary.set_index('year')
    summary.to_csv(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/C02_1987-2023_allLS_db_csv/{river}_sum_summary_skip.csv')
    
    pristine = maskstack[-9:, :, :]
    pristine_diffs = np.diff(pristine, axis=0)
    
    if save:
        saveroot = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/'
        compilation.savefig(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_compilation/C02_1987-2023_allLS_db_poster/{river}_compilation_sum_stackmasks_skip.png', dpi=400)
        
        np.save(os.path.join(saveroot, 'full/maskstack/', f'{river}_fullstack.npy'), maskstack, allow_pickle=True)
        np.save(os.path.join(saveroot, 'full/diffs/', f'{river}_diffs_fullstack.npy'), diffs, allow_pickle=True)
        
        np.save(os.path.join(saveroot, 'skip/maskstack/', f'{river}_skipstack.npy'), skipstack, allow_pickle=True)
        np.save(os.path.join(saveroot, 'skip/diffs/', f'{river}_diffs_skipstack.npy'), skipdiffs, allow_pickle=True)
        
        np.save(os.path.join(saveroot, 'pristine/maskstack/', f'{river}_pstack.npy'), pristine, allow_pickle=True)
        np.save(os.path.join(saveroot, 'pristine/diffs/', f'{river}_diffs_pstack.npy'), pristine_diffs, allow_pickle=True)

def maskstack(img=False, save=False):
    src = input('ENTER Parent folder: ')
    #rivlist = os.listdir(f'/Volumes/SAF_Data/remote-data/watermasks/{src}/')
    rivlist = ['agubh2']
    print(rivlist)
    
    resolution = (30 * 30) / 1e-6  # landsat resolution in km2
    
    for l, river in enumerate(rivlist): 
        print(f'Processing {river}. Its number {l+1} out of {len(rivlist)}')
        process_river(river, src, img, save, resolution)

# # Ensure the required imports
# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import rasterio

# Call the main function if necessary
# maskstack(img=True, save=True)  # Example call
