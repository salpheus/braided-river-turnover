#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:24:29 2024

@author: safiya
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import pandas as pd
# from scipy.ndimage.morphology import binary_erosion


root = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
rivlist = os.listdir(root)
rivlist.remove('.DS_Store')
rivlist.remove('brahmaputra_yangcun')
#%% get maximum wettable area boundary polygon using hole filling algorithm in rivgrapm
'''using im_utils largest_blobs and fill_holes code from rivgraph to make a binary image of the maximum wettable
   channel area during the observation period. Only need to run this code once to generate the polygon arrays.
   Run using the rivgraph conda environment and yo'll probaly need to run the im_utils.py script if you're running
   from a different working directory. Last run on Aug 7th. 
# '''
from im_utils import largest_blobs, fill_holes

ms_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack'
# riv = 'agubh2'
# for riv in rivlist[11:]:
#     print(riv)
#     if riv == 'congo_lukolela_bolobo':
#         riv = 'congo_destriped'
#     elif riv == 'kasai':
#         riv = 'kasai_destriped'
    # compos = np.sum(np.load(os.path.join(ms_path, f'{riv}_fullstack.npy'))[-25:, :, :], axis = 0) ##load and sum only the part of full stack that starts at 1999
    # compos[compos >= 1] = 1
    # I = largest_blobs(compos, nlargest = 1, action = 'keep')    
    # Ihf = fill_holes(I)
    
    # ## save binary numpy array of the holes filled polygon
    # np.save(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/holes_filled_polys/{riv}.npy', Ihf)

    # plt.figure(dpi = 300)
    # plt.imshow(Ihf, cmap = 'gray')
    # plt.title(riv)
    # plt.savefig(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/holes_filled_polys/{riv}_holesfilled.png')
    # plt.close()
    

#%% create masks of bars only using hole filler

for riv in rivlist:
    print(riv)
    if riv == 'congo_lukolela_bolobo':
        riv = 'congo_destriped'
    elif riv == 'kasai':
        riv = 'kasai_destriped'
    
    if riv =='agubh2':
        masks = np.load(os.path.join(ms_path, f'{riv}_fullstack.npy')) ##load masks
    else:
        masks = np.load(os.path.join(ms_path, f'{riv}_fullstack.npy'))[-25:, :, :] ##load only the part of full stack that starts at 1999
    bar_maps = np.zeros_like(masks)
    ## make bar masks
    for yr in range(masks.shape[0]): ## iterate through each mask
        I = largest_blobs(masks[yr, :, :], nlargest = 1, action = 'keep')    # find largest blobs
        Ihf = fill_holes(I) # fill holes
        
        # subtract polygons to get the bars only (0-0 = 0 (fp stays 0), 1-1 = 0 (channel = 0), 1-0 = 1, land in channel -> 1)
        bar_maps[yr, :, :] = Ihf-masks[yr, :, :]
        
        fig, ax = plt.subplots(2, dpi = 300)
        ax[0].imshow(bar_maps[yr, :, :], cmap = 'gray')
        ax[0].imshow(Ihf, cmap = 'Reds', alpha = 0.25)
        
        bar_maps[yr, :, :][bar_maps[yr, :, :]<1] = 0 ## for agubh2 (s0 far) small channels on the boundary that are disconnected don't get included in the holes filled poly, I am leaving them out tentatively
        ax[1].imshow(bar_maps[yr, :, :], cmap = 'gray')
        ax[1].imshow(Ihf, cmap = 'Reds', alpha = 0.25)
        
        fig.suptitle(f'{riv}, {1999+yr}')
        plt.savefig(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/bar_masks_imgs/{riv}_{yr+1999}_barmask.png')
        plt.close()
        
    ## save binary numpy array of the holes filled polygon
    np.save(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/bar_masks/{riv}_barmask.npy', bar_maps)

    
#%% actually calculate ndvi, may have to ru Evan's watermask_methods.py

## check to see if ndvi calculated over masked channels or complete image
import xarray as xr
import rasterio
rivlist.remove('agubh2')

def Ndvi(im):
    return (
        (im[4,:,:] - im[3,:,:])
        / (im[4,:,:] + im[3,:,:])
    )
# ndvi_df = pd.DataFrame(index = rivlist)


for riv in rivlist:    
    print(riv)
    img_paths = glob.glob(os.path.join(root, riv, 'image', '1999on', '*.tif')) 
    #mask_paths = glob.glob(os.path.join(root, riv, 'mask', '1999on', '*.tif')) 
    max_boundary = np.load(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/holes_filled_polys/{riv}.npy')
    lat_crop, long_crop = np.where(max_boundary == 0)
    
    ndvi_composite = np.zeros([25, max_boundary.shape[0], max_boundary.shape[1]])
    for yr, img in enumerate(img_paths):
        ds = rasterio.open(img)    
        im = ds.read()
        
        # mask = np.squeeze(rasterio.open(mask_paths[yr]).read(), axis = 0) ## load the mask, remove channel data
        
        # cropped_im = copy.deepcopy(im)
        # cropped_im[:, lat_crop, long_crop] = np.nan ## crops to active belt
        
        # ch_lat, ch_long = np.where(mask == 1)
        # cropped_im[:, ch_lat, ch_long] = np.nan ## remove parts of the domain that are water so that you don't calculate ndvi there

        # ndvi_composite[yr, :, :] = Ndvi(cropped_im) ## calculate NDVI only within the active channel belt
        ndvi_composite[yr, :, :] = Ndvi(im) ## calculate NDVI only within the active channel belt
 
        # plt.imshow(max_boundary)
        # plt.imshow(mask, cmap = 'grey')
        # plt.title(f'{riv}, {yr}, ndvi & mask')
        # plt.imshow(ndvi_composite[yr, :, :])
        # plt.savefig(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/ndvi_maps_qc/{riv}_{yr+1999}.png', dpi = 300)
        # plt.close()
        
    np.save(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/ndvi_stacks_withwater/{riv}.npy', ndvi_composite)
    
    
    # ndvi_df.loc[riv, 'min'] = np.nanmin(ndvi_composite)
    # ndvi_df.loc[riv, 'mean'] = np.nanmean(ndvi_composite)
    # ndvi_df.loc[riv, 'med'] = np.nanmedian(ndvi_composite)
    # ndvi_df.loc[riv, 'max'] = np.nanmax(ndvi_composite)
    # ndvi_df.loc[riv, 'q1'] = np.nanquantile(ndvi_composite, 0.25)
    # ndvi_df.loc[riv, 'q3'] = np.nanquantile(ndvi_composite, 0.75)
    
# ndvi_df.to_csv('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/ndvi_composite_stats.csv')
    
