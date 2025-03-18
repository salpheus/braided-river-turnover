#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 21:48:07 2024
make pngs of masks...for gifs bc photoshop hates me
@author: safiya
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import rasterio
import xarray as xr
import numpy.ma as ma
import copy
import matplotlib.colors as mcol

def normalize(array):
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    if array_max - array_min == 0:
        return array  # Avoid division by zero if all elements are the same
    return (array - array_min) / (array_max - array_min)
#%% load base data
riv = 'brahmaputra_pandu'

allimgs = glob.glob(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{riv}/image/1999on/*.tif')
ms = np.load(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack/{riv}_fullstack.npy')[-25:, :, :]

masks = copy.deepcopy(ms).astype('float')
masks[masks==0] = np.nan
#%%

plt.figure(dpi = 300, tight_layout = True)

ax = plt.gca()

for i, image in enumerate(allimgs):
    file = rasterio.open(image)
    r = normalize(file.read(7))
    g = normalize(file.read(5))
    b = normalize(file.read(4)) ## NIR
        
    img = np.dstack((r, g, b))
    
    ax.imshow(img)
    ax.imshow(masks[i, :, :], cmap = 'binary_r')
    ax.axis('off')
   
    plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/river_images/forgifs/satellite_img_only/{riv}_{i}.png', transparent = True, bbox_inches='tight', pad_inches=0)
    ax.clear()
#%%
base_image = glob.glob(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{riv}/image/1999on/*.tif')[0]
nturns = xr.load_dataset(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt_bulkstats/{riv}.nc').numturns.to_numpy()

file = rasterio.open(base_image)
r = normalize(file.read(7))
g = normalize(file.read(5))
b = normalize(file.read(4)) ## NIR
    
image = np.dstack((r, g, b))


#%% if you wnt the timeseries binary masks for the channel uncomment this
ms = np.load(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack/{riv}_fullstack.npy')[-25:, :, :]


corridor = np.load(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/holes_filled_polys/{riv}.npy')
corridor = ma.masked_equal(corridor, False)
masks = copy.deepcopy(ms).astype('float')
masks[masks==0] = np.nan

for mask in range(ms.shape[0]):

    plt.figure(figsize = (10, 10), dpi = 500, tight_layout = True)
    plt.imshow(image)
    plt.imshow(corridor, cmap = 'binary', alpha = .75)
    
    plt.imshow(masks[mask, :, :], cmap = 'binary_r')
    
    plt.axis('off')
    
    plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/river_images/forgifs/{riv}_{mask}.png', transparent = True, bbox_inches='tight', pad_inches=0)
    # plt.close()
    
#%% Make nturns maps
turn_cols =['#000079', '#3f137a', '#681f7b', '#8d2a7b', '#b2357c', '#cb4c84', '#e3638b', '#fb7b92', '#ffa43d', '#ffc635', '#ffe556']
nturns_cmap = mcol.ListedColormap(turn_cols, 25)
nturns_norm = mcol.Normalize(1, 15)

nturn_mask = ma.masked_equal(nturns, 0)

plt.figure(dpi = 500, tight_layout = True)
plt.imshow(image)
mask = plt.imshow(nturn_mask, cmap = nturns_cmap, norm = nturns_norm)  
plt.colorbar(mask, shrink = 0.5, label = 'Number of turnovers per pixel')
plt.axis('off')  

plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/herofigs/{riv}_nturns-sat.png', transparent = True, bbox_inches='tight', pad_inches=0)



#%% loop through and make nutns map
allnturns = glob.glob(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt_bulkstats/*.nc')[-2:]

img_files = [
 # 'amudaryadown',
 # 'amudaryanew',
 # 'betsiboka',
 # 'bhareli_wide',
 # 'brahmaputra_pandu_allyr',
 # 'colville',
 # 'congo_lukolela_bolobo',
 # 'indus_r2',
 # 'irrawaddy',
 # 'irrawaddy_up',
 # 'kasai',
 # 'lena',
 # 'mangoky',
 # 'ob_down',
 # 'ob_up',
 # 'rakaia',
 # 'southsask',
 # 'tanana',
 'yukon',
 'yukon_eagle']

plt.figure(dpi = 300, tight_layout = True)

ax = plt.gca()

for base, nt in enumerate(allnturns): 
    turns = xr.load_dataset(nt).numturns.to_numpy()
    
    file = rasterio.open(
        glob.glob(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{img_files[base]}/image/1999on/*.tif')[-1])

    r = normalize(file.read(7))
    g = normalize(file.read(5))
    b = normalize(file.read(4)) ## NIR
    
    image = np.dstack((r, g, b))
    
    ax.imshow(image)
    ax.axis('off')

    nturn_mask = ma.masked_equal(turns, 0)

    mask = plt.imshow(nturn_mask, cmap = nturns_cmap, norm = nturns_norm)  
    
    # plt.colorbar(mask, shrink = 0.5, label = 'Number of turnovers per pixel')
    

    plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/herofigs/nturns_saturate/{img_files[base]}_nturns-sat.png', transparent = True, bbox_inches='tight', pad_inches=0)

    ax.clear()
    
#%%

plt.figure(dpi = 300, tight_layout = True)
ax = plt.gca()

# turns = xr.load_dataset(glob.glob(f'/Volumes/SAF_Data/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt_bulkstats/*.nc')[0]).numturns.to_numpy()

ax.axis('off')

nturn_mask = ma.masked_equal(turns, 0)

mask = plt.imshow(nturn_mask, cmap = nturns_cmap, norm = nturns_norm)  
ax.set_xlim(1000, 9000)
# plt.colorbar(mask, shrink = 0.5, label = 'Number of turnovers per pixel')


plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/herofigs/nturns_saturate/agubh2_nturns-sat.png', transparent = True, bbox_inches='tight', pad_inches=0)

ax.clear()


