#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:26:18 2024

@author: safiya
"""
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy

#%% 

'''so basically im thinking we make a bad mask sandwich and fill in spaces that were water before and after the stripe year with water
...for doube bad years well just have to also do the same and figure out how to make it work'''

congo = np.load('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack/congo_destriped_fullstack.npy')
years = np.arange(1987, 2024)
realyrs = np.where(np.isin(years, np.arange(1999, 2024)))[0]
for yr in realyrs:
    plt.figure(figsize = (10, 10), dpi = 400)
    plt.imshow(congo[yr, :, :], cmap = 'binary')
    plt.title(years[yr])

congo_fix = copy.deepcopy(congo)
# badyrs = [2011]
# badyrs = [1999]
goodyear = 2020
good_yr_pos = np.where(np.isin(years, goodyear))[0]
badyrs = [2004, 2005, 2007, 2008, 2011]
bad_yr_pos = np.where(np.isin(years, badyrs))[0]

max_loss = pd.DataFrame(columns = badyrs, index = ['tot img change', 'wetted area change'])
riv = 'congo'

#%% 
plt.figure(figsize = (10, 10), dpi = 400)
plt.imshow(congo[26, :, :])
#%%
synslice = 12
synyr = 1999

pxdim = len(congo_fix[synslice, :, :].ravel()) ## find total numbr of px
wetpx = np.sum(congo_fix[synslice, :, :]) ## find number of wet px in total image

congopre = congo[23, :, :] #2003(16), 2006(19), 2010(23)
congosyn = congo[synslice, :, :] #2007(20), 2004(17), 2005(18), 2008(21), 2011(24)
congopost = congo[13, :, :] #2006(19), 2009(22), 2012(25)

# findstripes = congosyn + congopost ## add images to find points where value = 2
findstripes = congopre + congosyn + congopost ## add images to find points where value = 2
'''by adding the masks together opptions for pixel stack, only considering where value is 2 bc thats 
    a potential no data spot (striped pixel). Options are:
        [1, 0, 1] = stripe, or small bar that moved in 1 year
        [0, 1, 1] = place that became wet
        [1, 1, 0] = place that went dry
        
        then we will survey these pixels only and check for the [1, 0, 1] ordering'''
        
# np.save('/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/destriping/congo1999_change2.npy', findstripes)

#%%

# cmap = plt.cm.PiYG
# cmaplist = [cmap(i)for]
plt.figure(figsize = (12, 12), dpi = 400)
stripes = plt.imshow(findstripes, cmap = 'hsv', vmin = 0, vmax = 3)    
plt.colorbar(stripes)

#%% reset stripes to regular water pixels

search_locs = np.where(findstripes==1)
changed_px = np.zeros_like(congosyn)

for i, j in zip(search_locs[0], search_locs[1]):
    # print(i, j)
    if congopre[i, j]==1 and congosyn[i, j]==0 and congopost[i, j]==1:
        changed_px[i, j] = 1
## save array with 1 locating where pixel is surrounded by wet years before and after
np.save(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/destriping/{riv}{synyr}_changedpx.npy', changed_px)

##plot to see where changes are happening
plt.figure(figsize = (10, 10), dpi = 150)
plt.imshow(changed_px, cmap = 'Spectral', vmin = 0, vmax = 5)
plt.colorbar()    

new_slice = copy.deepcopy(congosyn)
new_slice[changed_px==1] = 1 ## np.nan # depending on what is better 

plt.figure(figsize = (12, 12), dpi = 400)
plt.imshow(new_slice, cmap = 'seismic', vmin = 0, vmax = 1)    
plt.title('destriped year')
plt.colorbar(stripes)

congo_fix[synslice, :, :] = new_slice

plt.figure(figsize = (10, 10), dpi = 150)
plt.imshow(congo_fix[synslice, :, :])


## update loss dataframe
max_loss.at['tot img change', synyr] = np.sum(changed_px)/pxdim
max_loss.at['wetted area change', synyr] = np.sum(changed_px)/wetpx

#%% save wetted pixel change csv for uncertainty/eroor analysis

# max_loss.to_csv(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/destriping/{riv}_destripe_err.csv')
np.save(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/destriping/{riv}_destriped.npy', congo_fix)
# 

## final check to make sure everrything ran ok. Plotting a sum image, max val shoule b 5
# summask = congo_fix[17, :, :] + congo_fix[18, :, :] + congo_fix[20, :, :] + congo_fix[21, :, :] + congo_fix[24, :, :]
summask = congo_fix[24, :, :] + congo[24, :, :]
plt.figure(figsize = (12, 12), dpi = 400)
plt.imshow(summask, cmap = 'Spectral', vmin = 0, vmax = 2)    
plt.title('All destriped years')
plt.colorbar()
#%% rewrite the geotiffs

yr = 1999
mslice = 12
path_to_tiff = ('/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/congo_lukolela_bolobo/mask/1999on')
fname = f'congo_lukolela_bolobo_{yr}_01_01_{yr}_12_31_mask.tif'
outname = f'congo_lukolela_bolobo_{yr}_01_01_{yr}_12_31_mask_ds.tif'

orig_mask = gdal.Open(os.path.join(path_to_tiff, fname))
# arr = orig_mask.ReadAsArray()

# plt.imshow(arr)    

def write_geotiff(filename, arr, in_ds):
    arr_type = gdal.GDT_Byte

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type) ## 1 = number of bands
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    
    ## create a new band and set the band data with the array data
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr) ##should maybe be calledcongo_fix, but this is a one time code
    band.FlushCache()
    band.ComputeStatistics(False)

fixed_mask = congo_fix[mslice, :, :]
    
write_geotiff(os.path.join(path_to_tiff, outname), fixed_mask, orig_mask)

fix_check = gdal.Open(os.path.join(path_to_tiff, outname))
img = fix_check.ReadAsArray()
plt.figure(figsize = (10, 10), dpi = 400)
plt.imshow(img)
plt.title(yr);