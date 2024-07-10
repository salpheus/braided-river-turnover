#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:12:34 2024

@author: safiya
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import Affine
import os
import glob
import pandas as pd

#%% make a df to store the crses of each maskstach

crs = pd.DataFrame(columns = ['river', 'CRS'])

#%%

# rivlist = os.listdir('/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/')
# rivlist = ['lena', 'congo_lukolela_bolobo']
rivlist = ['congo_destriped']
for rivname in rivlist:
    print(f'processing {rivname}')
    rivpath = f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack/{rivname}_fullstack.npy'
    
    # tiffpath = f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/{rivname}/mask/*.tif'
    tiffpath = f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/congo_lukolela_bolobo/mask/*.tif'
    latest_tiff = glob.glob(tiffpath)[-1]
    
    allyears = np.arange(1987, 2024)
    calc_years = np.where(np.isin(allyears, np.arange(1999, 2024)))[0]
    years = np.arange(1999, 2024)
    
    start_yr = calc_years[0] ## we are building the netcdf using data from 1999-2023 only
    riv = np.load(rivpath)[start_yr:, :, :] ## load the array

# Open the GeoTIFF file, use 2023 bc thats the best quality almost always
# get an array of lat and long points
    with rasterio.open(latest_tiff) as dataset:
        # Read the affine transformation matrix

        rivcrs = {'river': rivname, 'CRS': str(dataset.crs)}
        crs.loc[len(crs)] = rivcrs
        transform = dataset.transform
        # Read the dimensions of the raster
        width = dataset.width
        height = dataset.height
        
        # Create arrays to hold the latitude and longitude values
        lon = []
        lat = []
        
        for i in range(height):
            row_lon = []
            row_lat = []
            for j in range(width):
                x, y = transform * (j, i)  # Apply the affine transformation
                row_lon.append(x)
                row_lat.append(y)
            lon.append(row_lon)
            lat.append(row_lat)
        
        lon = np.array(lon)
        lat = np.array(lat)
        
    longitude = lon[0, :] ## just get the singular values
    latitude = lat[:, 0]    
    
    # make a makeshift netcdf
    
    ## create a DataArray, give it dimensions of time, lat and lon
    masks = xr.DataArray(riv, dims = ['year', 'lat', 'lon'], coords={'year': years,
                                                                             'lat': latitude,
                                                                             'lon': longitude})
    ## you can add dataarrays to a datset just like you wpuld in a dataframe, just specify the coords
    rivmasks = xr.Dataset()
    rivmasks['masks'] = masks
    
    rivmasks.to_netcdf(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc/{rivname}_masks.nc')
    
    
#%% save csv of the crses

crs.to_csv(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/mask_CRS.csv')

