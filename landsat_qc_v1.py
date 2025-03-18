#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:10:41 2024
Trying to pull the landscat dates for images used
@author: safiya
"""
import os 
from shapely.geometry import Polygon
import csv
import fiona
import ee
import pandas as pd
import numpy as np
import glob
from ee_datasets import maskL8sr, getLandsatCollection

#%%
    
# Function to calculate the number of pixels and masked pixels ~~ from Copilot
def calculate_pixels(image):
    total_pixels = image.select(0).unmask().reduceRegion( ##unmask the masked stuff and calculate totals
        reducer=ee.Reducer.count(),
        geometry=poly,
        scale=30,
        maxPixels=1e9
    ).get('uBlue')

    unmasked_pixels = image.select(0).reduceRegion( ## calculate masked totals
        reducer=ee.Reducer.count(),
        geometry=poly,
        scale=30,
        maxPixels=1e9
    ).get('uBlue')
    print(total_pixels, unmasked_pixels)
    # masked_pixels = total_pixels.subtract(unmasked_pixels)

    return image.set({
        'total_pixels': total_pixels,
        'unmasked_pixels': unmasked_pixels
    })


## process image to calculate the pixels and other metadata
def process_image(image):
    total_pixels, masked_pixels = calculate_pixels(image)
    date = image.date().format('YYYY-MM-dd').getInfo()
    collection = image.get('collection').getInfo()
    path = image.get('WRS_PATH').getInfo()
    row = image.get('WRS_ROW').getInfo()
    
    image_info_list.append({
        "date": date,
        "collection": collection,
        "path": path,
        "row": row,
        "total_pixels": total_pixels,
        "masked_pixels": masked_pixels
    })

# # Get the list of dates and collection information
def get_image_info(image):
    return ee.Feature(None, {
        'date': image.date().format('YYYY-MM-dd'),
        'collection': image.get('collection'),
        'path': image.get('WRS_PATH'),
        'row': image.get('WRS_ROW'),
        'img_id': image.id()
,
  })

#%%% organize the polygons
rivlist = ['amudaryadown','amudaryanew','betsiboka','bhareli_wide', 'brahmaputra_pandu',
           'colville', 'congo_lukolela_bolobo','indus_r2','irrawaddy','irrawaddy_up',
           'kasai','lena', 'mangoky','ob_down', 'ob_up', 'rakaia', 'southsask',
           'tanana', 'yukon', 'yukon_eagle']
arctic = ['colville', 'lena', 'ob_down', 'ob_up', 'tanana', 'yukon', 'yukon_eagle']


bn9 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL',]
bn8 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL',]
bn7 = ['SR_B1', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL',]
bn5 = ['SR_B1', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL',]
bns = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']

# Merge Landsat collections with a property indicating the collection
ls5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(bn5, bns).map(lambda img: img.set('collection', 'Landsat 5'))
ls7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterDate('1999-01-01', '2012-12-31').select(bn7, bns).map(lambda img: img.set('collection', 'Landsat 7'))
ls8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').select(bn8, bns).map(lambda img: img.set('collection', 'Landsat 8'))
ls9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').select(bn9, bns).map(lambda img: img.set('collection', 'Landsat 9'))

allLandsat = ls5.merge(ls7).merge(ls8).merge(ls9)
allLS_mask = allLandsat.map(maskL8sr)  # Map the bitmask onto the images

for river in rivlist:
    print('Processing: ', river)
    image_info_list = []
    riv_collection = pd.DataFrame(columns=['date', 'collection', 'path', 'row', 'img_id', 'totalpx', 'unmaskedpx'])
    polygon_path = f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/gpkgs_fnl/{river}.gpkg'

    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']
            poly_shape = Polygon(geom['coordinates'][0])
            poly = ee.Geometry.Polygon(geom['coordinates'])
    for year in range(1999, 2024):
        print(f'calculating: {year}')
        if river in arctic:
           start = f'{year}-05-01'
           end = f'{year}-09-30'
        else: 
            start = f'{year}-01-01'
            end = f'{year}-12-31'
    
        allLS_aoi = allLS_mask.filterBounds(poly).filterDate(start, end)
        
        allLS_list = allLS_aoi.toList(allLS_aoi.size())
        
        features = allLS_aoi.map(get_image_info).distinct(['date', 'collection', 'path', 'row', 'img_id'])
        dates = features.aggregate_array('date').getInfo()
        collections = features.aggregate_array('collection').getInfo()
        paths = features.aggregate_array('path').getInfo()
        rows = features.aggregate_array('row').getInfo()
        ids = features.aggregate_array('img_id').getInfo()
        
        #create df to store the aggregate data,i think this is faster than loop wise
        feat_data = pd.DataFrame([{'date': date, 'collection': collection, 'path': path, 'row': row, 'img_id': img_id} 
                                  for date, collection, path, row, img_id in zip(dates, collections, paths, rows, ids)])
        
       
        pixel_mask_data = pd.DataFrame(columns = ['totalpx', 'unmaskedpx', 'img_id']) ## stores all data for all images

        print(f'calculating pixels for {river}, num. images = {allLS_list.size().getInfo()}')
        for i in range(allLS_list.size().getInfo()):
            image = ee.Image(allLS_list.get(i))#.clip(poly)
            total_pixels = image.select(0).unmask().reduceRegion(  ## unmask the cloud cover bits and calculate the total pixels in the polygon
                reducer=ee.Reducer.count(),
                geometry=poly,
                scale=30,
                maxPixels=1e9
            ).get('uBlue').getInfo()

            unmasked_pixels = image.select(0).reduceRegion(  ### find the unmasked pixels, ee.reducer reduces the pixel area to the polygon specified
                reducer=ee.Reducer.count(),
                geometry=poly,
                scale=30,
                maxPixels=1e9
            ).get('uBlue').getInfo()
            
            px_df = pd.DataFrame({'totalpx': total_pixels, 'unmaskedpx': unmasked_pixels, 'img_id': image.id().getInfo()}, index = [0])
            pixel_mask_data = pd.concat((pixel_mask_data, px_df), axis = 0)

        riv_data = pd.merge(feat_data, pixel_mask_data, on = 'img_id') ## all data for the images for ONE year

        # Concatenate the DataFrame to the river collection DataFrame
        riv_collection = pd.concat((riv_collection, riv_data), axis=0) ## adds to mater collection file for all data in all years

    riv_collection['totalpx'][riv_collection['totalpx']==0] = np.nan ### replace any regions where total px = 0
    riv_collection['perc_unmasked'] = riv_collection['unmaskedpx'] / riv_collection['totalpx']

     # Write the DataFrame to a CSV file
    riv_collection.to_csv(os.path.join('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates', f'{river}_master_maskqc.csv')) 

    print(f"Image information with pixel counts has been written to for {river}.csv")















#%% 

csvs = glob.glob('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/*.csv')
for csv in csvs:
    name = csv.split('/')[-1].split('_LS_dates_collections.csv')[0]
    # Read the CSV file
    df = pd.read_csv(csv)
    
    # Extract the year and month from the 'Date' column
    df['Year'] = pd.to_datetime(df['date']).dt.year
    df['Month'] = pd.to_datetime(df['date']).dt.month
    
    # Create a pivot table with years as rows and months as columns
    pivot_table = df.pivot_table(index='Year', columns='Month', aggfunc='size', fill_value=0)
    
    # Rename the columns to month names
    if name in arctic:
        pivot_table.columns = ['May', 'Jun', 'Jul', 'Aug', 'Sep']
    else:
        pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(pivot_table)
    # Optionally, save the pivot table to a new CSV file
    pivot_table.to_csv(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/yearly_aggregate/{name}.csv')
    
    print(f"Pivot table {name} has been written to landsat_images_per_month.csv")
    
    
#%% new code to find the amount of masked pixels in a region
 