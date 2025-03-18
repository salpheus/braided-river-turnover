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

#%% define the get image period thing
# def rescale(image):
#     bns = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']
#     return image.select(bns).multiply(0.0000275).add(-0.2)

# def get_image_period(start, end, polygon, dataset='landsat'):
#     images = ee.List([])
#     if dataset == 'landsat':
#         allLandsat = getLandsatCollection()
#         images = allLandsat.map(
#             maskL8sr
#         ).filterDate(
#             start, end 
#         ).filterBounds(poly)#clip(polygon)) ## i removed median from here
#         images = ee.ImageCollection(images)
#         print('collection size: ', images.size().getInfo())
#         if len(images.first().bandNames().getInfo()):
#             images = images.map(rescale)

#     return ee.ImageCollection(images)
#     # return images.median().clip(polygon)
    
# Function to calculate the number of pixels and masked pixels ~~ from Copilot
def calculate_pixels(image):
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

    masked_pixels = total_pixels - unmasked_pixels

    return total_pixels, masked_pixels

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
# def get_image_info(image):
#     return ee.Feature(None, {
#         'date': image.date().format('YYYY-MM-dd'),
#         'collection': image.get('collection'),
#         'path': image.get('WRS_PATH'),
#         'row': image.get('WRS_ROW'),
#   })

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

# for river in rivlist:
#     riv_collection = pd.DataFrame(columns = ['date', 'collection', 'path', 'row'])
#     polygon_path = f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/gpkgs_fnl/{river}.gpkg'

#     polygon_name = polygon_path.split('/')[-1].split('.')[0]
#     with fiona.open(polygon_path, layer=polygon_name) as layer:
#         for feature in layer:
#             geom = feature['geometry']
#             poly_shape = Polygon(geom['coordinates'][0])
#             poly = ee.Geometry.Polygon(geom['coordinates'])
#     for year in range(1999, 2024):
#         if river in arctic:
#            start = f'{year}-05-01'
#            end = f'{year}-09-30'
#         else: 
#             start = f'{year}-01-01'
#             end = f'{year}-12-31'
    
#         allLS = allLandsat.map(maskL8sr) ## map the bitmask onto the images
#         allLS_aoi= allLS.filterBounds(poly).filterDate(start, end) ## crop to polygon
    
#         features = allLS_aoi.map(get_image_info).distinct(['date', 'collection', 'path', 'row'])

#         dates = features.aggregate_array('date').getInfo()
#         collections = features.aggregate_array('collection').getInfo()
#         paths = features.aggregate_array('path').getInfo()
#         rows = features.aggregate_array('row').getInfo()
        
#         # Combine dates and collections into a list of dictionaries
#         date_collect = pd.DataFrame([{'date': date, 'collection': collection, 'path': path, 'row': row} for date, collection, path, row in zip(dates, collections, paths, rows)])
#         riv_collection = pd.concat((riv_collection, date_collect), axis = 0)
    
#     # Write the dates and collection information to a CSV file
#     riv_collection.to_csv(os.path.join('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates', f'{river}_LS_dates_collections.csv')) 

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

for river in rivlist:
    image_info_list = []
    riv_collection = pd.DataFrame(columns = ['date', 'collection', 'path', 'row'])
    polygon_path = f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/gpkgs_fnl/{river}.gpkg'

    polygon_name = polygon_path.split('/')[-1].split('.')[0]
    with fiona.open(polygon_path, layer=polygon_name) as layer:
        for feature in layer:
            geom = feature['geometry']
            poly_shape = Polygon(geom['coordinates'][0])
            poly = ee.Geometry.Polygon(geom['coordinates'])
    for year in range(1999, 2024):
        if river in arctic:
           start = f'{year}-05-01'
           end = f'{year}-09-30'
        else: 
            start = f'{year}-01-01'
            end = f'{year}-12-31'
    
        allLS_aoi = allLandsat.filterBounds(poly).filterDate(start, end)
        allLS_mask = allLS_aoi.map(maskL8sr)  # Map the bitmask onto the images
        
        # Iterate over all images in the collection and calculate pixels

        allLS_mask.map(process_image) ## map the process image function to the masked landsat dataset (that is cropped to the poly)
                                      ## within this process_image the data is appended to the image data list for the df      
       

        # features = allLS_aoi.map(get_image_info).distinct(['date', 'collection', 'path', 'row'])

        # dates = features.aggregate_array('date').getInfo()
        # collections = features.aggregate_array('collection').getInfo()
        # paths = features.aggregate_array('path').getInfo()
        # rows = features.aggregate_array('row').getInfo()
        
        # # Combine dates and collections into a list of dictionaries
        # date_collect = pd.DataFrame([{'date': date, 'collection': collection, 'path': path, 'row': row} for date, collection, path, row in zip(dates, collections, paths, rows)])
        # riv_collection = pd.concat((riv_collection, date_collect), axis = 0)
    #  # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(image_info_list)

     # Write the DataFrame to a CSV file
    df.to_csv(os.path.join('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates', f'{river}_LS_dates_collections_maskdata.csv')) 
    # Write the dates and collection information to a CSV file
    # riv_collection.to_csv(os.path.join('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates', f'{river}_LS_dates_collections.csv')) 

    print(f"Image information with pixel counts has been written to for {river}.csv")
# # Iterate over all images in the collection and calculate pixels
# image_info_list = []

# allLS_mask.map(process_image)

# # Convert the list of dictionaries to a DataFrame
# df = pd.DataFrame(image_info_list)

# # Write the DataFrame to a CSV file
# df.to_csv('landsat_image_info_with_pixels.csv', index=False)













