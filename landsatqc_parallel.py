#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:07:24 2024

@author: safiya
"""
import os
import concurrent.futures
import fiona
import pandas as pd
import numpy as np
import ee
from shapely.geometry import Polygon
# from ee_datasets import maskL8sr

# Initialize Earth Engine API (Make sure you're authenticated)
ee.Initialize()

def maskL8sr(image):
    """
    Masks out clouds within the images
    """
    # Bits 3 and 5 are cloud shadow and cloud
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get pixel QA band
    qa = image.select('BQA')
    # Botos.path.joinh flags should be zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0)
    )
    return image.updateMask(mask)

def get_image_info(image):
    return ee.Feature(None, {
        'date': image.date().format('YYYY-MM-dd'),
        'collection': image.get('collection'),
        'path': image.get('WRS_PATH'),
        'row': image.get('WRS_ROW'),
        'img_id': image.id()
,
  })
# Function to process each river
def process_river(river, arctic):
    print(f'Processing: {river}')
    
    image_info_list = []
    riv_collection = pd.DataFrame(columns=['date', 'collection', 'path', 'row', 'img_id', 'totalpx', 'unmaskedpx'])
    polygon_path = f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/gpkgs_fnl/{river}.gpkg'
    
    # polygon_name = polygon_path.split('/')[-1].split('.')[0]
    
    with fiona.open(polygon_path, layer=river) as layer:
        for feature in layer:
            geom = feature['geometry']
            poly = ee.Geometry.Polygon(geom['coordinates'])

    for year in range(1999, 2024):
        print(f'Calculating for {river}: {year}')
        
        if river in arctic:
            start, end = f'{year}-05-01', f'{year}-09-30'
        else:
            start, end = f'{year}-01-01', f'{year}-12-31'
        
        allLS_aoi = allLS_mask.filterBounds(poly).filterDate(start, end)
        allLS_list = allLS_aoi.toList(allLS_aoi.size())
        
        features = allLS_aoi.map(get_image_info).distinct(['date', 'collection', 'path', 'row', 'img_id'])
        dates = features.aggregate_array('date').getInfo()
        collections = features.aggregate_array('collection').getInfo()
        paths = features.aggregate_array('path').getInfo()
        rows = features.aggregate_array('row').getInfo()
        ids = features.aggregate_array('img_id').getInfo()

        # Create a DataFrame for metadata
        feat_data = pd.DataFrame([
            {'date': date, 'collection': collection, 'path': path, 'row': row, 'img_id': img_id} 
            for date, collection, path, row, img_id in zip(dates, collections, paths, rows, ids)
        ])

        pixel_mask_data = pd.DataFrame(columns=['totalpx', 'unmaskedpx', 'img_id'])

        print(f'Calculating pixels for {river}, num. images = {allLS_list.size().getInfo()}')
        for i in range(allLS_list.size().getInfo()):
            image = ee.Image(allLS_list.get(i))
            
            total_pixels = image.select(0).unmask().reduceRegion(
                reducer=ee.Reducer.count(), geometry=poly, scale=30, maxPixels=1e9
            ).get('uBlue').getInfo()
            
            unmasked_pixels = image.select(0).reduceRegion(
                reducer=ee.Reducer.count(), geometry=poly, scale=30, maxPixels=1e9
            ).get('uBlue').getInfo()

            px_df = pd.DataFrame({
                'totalpx': total_pixels, 'unmaskedpx': unmasked_pixels, 'img_id': image.id().getInfo()
            }, index=[0])

            pixel_mask_data = pd.concat((pixel_mask_data, px_df), axis=0)

        riv_data = pd.merge(feat_data, pixel_mask_data, on='img_id')
        riv_collection = pd.concat((riv_collection, riv_data), axis=0)

    # Handle NaN for totalpx and calculate percentage of unmasked pixels
    riv_collection['totalpx'].replace(0, np.nan, inplace=True)
    riv_collection['perc_unmasked'] = riv_collection['unmaskedpx'] / riv_collection['totalpx']

    # Write to CSV
    riv_collection.to_csv(f'/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/imagery_dates/{river}_master_maskqc.csv')
    print(f'Image information for {river} has been written to CSV.')

# List of rivers and arctic regions
rivlist = ['amudaryanew', 'betsiboka', 'bhareli_wide', 'brahmaputra_pandu',
           'colville', 'congo_lukolela_bolobo', 'indus_r2', 'irrawaddy', 'irrawaddy_up',
           'kasai', 'lena', 'mangoky', 'ob_down', 'ob_up', 'rakaia', 'southsask',
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


# Parallelize the processing of rivers
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_river = {executor.submit(process_river, river, arctic): river for river in rivlist}
    for future in concurrent.futures.as_completed(future_to_river):
        river = future_to_river[future]
        try:
            future.result()
        except Exception as exc:
            print(f'{river} generated an exception: {exc}')
