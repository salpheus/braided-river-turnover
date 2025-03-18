#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:03:10 2024
Save key satellite photos for worldmap figure
@author: safiya
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

#%% define folders and paths
root_folder = '/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/C02_1987-2023_may/'
def pull_years(main_folder, year):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(main_folder):
        for filename in filenames:
            if str(year) in filename and 'image' in filename:
                file_paths.append(os.path.join(dirpath, filename))
    
    return file_paths

def normalize(array):
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    if array_max - array_min == 0:
        return array  # Avoid division by zero if all elements are the same
    return (array - array_min) / (array_max - array_min)
#%%
year = 2023
image_list = pull_years(root_folder, year)

for img in image_list:
    name = img.split('/')[-1].split('.')[0]
    print(name)
    file = rasterio.open(img)
    # r = normalize(file.read(4))
    # g = normalize(file.read(3))
    # b = normalize(file.read(2))
   
    ## false colour
    r = normalize(file.read(7))
    g = normalize(file.read(5))
    b = normalize(file.read(4)) ## NIR
    
    image = np.dstack((r, g, b))
    
    plt.figure(dpi = 500)
    plt.imshow(image)
    plt.savefig(f'/Volumes/SAF_Data/SAF_Data/remote-data/manuscript_figs/river_images/754fc_{name}.svg', transparent = True, bbox_inches='tight', pad_inches=0)
    # plt.close()
