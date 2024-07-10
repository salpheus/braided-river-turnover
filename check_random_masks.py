#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:34:21 2024

@author: safiya
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from PIL import Image

folname = 'C02SR_Batch_WL2_Mar25_masks_clipped'
root = '/Volumes/SAF_Data/remote-data/watermasks'
river = 'irrawaddy'
path = os.path.join(root, folname, river, 'mask')

tif_files = glob.glob(os.path.join(path, '*.tif'))

# get a random sample from the list of tif_files
rand_mask = random.choice(tif_files)
mask = Image.open(rand_mask)

plt.figure(figsize = (10, 10), dpi = 150, tight_layout = True) 
plt.imshow(mask, cmap = 'gray')
plt.title(rand_mask)