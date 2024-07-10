#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:07:17 2024

@author: safiya

get banklines from images
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#%%

rivname = 'rakaia'
rivpath = f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/full/maskstack/{rivname}_fullstack.npy'

riv = np.load(rivpath)

allyears = np.arange(1987, 2024)
calc_years = np.where(np.isin(allyears, np.arange(1999, 2024)))[0]
#%%
yr = 36
img = riv[yr, :, :].astype('uint8')

# Find contours
contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
empty_img = np.zeros_like(img)

cv.drawContours(empty_img, contours, -1, (255), 1)
plt.figure(figsize=(16, 6), dpi = 300)
plt.imshow(empty_img, cmap='gray')
plt.title('Outermost Contour')
plt.axis('off')
plt.show()

#%%

plt.figure(figsize = (10, 12), dpi = 400)

plt.imshow(img, cmap = 'gray')