#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:40:00 2024

@author: safiya
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.ma as ma

root = '/Volumes/SAF_Data/remote-data/arrays/C02_SR_arrays/pristine/maskstack'

amguema = np.load(os.path.join(root, 'amguema_pstack.npy'))
tanana = np.load(os.path.join(root, 'tanana_pstack.npy'))
yukon = np.load(os.path.join(root, 'yukon_pstack.npy'))

amguema_s = np.load(os.path.join(root, 'amguema_summer_pstack.npy'))
tanana_s = np.load(os.path.join(root, 'tanana_summer_pstack.npy'))
yukon_s = np.load(os.path.join(root, 'yukon_summer_pstack.npy'))

rivers = 3

years = np.arange(2013, 2022)

fig, ax = plt.subplots(3, 3, figsize = (10, 10), dpi = 500)

a = ax.ravel()
for i in range(len(a)):
    a[i].imshow(ma.masked_equal(amguema[i, :, :], 0), cmap = 'bwr', label = 'full')
    a[i].imshow(ma.masked_equal(amguema_s[i, :, :], 0), cmap = 'bwr_r', label = 'summer')
    a[i].set_title(f'Amg {years[i]} b=full')
    
fig, ax = plt.subplots(3, 3, figsize = (10, 10), dpi = 500)

a = ax.ravel()
for i in range(len(a)):
    a[i].imshow(ma.masked_equal(tanana[i, :, :], 0), cmap = 'bwr')
    a[i].imshow(ma.masked_equal(tanana_s[i, :, :], 0), cmap = 'bwr_r')
    a[i].set_title(f'Tan {years[i]} b=full')
    
fig, ax = plt.subplots(3, 3, figsize = (10, 10), dpi = 500)

a = ax.ravel()
for i in range(len(a)):
    a[i].imshow(ma.masked_equal(yukon[i, :, :], 0), cmap = 'bwr')
    a[i].imshow(ma.masked_equal(yukon_s[i, :, :], 0), cmap = 'bwr_r')
    a[i].set_title(f'Yuk {years[i]} b=full')