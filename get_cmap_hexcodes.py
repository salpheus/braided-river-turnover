#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:31:11 2024

@author: safiya
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import numpy as np

static_csv = pd.read_excel('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx')
kg_rgb = pd.read_csv('/Volumes/SAF_Data/remote-data/watermasks/admin/kgcols.csv')
print(static_csv.columns)

static_merge = pd.merge(static_csv, kg_rgb, left_on = 'kg_clim', right_on = 'kgclim', how = 'inner').drop(columns = 'kgclim')

hex_codes = pd.DataFrame(columns = static_merge.columns)
## create colormap, otherwise use a standard ome
swatches =['#ffffe4', '#fff4b8', '#ffe98b', '#ffde5a', '#ffd126', '#ffc31d', '#ffb416', '#ffa50f', '#ff9608', '#ff8500']

# cmap = plt.cm.viridis ## 
cmap = mcolors.ListedColormap(swatches)

continuous_cols = static_csv.columns.to_list()[4:]

## notmalise the atrribute values to [0, 1]
for attr in continuous_cols:
#attr = 'bedload-ssc_qw'
    attr_normalised = static_csv[attr].to_numpy()
    attr_normalised = (attr_normalised-attr_normalised.min())/(attr_normalised.max()-attr_normalised.min())
    norm = mcolors.Normalize(vmin=static_csv[attr].min(), vmax=static_csv[attr].max())
    print(static_csv[attr].min(), static_csv[attr].max())
    
    ## apply the colormap to the normalised values and convert them to hex values
    hex_codes[attr] = static_csv[attr].apply(lambda x:mcolors.to_hex(cmap(norm(x))))

for attr in ['river', 'kg_clim', 'description', 'rgb']:
    hex_codes[attr] = static_merge[attr]
    
hex_codes.to_csv('/Volumes/SAF_Data/remote-data/watermasks/admin/inventory-hex_codes.csv')
# view cbar
fig, ax = plt.subplots(figsize=(6, 1), dpi = 300)
fig.subplots_adjust(bottom=0.5)

# Create a scalar mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.axis('off')

# Add colorbar to the plot
cbar = plt.colorbar(sm, orientation='horizontal', ax=ax, aspect = 50)
cbar.set_label('(Qs_bed +Qs_susp / Mean Qw)')
