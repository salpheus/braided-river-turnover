#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:18:38 2024
stack agubh2
@author: safiya
"""

import numpy as np
import rasterio
import pandas as pd
import os 
import glob
import matplotlib.pyplot as plt
import natsort

#%%

fpath = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/agubh2/mask'

masks = glob.glob(os.path.join(fpath, '*tif'))

template = rasterio.open(masks[0])
lat = template.height
long = template.width

maskstack = np.empty([len(masks), lat, long], dtype=int)
summary = pd.DataFrame(columns=['year', 'wet px', 'wet area'])

compilation, ax = plt.subplots(5, 5, sharex = True, sharey = True, dpi = 400, tight_layout = True, figsize = (16, 8))
ax = ax.ravel()
river = 'agubh2'
for t, mpath in enumerate(masks):
    file = os.path.basename(mpath)
    year = file.split("_")[-1].split('.')[0]
    print(year)
    im = rasterio.open(mpath).read()
    maskstack[t, :, :] = im
    
    nwet = np.sum(maskstack[t, :, :])
    area_wet = nwet/1e6 ## area in km, these cells are interpolated already
    
    df = pd.DataFrame([[year, nwet, area_wet]], columns=summary.columns)
    summary = pd.concat([summary, df], axis=0, ignore_index=True)
    
    # ax[t].imshow(np.reshape(im, (im.shape[1], im.shape[2])), cmap='gray')
    # ax[t].set_title(f'agubh2 {year}')
    
    diffs = np.diff(maskstack, axis=0)
    
    summary.set_index('year')
    summary.to_csv('/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/C02_1987-2023_allLS_db_csv/agubh2_sum_summary_skip.csv')

    saveroot = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/'
    compilation.savefig(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_compilation/C02_1987-2023_allLS_db_poster/{river}_compilation_sum_stackmasks_skip.png', dpi=400)
    
    np.save(os.path.join(saveroot, 'full/maskstack/', f'{river}_fullstack.npy'), maskstack, allow_pickle=True)
    np.save(os.path.join(saveroot, 'full/diffs/', f'{river}_diffs_fullstack.npy'), diffs, allow_pickle=True)
    
#%%

from puller import get_paths
import os
from mobility import get_mobility_rivers
from gif import get_stats
from gif import get_mobility


#%%

base = f'/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may/agubh2/'
fps = glob.glob(os.path.join(base, 'mask_crs', '*.tif'))
out_paths = {base: fps}
poly = f"/Volumes/SAF_Data/remote-data/watermasks/gpkgs_fnl/agubh2.gpkg"

mobility_csvs = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/mobcsvs'
out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999' # blocked fit_stats outputs
mobility_out = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/mobility'
figout = '/Volumes/SAF_Data/remote-data/greenberg-mobility-outputs/C02_1987-2023_may_1999/figs'

get_mobility_rivers(poly, out_paths, 'agubh2')  

''' if you get error: CPLE_AppDefinedError: Too many points (435 out of 441) failed to transform, unable to compute output bounds.
    while running get_mobility_rivers change the scale to 1 (line 28 mobility.py)'''

# compille csvs and compute curves
stop = 30
allcsvs = glob.glob(os.path.join(mobility_csvs, '*.csv')) 
path_list = natsort.natsorted(allcsvs)
river = 'agubh2'    
    
get_stats(os.path.join(mobility_csvs, f'{river}_yearly_mobility.csv'), river, out, 30)
get_mobility(pd.read_csv(os.path.join(out, f'{river}_fit_stats.csv'), header = 0), os.path.join(mobility_out, f'{river}_mobility.csv'))
