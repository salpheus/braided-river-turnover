#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:30:27 2024

@author: safiya
"""
import xarray as xr
import pandas as pd
import glob
from multiprocessing import Pool
import numpy as np
import os
from pull_proportion_turnover import get_ncfiles


def get_ptt_files(ncbase):
    """Get PTT and stats files from the specified base directory."""
    ptt_files = glob.glob(os.path.join(ncbase, 'ptt-flags', '*.nc'))
    stats_files = glob.glob(os.path.join(ncbase, 'turnstats', '*.nc'))

    return ptt_files, stats_files


def describe_turnover(args):
    """Describe the turnover trends for a PTT cube and save to NetCDF."""
    ptt_file, stats_file = args
    try:
        name = os.path.basename(ptt_file).split('.')[0].split('_masks_full')[0]  # Get the base name without extension
        print(f"Processing {name}")

        riv_ptt = xr.load_dataset(ptt_file)
        riv_stats = xr.load_dataset(stats_file)

        bulk_stats = xr.Dataset(coords=riv_stats.coords)
        bulk_stats['numturns'] = riv_stats.numturns
        bulk_stats['maxtime'] = riv_stats.maxtime

        bulk_stats['meantime'] = riv_ptt.PTT.mean(dim=['year'], skipna=True)
        bulk_stats['meantime_wetfor'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.mean(dim='year', skipna=True)
        bulk_stats['meantimed_dryfor'] = riv_ptt.where(riv_ptt.PTTFlags == 1).PTT.mean(dim='year', skipna=True)

        bulk_stats['maxtime_w4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.max(dim='year', skipna=True)
        bulk_stats['maxtimed_d4'] = riv_ptt.where(riv_ptt.PTTFlags == 1).PTT.max(dim='year', skipna=True)

        bulk_stats['medtime'] = riv_ptt.PTT.median(dim=['year'], skipna=True)
        bulk_stats['medtime_w4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.median(dim='year', skipna=True)
        bulk_stats['medtimed_w4'] = riv_ptt.where(riv_ptt.PTTFlags == 1).PTT.max(dim='year', skipna=True)

        bulk_stats['stdtime'] = riv_ptt.PTT.std(dim=['year'], skipna=True)
        bulk_stats['sdtime_w4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.std(dim='year', skipna=True)
        bulk_stats['sdtime_d4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.std(dim='year', skipna=True)

        output_path = f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt_bulkstats/{name}.nc'
        bulk_stats.to_netcdf(output_path)
    except Exception as e:
        print(f"Failed to process {ptt_file} and {stats_file}: {e}")


def process_ncs(ptt_files, stats_files):
    """Process NetCDF files using a pool of worker processes."""
    with Pool() as pool:
        pool.map(describe_turnover, zip(ptt_files, stats_files))


if __name__ == '__main__':
    base_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover'
    ptt_files, stats_files = get_ptt_files(base_path)
    process_ncs(ptt_files, stats_files)
