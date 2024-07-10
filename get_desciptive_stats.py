#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:21:43 2024
 
Pull stats and add them to the turn stats netcdf
@author: safiya
"""
import xarray as xr
import pandas as pd
import glob
from multiprocessing import Process
import numpy as np
import os
from pull_proportion_turnover import get_ncfiles


def get_ptt_files(ncbase): #ncbase is the path to the turnover base root, collect all the files in the path
    
    ptt_files = glob.glob(os.path.join(ncbase, 'ptt-flags', '*.nc'))
    stats_files = glob.glob(os.path.join(ncbase, 'turnstats', '*.nc'))

    return(ptt_files, stats_files)


def describe_turnover(ptt_file, stats_file):
    '''describe the turnover trends for a ptt cube then add it to the turnstats netcdfs'''
    
    name = ptt_file.split('/')[-1].split('.')[0].split('_masks_full')[0] ## nc file should be the masks_full.nc file
   
    print(name)
    riv_ptt = xr.load_dataset(ptt_file)
    riv_stats = xr.load_dataset(stats_file)
    
    bulk_stats = xr.Dataset(coords = riv_stats.coords)
    
    bulk_stats['numturns'] = riv_stats.numturns
    bulk_stats['maxtime'] = riv_stats.maxtime
    
    bulk_stats['meantime'] = riv_ptt.PTT.mean(dim = ['year'], skipna = True)
    bulk_stats['meantime_wetfor'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.mean(dim = 'year', skipna = True) # flags describe the stage that occurred previously 
    bulk_stats['meantimed_dryfor'] = riv_ptt.where(riv_ptt.PTTFlags == 1).PTT.mean(dim = 'year', skipna = True)
    
    bulk_stats['maxtime_w4']= riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.max(dim = 'year', skipna = True) # flags describe the stage that occurred previously 
    bulk_stats['maxtimed_d4'] = riv_ptt.where(riv_ptt.PTTFlags == 1).PTT.max(dim = 'year', skipna = True)
    
    bulk_stats['medtime'] = riv_ptt.PTT.median(dim = ['year'], skipna = True)
    bulk_stats['medtime_w4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.median(dim = 'year', skipna = True)
    bulk_stats['medtimed_w4'] = riv_ptt.where(riv_ptt.PTTFlags == 1).PTT.max(dim = 'year', skipna = True)
    
    bulk_stats['stdtime'] = riv_ptt.PTT.std(dim = ['year'], skipna = True)
    bulk_stats['sdtime_w4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.std(dim = 'year', skipna = True)
    bulk_stats['sdtime_d4'] = riv_ptt.where(riv_ptt.PTTFlags == -1).PTT.std(dim = 'year', skipna = True)
    
    bulk_stats.to_netcdf(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt_bulkstats/{name}.nc')
    
# set up multiprocessing

def process_ncs(ptt_files, stats_files):
    processes = []
    
    for pttf, statsf in zip(ptt_files, stats_files):
        process = Process(target=describe_turnover, args=(pttf, statsf,))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

if __name__ == '__main__':
    base_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover'
    ptt_files, stats_files= get_ptt_files(base_path)
    process_ncs(ptt_files, stats_files)
    
    
    
    
    