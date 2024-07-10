# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from multiprocessing import Process
from whackamole import pixel_turn_time
import xarray as xr
import os
import glob
import numpy as np

# set up paths etc

def get_ncfiles(path):
    base = '/Volumes/SAF_Data/remote-data'
    ncpaths = glob.glob(os.path.join(path, '*.nc'))
    print(ncpaths)
    return ncpaths

def calc_turnover(ncfile):
    '''calculate turnover time using an input array of a netcdf file and export the following: 
        1. an nc with turnover stats for: (1) longest wet period, (2) longest dry, (3) longest all, (4) number ot turnovers
        2. an nc with the positions of turnover
        3. an nc of the lenggth of turnovers
    '''
    name = ncfile.split('/')[-1].split('.')[0]
    print(name)
    riv = xr.open_dataset(ncfile)
    mask_array = riv.masks.to_numpy()
    
    diffs = np.diff(mask_array, axis = 0)
    
    flags, ptt = pixel_turn_time(mask_array, diffs)
  
    ## insert a slize of nan values to the flags array

    #ptt is an array of nan with integers where there is turnover. The value of the integer is the length of turnover
    num_turns_array = np.count_nonzero(~np.isnan(ptt), axis = 0)
    max_turntime = np.nanmax(ptt, axis = 0)
    
    turnstats = xr.Dataset(coords = riv.coords)
    turnstats = turnstats.drop_vars('year')
    
    turnstats['numturns'] =  (['lat', 'lon'], num_turns_array)
    turnstats['maxtime'] =  (['lat', 'lon'], max_turntime)
    
    ##export turnover fils
    full_turn = xr.Dataset(coords = riv.coords)
    full_turn['PTT'] = (['year', 'lat', 'lon'], ptt)
    full_turn['PTTFlags'] = (['year', 'lat', 'lon'], flags)
    
    turnstats.to_netcdf(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats/{name}_stats.nc')
    full_turn.to_netcdf(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt-flags/{name}_full.nc')

# set up multiprocessing

def process_files(ncfiles):
    processes = []
    
    for ncfile in ncfiles:
        process = Process(target=calc_turnover, args=(ncfile,))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

if __name__ == '__main__':
    base_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc'
    ncfiles = get_ncfiles(base_path)
    process_files(ncfiles)