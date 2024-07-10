#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:25:38 2024
 build a netcdf file with statstics on the spatial turnover for each river using parallel computing
@author: safiya
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
from multiprocessing import Process
import glob


def get_ncfiles(path):
    ncpaths = glob.glob(os.path.join(path, '*.nc'))
    return ncpaths
    
def get_nc_stats(ncfile):
    '''make a mastercsv file for each river turnover file and populate it with descriptive statistics for each year on record'''
    
    name = ncfile.split('/')[-1].split('.')[0].split('_masks_full')[0] ## nc file should be the masks_full.nc file
    print(name)
    riv = xr.open_dataset(ncfile)
    
    areafile = pd.read_csv(f'/Volumes/SAF_Data/remote-data/watermasks/admin/mask_database_csv/C02_1987-2023_allLS_db_csv/{name}_sum_summary_skip.csv',
                           skiprows = np.arange(1, 13), header = 0, usecols = ['year', 'wet px']).set_index('year')

    # csv structure
    ### yr | wet px | total turn | total turn wet | total turn dry | total prop turn | total prop turn wet | total prop turn dry
    
    ## recall flags: -1 = went dry, 1 = went wet, flags have zeros, ptt has nans
    
    total_turn = riv.PTT.count(dim = ['lat', 'lon']).to_dataframe()
    total_prop_turn = total_turn['PTT']/areafile['wet px']

    turn_wet = riv.PTTFlags.where(riv.PTTFlags == 1).count(dim = ['lat', 'lon']).to_dataframe() # find amount of cells that went wet per year
    total_turn_wet = turn_wet['PTTFlags']/areafile['wet px']

    turn_dry = riv.PTTFlags.where(riv.PTTFlags == -1).count(dim = ['lat', 'lon']).to_dataframe() # find amount of cells that went wet per year
    total_turn_dry = turn_dry['PTTFlags']/areafile['wet px']

    areafile = pd.concat([areafile, 
                          total_turn, turn_wet, turn_dry, 
                          total_prop_turn, total_turn_wet, total_turn_dry], axis = 1)
    areafile.columns = ['wet_px_area', 
                               'total_turnover', 'total_turnwet', 'total_turndry',
                               'prop_turnover', 'prop_turnwet', 'prop_turndry']
                                                                                    
    areafile.to_csv(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats/propturn/{name}_propturn.csv')                                                                              

def process_files(ncfiles):
    processes = []
    
    for ncfile in ncfiles:
        process = Process(target=get_nc_stats, args=(ncfile,))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

if __name__ == '__main__':
    base_path = '/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/ptt-flags'
    ncfiles = get_ncfiles(base_path)
    process_files(ncfiles)
    
    
    
    
#%% pull all csvs and make one master csv dataset

# csvs = glob.glob(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/turnstats/propturn/*_propturn.csv')
# rivlist = [c.split('/')[-1].split('_propturn.csv')[0] for c in csvs] ## get a list of names
# dataframes = [pd.read_csv(f, header = 0, index_col = 0) for f in csvs]

# ## convert dframes to DataArrays then combine into a Dataset
# data_arrays = []
# for df in dataframes: 
#     da = xr.DataArray(df,
#                       dims = ['year', 'variables'], 
#                       coords = {'year': df.index.values, 
#                                 'variables': df.keys()})
#     data_arrays.append(da)
    
# ## concatenate into an xarray dataset
# time_turnover = xr.concat(data_arrays, dim = 'river')
# time_turnover = time_turnover.assign_coords(river = rivlist)

# time_turnover.to_netcdf(f'/Volumes/SAF_Data/remote-data/arrays/C02_1987-2023_allLS_db/1999_nc_turnover/masters/allrivers_spatial-turnover-stats.nc')
    














