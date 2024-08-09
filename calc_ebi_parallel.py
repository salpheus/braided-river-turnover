import os
import math
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio.mask import mask
import glob
import rasterio.features
from multiprocessing import Pool

# Function to extract raster values along a linestring
def extract_raster_values_along_line(linestring, raster):
    num_points = int(linestring.length)  # or adjust based on your need
    points = [linestring.interpolate(float(i) / num_points, normalized=True) for i in range(num_points + 1)]
   
    # Convert points to numpy arrays of x and y coordinates
    x_coords = np.array([point.x for point in points])
    y_coords = np.array([point.y for point in points])
    
    # Get the raster indices for all points
    rows, cols = np.array(raster.index(x_coords, y_coords))

    # Check if any point is out of bounds
    if np.any(rows < 0) or np.any(rows >= raster.height) or np.any(cols < 0) or np.any(cols >= raster.width):
        return None  # Skip the entire line if any point is out of bounds

    # Read the raster data once
    raster_data = raster.read(1)

    # Extract the values using the indices
    raster_values = raster_data[rows, cols]
    
    return raster_values.tolist()

def process_river(river, results_base, all_river_tiffs, years):
    print(f"Processing river: {river}")

    centerline_fol = os.path.join(results_base, river)
    
    # fig, ax = plt.subplots(5, 5, figsize=(15, 15), dpi=True, sharex=True, sharey=True)
    # ax = ax.ravel()
    error_years = []
    
    for a, year in enumerate(years):
        save_path = os.path.join(results_base, river, f'ebi_ww_{year}.csv')
        if os.path.exists(save_path):
            continue
        try:
            transect_path = os.path.join(centerline_fol, str(year), f'{river}_meshlines.shp')
            if not os.path.exists(transect_path):
                continue
            
            transect = gpd.read_file(transect_path)
            
            raster_path = glob.glob(os.path.join(all_river_tiffs, river, 'mask/1999on', f'*{year}*.tif'))
            if not raster_path:
                continue
            
            raster = rasterio.open(raster_path[0])
            
            if transect.crs != raster.crs:
                transect = transect.to_crs(raster.crs)

            transect_ebi = pd.DataFrame(columns=['ebi', 'wetted_width'], index=transect['FID'])
            
            for idx, row in transect.iterrows():
                linestring = row.geometry
                raster_values = extract_raster_values_along_line(linestring, raster)
                
                if raster_values:
                    raster_values = np.array(raster_values, dtype=int)
                    
                    edges = np.diff(raster_values)
                    l_edges = np.where(edges == 1)[0]
                    r_edges = np.where(edges == -1)[0]
                    
                    if len(l_edges) > len(r_edges):
                        widths = r_edges - l_edges[:len(r_edges)]
                    
                    elif len(r_edges)>len(l_edges):
                        widths = r_edges[1:] - l_edges
                    
                    else:
                        widths = r_edges - l_edges
                    
                    wetted_width = np.nansum(widths)
                    
                    if wetted_width > 0:
                        ebi = -1 * np.nansum((widths / wetted_width) * np.log2(widths / wetted_width))
                        ebi = 2 ** ebi
                    else:
                        ebi = 0
                    
                    transect_ebi.loc[idx, 'ebi'] = ebi
                    transect_ebi.loc[idx, 'wetted_width'] = wetted_width
            
            # ax[a].plot(transect_ebi.index.values, transect_ebi['ebi'])
            # ax[a].set_title(f'{river} river, {year}')
            # ax[a].set_xlabel('transect number')
            # ax[a].set_ylabel('ebi')
            
            transect_ebi.to_csv(save_path)
        except Exception as e:
            print(f"Error processing year {year} for river {river}: {e}")
            error_years.append(year)
    
    # plt.savefig(os.path.join(results_base, river, f'{river}_ebi.png'))

    if error_years:
        print(f"Errors occurred for river {river} in years: {error_years}")

# Main script
if __name__ == '__main__':
    all_river_tiffs = '/Volumes/SAF_Data/remote-data/watermasks/C02_1987-2023_may'
    all_rivers = os.listdir(all_river_tiffs)
    exclude_list = ['brahmaputra_yangcun', '.DS_Store', 'congo_new', 'agubh2']
    all_rivers = [riv for riv in all_rivers if riv not in exclude_list]
    
    results_base = '/Volumes/SAF_Data/remote-data/rivgraph_transects_curated'
    years = np.arange(1999, 2024)

    with Pool() as pool:
        pool.starmap(process_river, [(river, results_base, all_river_tiffs, years) for river in all_rivers])

#%%% for yukon and locs with a lot of missing points

def extract_raster_values_along_line(linestring, raster):
    num_points = int(linestring.length)  # or adjust based on your need
    points = [linestring.interpolate(float(i) / num_points, normalized=True) for i in range(num_points + 1)]
    
    # Convert points to numpy arrays of x and y coordinates
    x_coords = np.array([point.x for point in points])
    y_coords = np.array([point.y for point in points])
    
    raster_values = []
    valid_points = []
    
    # Read the raster data once
    raster_data = raster.read(1)
    
    for x, y in zip(x_coords, y_coords):
        # Get the raster indices for the current point
        row, col = raster.index(x, y)
        
        # Check if the point is within raster bounds
        if 0 <= row < raster.height and 0 <= col < raster.width:
            val = raster_data[row, col]
            raster_values.append(val)
            valid_points.append((x, y))
        # Skip the point if it is out of bounds
    
    return raster_values, valid_points
#%% GO IN AFTER TO PROCESS BAD YEARS

river = 'ob_down'
years = np.arange(2005, 2009)

centerline_fol = os.path.join(results_base, river)

error_years = []

for a, year in enumerate(years):
    
    print(year)
    save_path = os.path.join(results_base, river, f'ebi_ww_{year}.csv')
    if os.path.exists(save_path):
        continue
    try:
        transect_path = os.path.join(centerline_fol, str(year), f'{river}_meshlines.shp')
        if not os.path.exists(transect_path):
            continue
        
        transect = gpd.read_file(transect_path)
        
        raster_path = glob.glob(os.path.join(all_river_tiffs, river, 'mask/1999on', f'*{year}*.tif'))
        if not raster_path:
            continue
        
        raster = rasterio.open(raster_path[0])
        
        if transect.crs != raster.crs:
            transect = transect.to_crs(raster.crs)
    
        transect_ebi = pd.DataFrame(columns=['ebi', 'wetted_width'], index=transect['FID'])
        
        for idx, row in transect.iterrows():
            linestring = row.geometry
            raster_values, points = extract_raster_values_along_line(linestring, raster)
            
            if raster_values:
                raster_values = np.array(raster_values, dtype=int)
                
                edges = np.diff(raster_values)
                l_edges = np.where(edges == 1)[0]
                r_edges = np.where(edges == -1)[0]
                
                if len(l_edges) > len(r_edges):
                    widths = r_edges - l_edges[:len(r_edges)]
                
                elif len(r_edges)>len(l_edges):
                    widths = r_edges[1:] - l_edges
                
                else:
                    widths = r_edges - l_edges
                
                wetted_width = np.nansum(widths)
                
                if wetted_width > 0:
                    ebi = -1 * np.nansum((widths / wetted_width) * np.log2(widths / wetted_width))
                    ebi = 2 ** ebi
                else:
                    ebi = 0
                
                transect_ebi.loc[idx, 'ebi'] = ebi
                transect_ebi.loc[idx, 'wetted_width'] = wetted_width
        
        transect_ebi.to_csv(os.path.join(results_base, river, f'ebi_ww_{year}.csv'))
    except Exception as e:
        print(f"Error processing year {year} for river {river}: {e}")
        error_years.append(year)

# plt.savefig(os.path.join(results_base, river, f'{river}_ebi.png'))

if error_years:
    print(f"Errors occurred for river {river} in years: {error_years}")
    
    
#%%

for river in all_rivers:
    fig, ax = plt.subplots(5, 5, figsize = (20, 17), tight_layout = True, sharex = True, sharey = True)
    ax = ax.ravel()
    for a, file in enumerate(glob.glob(os.path.join(results_base, river, '*.csv'))):
        name = file.split('/')[-1].split('.csv')[0]
        df = pd.read_csv(file)
        
        ax[a].plot(df.index.values, df['ebi'], 'k', lw = .8, zorder = 100)
        ax[a].set_title(f'{river}, med = --, {name}')
        
        desc = df.describe()
        ax[a].fill_between(np.arange(len(df)), desc.loc['25%', 'ebi'], desc.loc['75%', 'ebi'], 
                           fc = 'xkcd:light grey', alpha = 0.5 )
        ax[a].axhline(y = desc.loc['50%', 'ebi'], c = 'r', ls = '--')
        ax[a].axhline(y = desc.loc['mean', 'ebi'], c = 'r', ls = '-')
        
        
    plt.savefig(os.path.join(results_base, river, f'{river}_ebi.png'))
    
    
        
        