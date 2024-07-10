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
    
    raster_values = []
    for point in points:
        row, col = raster.index(point.x, point.y)
        raster_values.append(raster.read(1)[row, col])
        
    return raster_values, points

def process_river(river, results_base, all_river_tiffs, years):
    print(f"Processing river: {river}")

    centerline_fol = os.path.join(results_base, river)
    
    fig, ax = plt.subplots(5, 5, figsize=(15, 15), dpi=True, sharex=True, sharey=True)
    ax = ax.ravel()
    error_years = []
    
    for a, year in enumerate(years):
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
                
                raster_values = np.array(raster_values, dtype=int)
                
                edges = np.diff(raster_values)
                l_edges = np.where(edges == 1)[0]
                r_edges = np.where(edges == -1)[0]
                
                widths = r_edges - l_edges
                
                wetted_width = np.nansum(widths)
                
                if wetted_width > 0:
                    ebi = -1 * np.nansum((widths / wetted_width) * np.log2(widths / wetted_width))
                    ebi = 2 ** ebi
                else:
                    ebi = 0
                
                transect_ebi.loc[idx, 'ebi'] = ebi
                transect_ebi.loc[idx, 'wetted_width'] = wetted_width
            
            ax[a].plot(transect_ebi.index.values, transect_ebi['ebi'])
            ax[a].set_title(f'{river} river, {year}')
            ax[a].set_xlabel('transect number')
            ax[a].set_ylabel('ebi')
            
            transect_ebi.to_csv(os.path.join(results_base, river, f'ebi_ww_{year}.csv'))
        except Exception as e:
            print(f"Error processing year {year} for river {river}: {e}")
            error_years.append(year)
    
    plt.savefig(os.path.join(results_base, river, f'{river}_ebi.png'))

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
