#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:36:05 2024
Process GRDC discharge data and calculate discharge metrics form Hansford et al., 2020 and intermittency

@author: safiya
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = '/Volumes/SAF_Data/SAF_Data/CHAPTER2/turnover_discharge_records/'
dv_df = pd.DataFrame(columns = ['river', 'intermittency', 'mean_DVIa', 'DVIc', 'mean_on_record'])
#%% FOR DAILY DATA
## create a place to store metadata
records = glob.glob(os.path.join(root, 'MOD_C02_1987-2023_may_ds_daily/*.txt'))
fig, ax = plt.subplots(3, 4, figsize  = (15, 8), tight_layout = True, dpi = 300)
ax = ax.ravel()

for a, record in enumerate(records):
    # # Open the file and extract metadata I already did this
    with open(record, 'r', encoding='latin1') as file:
        for line in file:
            if line.startswith('# River:'):
                river = line.split(':')[-1].strip()
            # if line.startswith('# Catchment area (km'):
    #             catchment_area = float(line.split(':')[-1].strip())
    #         if line.startswith('DATA'):
    #             break  # Stop after finding 'DATA' (end of metadata section)
               
    #     print(river, catchment_area)
    
    print(river)
    
    df = pd.read_csv(record, sep = ";",
                     skiprows=37, names = ['Date', 'Time', 'Discharge'], 
                     na_values = '-999.000', 
                     encoding='latin1')
    
    df['Time'] = df['Time'].replace('--:--', '00:00')
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df = df.drop(['Date', 'Time'], axis = 1)
    
    df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True, errors='coerce')
    df = df.dropna(how = 'any')
    
    # years = df['datetime'].dt.year.unique().tolist() ### get list of years
   
    ## get number of days in each year
    df['year'] = df['datetime'].dt.year
    df['date'] = df['datetime'].dt.date
    days_per_year = df.groupby('year')['date'].nunique().to_numpy()
    nyears = np.sum(days_per_year)/365.25 ## instead of reading the actual years im finding the num days con to year assuming a year is 365.25 days
    

    
    df = df.set_index('datetime') 
    ax[a].plot(df['Discharge'])
    ax[a].set_title(river)
    ax[a].set_ylabel('Discharge m3/s')
    ax[a].set_xlabel('Date')
    ## calculate discharge variability metrics. be wary of the fact that there are incomplete years
    
    # (1) intermittency
    # convert to m3/s to m3/day then sum to get annual total water flux
    # df['Discharge_m3/day'] = df['Discharge']*86400 ## s to day
    # waterflux_eachyr = df['Discharge_m3/day'].resample('YE').sum().to_numpy() ## how much water going through the system each year, ARRAY 
    # waterflux_eachyr = waterflux_eachyr[np.nonzero(waterflux_eachyr)]    ## drop nonzero years bc these arent counted in the calculation
    
    # bankfull_eachyr = df['Discharge_m3/day'].resample('YE').median().dropna(how = 'any')*days_per_year
    # intermit = waterflux_eachyr.sum()/(bankfull_eachyr.sum())#*nyears)

    ## calc yearly avergae discharge
    qw_avg_mo = df['Discharge'].resample('M').mean() ## average monthly discharge
    # Group by both year and month and calculate the average for each month across all years
    monthly_avg = df.groupby(df.index.month).mean()
    
    # Calculate the number of unique months that contributed to each average
    monthly_count = df.groupby([df.index.year, df.index.month]).size().groupby(level=1).size()
    
    # Rename the index to month names for better readability
    monthly_avg.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_count.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Combine the average and count dataframes
    monthly_stats = pd.DataFrame(index=monthly_avg.index)
    monthly_stats['average'] = monthly_avg['Discharge']
    monthly_stats['count'] = monthly_count
    # Rename columns for clarity
    monthly_stats.columns = ['average_value', 'month_count']


    qw_avg_dry = qw_avg_mo.resample('YE').min() ## average monthly discharge
    qw_avg_wet = qw_avg_mo.resample('YE').max() ## average monthly discharge
    
    qw_avg_yr = df['Discharge'].resample('YE').mean() ## average yearly dischagre
    ax[a].axhline(qw_avg_yr.mean(), c = 'r', ls = '--')
    dvia = (qw_avg_wet-qw_avg_dry)/qw_avg_yr
    dvia.to_csv(f'/Volumes/SAF_Data/SAF_Data/CHAPTER2/turnover_discharge_records/{river}_dvia.csv')
    
    dvic = (qw_avg_mo.max()-qw_avg_mo.min())/qw_avg_yr.mean()

    dvdata = pd.DataFrame({'river':[river],
                           'intermittency': [intermit],
                           'mean_DVIa': [np.mean(dvia)], 
                           'DVIc': [dvic],
                           'mean_on_record': df['Discharge'].mean()})
    
    dv_df = pd.concat((dv_df, dvdata))


#%% FOR MONTHY DATA
## create a place to store metadata
records = glob.glob(os.path.join(root, 'MOD_C02_1987-2023_may_ds_monthly/*.txt'))

fig, ax = plt.subplots(3, 4, figsize  = (15, 8), tight_layout = True, dpi = 300)
ax = ax.ravel()

for a, record in enumerate(records):
    # # Open the file and extract metadata I already did this
    with open(record, 'r', encoding='latin1') as file:
        for line in file:
            if line.startswith('# River:'):
                river = line.split(':')[-1].strip()
            # if line.startswith('# Catchment area (km'):
    #             catchment_area = float(line.split(':')[-1].strip())
    #         if line.startswith('DATA'):
    #             break  # Stop after finding 'DATA' (end of metadata section)
               
    #     print(river, catchment_area)
    
    print(river)
    
    df = pd.read_csv(record, sep = ";",
                     skiprows=39, names = ['Date', 'Time', 'Discharge', 'Calculated', 'Flag'], 
                     na_values = '-999.000', 
                     encoding='latin1')
    df = df.drop(['Calculated', 'Flag'], axis=1)    
    df['Time'] = df['Time'].replace('--:--', '00:00')
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df = df.drop(['Date', 'Time'], axis = 1)
    
    df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True, errors='coerce')

    df = df.dropna(how = 'any')
       
    ## get number of days in each year
    df['year'] = df['datetime'].dt.year
    df['date'] = df['datetime'].dt.date
    mos_per_year = df.groupby('year')['date'].nunique().to_numpy()
    nyears = np.sum(mos_per_year)/12#(365.25/12) ## instead of reading the actual years im finding the num days con to year assuming a year is 365.25 days
    
    df = df.set_index('datetime') 
    ax[a].plot(df['Discharge'])
    ax[a].set_title(river)
    ax[a].set_ylabel('Discharge m3/s')
    ax[a].set_xlabel('Date')
    ## calculate discharge variability metrics. be wary of the fact that there are incomplete years
    
    # (1) intermittency
    # convert to m3/s to m3/day then sum to get annual total water flux
    df['Discharge_m3/day'] = df['Discharge']*86400*(365.23/12) ## s to day to month
    waterflux_eachyr = df['Discharge_m3/day'].resample('YE').sum().to_numpy() ## how much water going through the system each year, ARRAY 
    waterflux_eachyr = waterflux_eachyr[np.nonzero(waterflux_eachyr)]    ## drop nonzero years bc these arent counted in the calculation
    
    bankfull_eachyr = df['Discharge_m3/day'].resample('YE').mean().dropna(how = 'any')*mos_per_year
    intermit = waterflux_eachyr.sum()/(bankfull_eachyr.sum())#*nyears)

    ## calc yearly avergae discharge
    qw_avg_mo = df['Discharge'].resample('ME').mean() ## average monthly discharge
    qw_avg_dry = qw_avg_mo.resample('YE').min() ## average monthly discharge
    qw_avg_wet = qw_avg_mo.resample('YE').max() ## average monthly discharge
    
    qw_avg_yr = df['Discharge'].resample('YE').mean() ## average yearly dischagre
    ax[a].axhline(qw_avg_yr.mean(), c = 'r', ls = '--')
    dvia = (qw_avg_wet-qw_avg_dry)/qw_avg_yr
    dvia.to_csv(f'/Volumes/SAF_Data/SAF_Data/CHAPTER2/turnover_discharge_records/{river}_dvia.csv')

    dvic = (qw_avg_mo.max()-qw_avg_mo.min())/qw_avg_yr.mean()

    dvdata = pd.DataFrame({'river':[river],
                           'intermittency': [intermit],
                           'mean_DVIa': [np.mean(dvia)], 
                           'DVIc': [dvic],
                           'mean_on_record': df['Discharge'].mean()})
    
    dv_df = pd.concat((dv_df, dvdata))
plt.plot(df['Discharge'])
#%% DEAL WITH INDUS

reaches = glob.glob(os.path.join(root, 'indus_monthly_discharge/*.txt'))
plt.figure(figsize = (6, 3), dpi = 300, tight_layout = True)
indus_sumdf = pd.DataFrame()

for r, reach in enumerate(reaches):
    df = pd.read_csv(reach, sep = ";",
                     skiprows=39, names = ['Date', 'Time', f'Original_{r}', 'Calculated', 'Flag'], 
                     na_values = '-999.000', 
                     encoding='latin1')
    df = df.drop(['Calculated', 'Flag'], axis=1)    
    df['Time'] = df['Time'].replace('--:--', '00:00')
    df['datetime'] = df['Date'] + ' ' + df['Time']
    df = df.drop(['Date', 'Time'], axis = 1)
    
    df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True, errors='coerce')
    
    df = df.dropna(how = 'any')
    df = df.set_index('datetime') 
    # print(reach)
    # print(df)
    indus_sumdf = pd.concat((indus_sumdf, df), axis = 1)
indus_sumdf = indus_sumdf.dropna(how = 'any')

indus_sum = pd.DataFrame(indus_sumdf.sum(axis = 1)).reset_index()
indus_sum['year'] = indus_sum['datetime'].dt.year
indus_sum['date'] = indus_sum['datetime'].dt.date
mos_per_year = indus_sum.groupby('year')['date'].nunique().to_numpy()

nyears = np.sum(mos_per_year)/12 ## instead of reading the actual years im finding the num days con to year assuming a year is 365.25 days

indus_sum = indus_sum.set_index('datetime') 
indus_sum = indus_sum.rename(columns = {0:'Discharge'})

plt.plot(indus_sum['Discharge'])
plt.title('Indus_sum')
plt.ylabel('Discharge m3/s')
plt.xlabel('Date')

## calculate discharge variability metrics. be wary of the fact that there are incomplete years
# (1) intermittency
# convert to m3/s to m3/day then sum to get annual total water flux
indus_sum['Discharge_m3/day'] = indus_sum['Discharge']*86400*(365.23/12) ## s to day, summed for a month because each value should be the amount of flow per month
waterflux_eachyr = indus_sum['Discharge_m3/day'].resample('YE').sum().to_numpy() ## how much water going through the system each year, ARRAY 
waterflux_eachyr = waterflux_eachyr[np.nonzero(waterflux_eachyr)]    ## drop nonzero years bc these arent counted in the calculation

bankfull_eachyr = indus_sum['Discharge_m3/day'].resample('YE').mean().dropna(how = 'any')*mos_per_year
intermit = waterflux_eachyr.sum()/(bankfull_eachyr.sum())#*nyears)

## calc yearly avergae discharge
qw_avg_mo = indus_sum['Discharge'].resample('ME').mean() ## average monthly discharge
qw_avg_dry = qw_avg_mo.resample('YE').min() ## average monthly discharge
qw_avg_wet = qw_avg_mo.resample('YE').max() ## average monthly discharge

qw_avg_yr = indus_sum['Discharge'].resample('YE').mean() ## average yearly dischagre
plt.axhline(qw_avg_yr.mean(), c = 'r', ls = '--')
dvia = (qw_avg_wet-qw_avg_dry)/qw_avg_yr
dvia.to_csv(f'/Volumes/SAF_Data/SAF_Data/CHAPTER2/turnover_discharge_records/Indus_r2_dvia.csv')

dvic = (qw_avg_mo.max()-qw_avg_mo.min())/qw_avg_yr.mean()

dvdata = pd.DataFrame({'river':'indus_sum',
                       'intermittency': [intermit],
                       'mean_DVIa': [np.mean(dvia)], 
                       'DVIc': [dvic],
                       'mean_on_record': indus_sum['Discharge'].mean()})

dv_df = pd.concat((dv_df, dvdata))