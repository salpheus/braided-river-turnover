#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 20:18:22 2024

@author: safiya
"""
import panadas as pd
import matplotlib.pyplot as plt
import numpy as np

inventory = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/watermasks/admin/inventory-offline.xlsx', sheet_name = 'inventory_py', index_col=0)
#%% make figure comparing discharge measurements

plt.figure(figsize = (10, 4), dpi = 150)
ax = plt.gca()
ax.set_xticks(range(len(inventory))) ## alays set tick range before prescribing labels
ax.set_xticklabels(inventory.index.values);
ax.xaxis.set_tick_params(rotation=70)

plt.plot(inventory['mean_annu_qw_sc'], marker = 'o', label = 'WBMSed, Cohen et al., 2013, 2022', ls = '--')
plt.plot(inventory['mean_qw_annu_m3s'], marker = 'o', label = 'RAPID, Lin., et al 2019, David, 2019', ls = '--')
plt.plot(inventory['mean_on_record'], marker = 'o', label = 'GRDC', ls = '--')

inventory['mean_discharge_global'] = np.mean((inventory['mean_on_record'], 
                                              inventory['mean_qw_annu_m3s'], 
                                              inventory['mean_annu_qw_sc']), axis = 0)
plt.plot(inventory['mean_discharge_global'], label = 'Group average', c = 'k', marker = 'o')
plt.legend()
plt.ylabel('Discharge $m^3/s$')