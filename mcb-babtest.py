#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 01:41:46 2024
ls regress bar ratios
@author: safiya
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

bardata = pd.read_excel('/Volumes/SAF_Data/SAF_Data/remote-data/rivgraph_transects_curated/model_ebi_bi_plot.xlsx', index_col = 0, sheet_name = 'remapped-22oct')
#%%

# biM, biC, biR, biP, biSE = stats.linregress(bardata['end TS BI'], bardata['MCB:BAB'])


# fit line y = mx (gradient has to be 0 because if there is no braiding, there are no bars)
x = np.vstack([bardata['end TS BI'].to_numpy(), np.zeros(len(bardata['end TS BI'].to_numpy()))]).T
y = bardata['MCB:BAB']

lsx, res, rank, s = np.linalg.lstsq(x, y)
## from inverse theory notes once the chi2 value approaches n (observations) or normalised approaches 1 its a good fit
norm_chi = 1/len(x) * np.sum(
    ((y-(lsx[0]*x[:, 0]))**2)/(np.var(y))**2)

xvals = np.linspace(0, 7)
yvals = lsx[0]*xvals + lsx[1]

plt.figure(figsize = (6, 6), dpi = 300, tight_layout = True)

plt.scatter(bardata['end TS BI'], bardata['MCB:BAB'], s = 125, c = 'k', ec = 'k', label = 'Observed data')
plt.plot(xvals, yvals, c = 'r', label = f'Least squares regression \n y = {np.round(lsx[0], 2)}BI$_{a}$, $Χ^{2}$ = {np.round(norm_chi, 2)}')
plt.xlabel('Bradied Index, $BI_{a}$')
plt.ylabel('MCB-BAB ratio')
# plt.title(f'y = {np.round(lsx[0], 2)}x, $Χ^{2}$ = {np.round(norm_chi, 2)}')
plt.legend()
plt.savefig('/Volumes/SAF_Data/SAF_Data/CHAPTER3/manu_figs/barratio-regression.svg', transparent = True)
