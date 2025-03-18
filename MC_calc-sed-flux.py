#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:09:34 2024

@author: safiya
"""

# ## MATLAB script for monte carlo simulation on stratigraphic paleoslope to bedload model
# clear all
# close all
# clc

# #Inputs
# load('stratmeas.mat'); #comment vars below if using measured vals
# #go to random var generation to switch random values

# #D50=.0003; #median grain diameter (m)
# # Hbf=5;   #mean bankfull depth (m)
# # sHbf=2; #standard error of mean bankfull depth (m)
# # Tsm=0.3;   #mean bedset thickness (m)
# # sTsm=0.1; #standard error of mean set thickness
# # width=200; #mean channel width (m)
# # swidth=20; #standard error of mean channel width (m)

# nsamples=10000000;   #number of iterations of each parameter
# p=0.35; #bed porosity

# #parameter descriptions
# # for paleoslope model from Trampush et al. 2014
# a0=-2.08; #mean alpha0
# sa0=0.036; #standard dev alpha0
# a1=0.254; #mean alpha1
# sa1=0.016; #standard dev alpha1
# a2=-1.09; #mean alpha2
# sa2=0.044; #standard dev alpha2
# #for bedform height from leClair and Bridge 2001 for set thickness to bedform height
# g=2.9;  #mean bedform height relative to set thickness
# sg=0.7; #standard dev of g

# #from Mahon et al., in prep slope-migration rate parameter
# b1=1.305; #mean beta1  on slope
# sb1=.0515; #standard dev beta1
# b0=0.6113; #mean beta0 intercept term
# sb0=0.144; #standard dev beta0

# #generate random values for parameters 
# tic
# ra0=normrnd(a0,sa0,1,nsamples);
# ra1=normrnd(a1,sa1,1,nsamples);
# ra2=normrnd(a2,sa2,1,nsamples);
# rb1=normrnd(b1,sb1,1,nsamples);
# rb0=normrnd(b0,sb0,1,nsamples);
# rg=normrnd(g,sg,1,nsamples);
# rTsm=normrnd(Tsm,sTsm,1,nsamples);
# rTsm(rTsm<=0)=Tsm;
# # rHbf=normrnd(Hbf,sHbf,1,nsamples);
# # rHbf(rHbf<=0)=Hbf;
# # rwidth=normrnd(width,swidth,1,nsamples);
# # rwidth(rwidth<=0)=width;
# #rTsm=datasample(Tsm,nsamples)';
# rHbf=datasample(Hbf,nsamples)'; #sample from measured bar heights
# rwidth=datasample(width,nsamples)'; #sample from measured channel widths

# #calculate paleoslope after Trampush et al. 2014
# pslope=10.^(ra0+ra1.*log10(D50)+ra2.*log10(rHbf)); 
# # calculate bedload fluxes
# qs=0.5*(1-p).*rg.*rTsm.*(10.^(rb0+(rb1.*log10(pslope))));  #unit width bedload (m2/s) after Simons et al., 1965
# Qs=qs.*rwidth; #bedload flux (m3/s)

# #summary statistics
# meanS=mean(pslope)
# meanQ=mean(Qs)
# meanq=mean(qs)
# medQ=median(Qs)
# devQ=std(Qs)
# skewQ=skewness(Qs)
# Quantq=quantile(qs,[0.02 0.09 .25 .5 .75 .91 .98])
# QuantQ=quantile(Qs,[0.02 0.09 .25 .5 .75 .91 .98])

# toc

# figure 
# edges=logspace(-6,-1,100);
# histogram(qs,edges);
# xlabel('unit bedload flux (m2/s)');
# ylabel('probability')
# set(gca,'xscale','log')

# figure 
# edges=logspace(-4,-2,100);
# histogram(pslope, edges)
# xlabel('paleoslope (-)');
# ylabel('probability')
# set(gca,'xscale','log')

# figure 
# edges=logspace(-3,2,10000);
# histogram(Qs,edges);
# xlabel('bedload flux (m3/s)');
# ylabel('probability')
# set(gca,'xscale','log')


#%% rewriting rob mahons script in py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)
import pandas as pd

data = pd.read_excel('/Users/safiya/Desktop/calculate sed flux_defense.xlsx', sheet_name = 'py-compile', index_col = 0)
heights = pd.read_excel('/Users/safiya/Desktop/calculate sed flux_defense.xlsx', sheet_name = 'bar-heights')
#%% ROBs initial vlaues

# Parameters
D50 = 0.0003  # median grain diameter (m)
Hbf = 5  # mean bankfull depth (m)
sHbf = 2  # standard error of mean bankfull depth (m)
Tsm = 0.3  # mean bedset thickness (m)
sTsm = 0.1  # standard error of mean set thickness
width = 200  # mean channel width (m)
swidth = 20  # standard error of mean channel width (m)

nsamples = 10_000_000  # number of iterations of each parameter
p = 0.35  # bed porosity

# Paleoslope model parameters (from Trampush et al. 2014)
a0, sa0 = -2.08, 0.036  # mean and standard deviation for alpha0
a1, sa1 = 0.254, 0.016  # mean and standard deviation for alpha1
a2, sa2 = -1.09, 0.044  # mean and standard deviation for alpha2

# Bedform height parameters (from LeClair and Bridge 2001)
g, sg = 2.9, 0.7  # mean and standard deviation for g

# Slope-migration rate parameters (Mahon et al., in prep)
b1, sb1 = 1.305, 0.0515  # mean and standard deviation for beta1
b0, sb0 = 0.6113, 0.144  # mean and standard deviation for beta0

# Generate random values for parameters
ra0 = np.random.normal(a0, sa0, nsamples)
ra1 = np.random.normal(a1, sa1, nsamples)
ra2 = np.random.normal(a2, sa2, nsamples)
rb1 = np.random.normal(b1, sb1, nsamples)
rb0 = np.random.normal(b0, sb0, nsamples)
rg = np.random.normal(g, sg, nsamples)

# Random values for Tsm, ensuring positive values
rTsm = np.random.normal(Tsm, sTsm, nsamples)
rTsm[rTsm <= 0] = Tsm

# Sampled values for Hbf and width
rHbf = np.random.choice([Hbf], nsamples)
rwidth = np.random.choice([width], nsamples)

# Calculate paleoslope (Trampush et al. 2014)
pslope = 10 ** (ra0 + ra1 * np.log10(D50) + ra2 * np.log10(rHbf))

# Calculate bedload fluxes
qs = 0.5 * (1 - p) * rg * rTsm * (10 ** (rb0 + (rb1 * np.log10(pslope))))
Qs = qs * rwidth  # total bedload flux (m^3/s)

# Summary statistics
meanS = np.mean(pslope)
meanQ = np.mean(Qs)
meanq = np.mean(qs)
medQ = np.median(Qs)
devQ = np.std(Qs)
skewQ = (np.mean((Qs - meanQ) ** 3)) / (devQ ** 3)
Quantq = np.quantile(qs, [0.02, 0.09, 0.25, 0.5, 0.75, 0.91, 0.98])
QuantQ = np.quantile(Qs, [0.02, 0.09, 0.25, 0.5, 0.75, 0.91, 0.98])

# Print results
print("Mean Paleoslope:", meanS)
print("Mean Total Bedload Flux (Qs):", meanQ)
print("Mean Unit Width Bedload Flux (qs):", meanq)
print("Median Total Bedload Flux (Qs):", medQ)
print("Standard Deviation of Qs:", devQ)
print("Skewness of Qs:", skewQ)
print("Quantiles of qs:", Quantq)
print("Quantiles of Qs:", QuantQ)

# Ensure the values are positive for log histograms
qs = qs[qs > 0]
pslope = pslope[pslope > 0]
Qs = Qs[Qs > 0]

# Figure 1: Histogram of qs (unit bedload flux)
edges_qs = np.logspace(-6, -1, 100)
plt.figure()
plt.hist(qs, bins=edges_qs, density=True, edgecolor='black')
plt.xscale('log')
plt.xlabel('Unit bedload flux (m²/s)')
plt.ylabel('Probability')
plt.title('Histogram of Unit Bedload Flux (qs)')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

#%%
# Figure 2: Histogram of pslope (paleoslope)

## script for monte carlo simulation on stratigraphic paleoslope to bedload model
clear all
close all
clc

#Inputs
load('stratmeas.mat'); #comment vars below if using measured vals
#go to random var generation to switch random values

#D50=.0003; #median grain diameter (m)
# Hbf=5;   #mean bankfull depth (m)
# sHbf=2; #standard error of mean bankfull depth (m)
# Tsm=0.3;   #mean bedset thickness (m)
# sTsm=0.1; #standard error of mean set thickness
# width=200; #mean channel width (m)
# swidth=20; #standard error of mean channel width (m)

nsamples=10000000;   #number of iterations of each parameter
p=0.35; #bed porosity

#parameter descriptions
# for paleoslope model from Trampush et al. 2014
a0=-2.08; #mean alpha0
sa0=0.036; #standard dev alpha0
a1=0.254; #mean alpha1
sa1=0.016; #standard dev alpha1
a2=-1.09; #mean alpha2
sa2=0.044; #standard dev alpha2
#for bedform height from leClair and Bridge 2001 for set thickness to bedform height
g=2.9;  #mean bedform height relative to set thickness
sg=0.7; #standard dev of g

#from Mahon et al., in prep slope-migration rate parameter
b1=1.305; #mean beta1  on slope
sb1=.0515; #standard dev beta1
b0=0.6113; #mean beta0 intercept term
sb0=0.144; #standard dev beta0

#generate random values for parameters 
tic
ra0=normrnd(a0,sa0,1,nsamples);
ra1=normrnd(a1,sa1,1,nsamples);
ra2=normrnd(a2,sa2,1,nsamples);
rb1=normrnd(b1,sb1,1,nsamples);
rb0=normrnd(b0,sb0,1,nsamples);
rg=normrnd(g,sg,1,nsamples);

rTsm=normrnd(Tsm,sTsm,1,nsamples);
rTsm(rTsm<=0)=Tsm;
# rHbf=normrnd(Hbf,sHbf,1,nsamples);
# rHbf(rHbf<=0)=Hbf;
# rwidth=normrnd(width,swidth,1,nsamples);
# rwidth(rwidth<=0)=width;
#rTsm=datasample(Tsm,nsamples)';
rHbf=datasample(Hbf,nsamples); #sample from measured bar heights
rwidth=datasample(width,nsamples); #sample from measured channel widths

#calculate paleoslope after Trampush et al. 2014
pslope=10.^(ra0+ra1.*log10(D50)+ra2.*log10(rHbf)); 
# calculate bedload fluxes
qs=0.5*(1-p).*rg.*rTsm.*(10.^(rb0+(rb1.*log10(pslope))));  #unit width bedload (m2/s) after Simons et al., 1965
Qs=qs.*rwidth; #bedload flux (m3/s)

#summary statistics
meanS=mean(pslope)
meanQ=mean(Qs)
meanq=mean(qs)
medQ=median(Qs)
devQ=std(Qs)
skewQ=skewness(Qs)
Quantq=quantile(qs,[0.02 0.09 .25 .5 .75 .91 .98])
QuantQ=quantile(Qs,[0.02 0.09 .25 .5 .75 .91 .98])

toc

figure 
edges=logspace(-6,-1,100);
histogram(qs,edges);
xlabel('unit bedload flux (m2/s)');
ylabel('probability')
set(gca,'xscale','log')

figure 
edges=logspace(-4,-2,100);
histogram(pslope, edges)
xlabel('paleoslope (-)');
ylabel('probability')
set(gca,'xscale','log')

figure 
edges=logspace(-3,2,10000);
histogram(Qs,edges);
xlabel('bedload flux (m3/s)');
ylabel('probability')
set(gca,'xscale','log')
#%% calculate unit qs

flierprops = dict(marker='o', markerfacecolor='xkcd:gray', markersize=2,  markeredgecolor='xkcd:gray')
meanprops = dict(marker = 'o', markerfacecolor = 'blue', ms = 0, mec = 'k', mew = 0, linestyle = '--', linewidth = 1.5, color = 'k')
meanlineprops = dict(linestyle = '-', lc = 'k', lw = 2)
boxprops = dict(color = 'k', linewidth = 1.5)
capprops = dict(color = 'k', linewidth = 1.5)
whiskerprops = dict(color = 'k', linecolor = 'k')
boxwidth = 0.7
linewidth = 1.5
# plt.figure(figsize = (6*1.5, 2*1.5), tight_layout = True, dpi = 300)

fig, ax = plt.subplots(1, 6, figsize = (15, 3.5), tight_layout = True, dpi = 300, sharex = False, sharey = True)
ax = ax.ravel()
# Paleoslope model parameters (from Trampush et al. 2014)
a0, sa0 = -2.08, 0.036  # mean and standard deviation for alpha0
a1, sa1 = 0.254, 0.016  # mean and standard deviation for alpha1
a2, sa2 = -1.09, 0.044  # mean and standard deviation for alpha2

# Bedform height parameters (from LeClair and Bridge 2001)
g, sg = 2.9, 0.7  # mean and standard deviation for g

# Slope-migration rate parameters (Mahon et al., in prep)
b1, sb1 = 1.305, 0.0515  # mean and standard deviation for beta1
b0, sb0 = 0.6113, 0.144  # mean and standard deviation for beta0

nsamples = 10_000_000  # number of iterations of each parameter
p = 0.35  # bed porosity

sTsm = 0.15  # standard error of mean set thickness

cols = ['#f9b197', '#add799', '#f9b197', '#f9b197', '#83cee7', '#83cee7']

for idx, riv in enumerate(data.index.values):
    
    
    # Generate random values for parameters
    ra0 = np.random.normal(a0, sa0, nsamples)
    ra1 = np.random.normal(a1, sa1, nsamples)
    ra2 = np.random.normal(a2, sa2, nsamples)
    rb1 = np.random.normal(b1, sb1, nsamples)
    rb0 = np.random.normal(b0, sb0, nsamples)
    rg = np.random.normal(g, sg, nsamples)


    d50mm = data.loc[riv, 'd50mm']/1000
    # hbfm = data.loc[riv, 'hbf_m']
    Tsm = data.loc[riv, 'tsm_m']
    hbfm = heights[riv].dropna(how = 'all')
    
    # Random values for Tsm, ensuring positive values
    rTsm = np.random.normal(Tsm, sTsm, nsamples)
    rTsm[rTsm <= 0] = Tsm
    
    meanhbf = np.mean(hbfm)
    sdhbf = np.std(hbfm)

    # Sampled values for Hbf and width
    # rHbf = np.random.normal(meanhbf, sdhbf, nsamples) ##cgpt
    # rHbf[rHbf<=0] = meanhbf
    rHbf = np.random.choice(hbfm, nsamples) ##cgpt

    # Calculate paleoslope (Trampush et al. 2014)
    pslope = 10 ** (ra0 + ra1 * np.log10(d50mm) + ra2 * np.log10(rHbf))
    
    # Calculate bedload fluxes
    qs = (0.5 * (1 - p) * rg * rTsm * (10 ** (rb0 + (rb1 * np.log10(pslope)))))*365.25*86400
    
    # Summary statistics
    meanS = np.mean(pslope)
    meanq = np.mean(qs)
    Quantq = np.quantile(qs, [0.02, 0.09, 0.25, 0.5, 0.75, 0.91, 0.99])
    
    # Print results
    print("Mean Paleoslope:", meanS)
    print("Mean Unit Width Bedload Flux (qs):", meanq)
    print("Quantiles of qs:", Quantq)


    # qs = qs[qs>Quantq[0]]
    ax[idx].hist(qs, bins = np.arange(0, Quantq.max(), 250), ec = 'k', lw = .35, fc = cols[idx],  density = False)
    ax[idx].set_title(data.index.values[idx])
    ax[idx].axvline(np.median(qs), c = 'k', ls = '--', lw = 1, label = f'Median = {int(np.round(np.median(qs), 0))}')
    ax[idx].axvline(qs.mean(), c = 'k', lw = 1,  label = f'Mean = {int(np.round(np.mean(qs), 0))}')
    ax[idx].xaxis.set_major_locator(MultipleLocator(2500))
    ax[idx].xaxis.set_minor_locator(MultipleLocator(500))
    
    ax[idx].legend()
    
    ax[idx].tick_params(axis='both', which='minor', labelcolor = 'k')
    # ax[idx].set_xscale('log')
ax[3].set_xlabel('Unit Sed Flux (m$^{2}$/yr')
ax[0].set_ylabel('Count')

#     cutoff = len(qs)
#     qs = qs[qs<np.quantile(qs, 0.999)]
#     perc_cut = 1-(len(qs)/cutoff)
    
#     print('percent cut out = ', perc_cut)
#     violins = plt.violinplot(qs, positions = [idx], 
#                     showmedians = False, showextrema = True, widths=boxwidth, bw_method = 0.3)

#     for pc in violins['bodies']:
#         pc.set_facecolor('white')
#         pc.set_edgecolor('black')
#         pc.set_alpha(1)
#         pc.set_linewidth(.5)
 
#         violins['cmins'].set_color('k')
#         violins['cmins'].set_linewidth(1)
#         violins['cmaxes'].set_color('k')
#         violins['cmaxes'].set_linewidth(1)
#         violins['cbars'].set_color('k')
#         violins['cbars'].set_linewidth(1)

#     qsmin, q1, qsmed, q3, emax = np.nanquantile(qs, [.05, .25, .5, .75, .95])

#     # plt.vlines(idx+1, ymin = emin, ymax = emax, ec = 'k', zorder = 100)
#     plt.vlines(idx, ymin = q1, ymax = q3, ec = 'k', lw  = 5, zorder = 101)
#     plt.scatter(idx, qsmed, c = 'w', marker = 'o', s = 5, zorder = 102)


# ax = plt.gca()
# ax.set_xticks(np.arange(idx+1)) ## alays set tick range before prescribing labels
# ax.set_xticklabels(data.index.values);
# ax.set_ylabel('Unit Sed Flux ($m^{2}/yr$')
plt.savefig('/Volumes/SAF_Data/SAF_Data/CHAPTER3/manu_figs/mc_sedflux_redo.svg')


#%% PLOTS




# Ensure the values are positive for log histograms
qs = qs[qs > 0]
pslope = pslope[pslope > 0]
Qs = Qs[Qs > 0]

# Figure 1: Histogram of qs (unit bedload flux)
edges_qs = np.logspace(-6, -1, 100)
plt.figure()
plt.hist(qs, bins=edges_qs, density=True, edgecolor='black')
plt.xscale('log')
plt.xlabel('Unit bedload flux (m²/s)')
plt.ylabel('Probability')
plt.title('Histogram of Unit Bedload Flux (qs)')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

#%% just doing a regular shear stress inversion
# d90 = [1e-3, .5e-3, 1e-3, .75e-3, .5e-3, .5e-3] ## in the regular order
# chezy = 18*np.log10(4*data['hbf_m']/d90)
rho = 1650 #kgm3
g = 9.81 # ms2

qs_nd = 8*((data['tb_avg']-0.047)**.5)
qs = 


