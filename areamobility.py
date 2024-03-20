#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:19:22 2024

@author: safiya
"""
import numpy as np


def ari(maskstack, baseline_start = 0):
    '''with a defined baseline start recoalculate area based mobility metrics working iteratively through, between timesteps
       where A_ri = Ad (initial baseline dry px) - Ai (area of dry px yet to be touched)
       changing the baseline every time, then taking the average based on duration since baseline'''
       
    num_yrs = maskstack.shape[0]
    baseline_range = np.arange(baseline_start, maskstack.shape[0]-1)