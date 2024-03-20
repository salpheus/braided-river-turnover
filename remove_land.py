#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 14:34:22 2024

@author: safiya
"""
import numpy as np
import copy

def make_land_nan(maskstack):
    
    '''make all land pixels nan'''
    maskstack_nan = copy.deepcopy(maskstack)*1.0
    maskstack_nan[maskstack==0] = np.nan
    
    return maskstack_nan
