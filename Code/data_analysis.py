import numpy as np
import pandas as pd
from scipy import stats
from scipy import linalg
import xarray as xr
import matplotlib.pyplot as plt

def PCA(data):
    
    # Step 1: convert the data into matrix form
    dat_array = data.values
    dat_mean = np.mean(dat_array, axis= 0)
    dat_stdev = np.std(dat_array, axis = 0, ddof = 1)
    dat_standardized = (dat_array - dat_mean)/dat_stdev
    cov_mat = np.cov(dat_standardized, rowvar=False)
    val, vec = linalg.eig(cov_mat)
    # The values in 'val' have a strange format, so I use np.real() to make them regular values. This technique was used in w9 lecture.
    val = np.real(val)
    ind = np.argsort(-1*val)
    val_sorted = val[ind]
    vec_sorted = vec[:,ind]
    val_frac = (val_sorted/np.sum(val_sorted))*100
    val_diag = np.diag(val_sorted)
    fac_load = np.matmul(vec_sorted, val_diag**.5)
    
    return dat_array, val_sorted, val_frac, vec_sorted, fac_load

def PCA_ustd(data):
    
    # Step 1: convert the data into matrix form
    dat_array = data.values
    cov_mat = np.cov(dat_array, rowvar=False)
    val, vec = linalg.eig(cov_mat)
    # The values in 'val' have a strange format, so I use np.real() to make them regular values. This technique was used in w9 lecture.
    val = np.real(val)
    ind = np.argsort(-1*val)
    val_sorted = val[ind]
    vec_sorted = vec[:,ind]
    val_frac = (val_sorted/np.sum(val_sorted))*100
    val_diag = np.diag(val_sorted)
    fac_load = np.matmul(vec_sorted, val_diag**.5)
    
    return val_sorted, val_frac, vec_sorted, fac_load

    