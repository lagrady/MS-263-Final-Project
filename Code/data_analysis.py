import numpy as np
import pandas as pd
from scipy import stats
from scipy import linalg
import xarray as xr
import matplotlib.pyplot as plt

def PCA(data):
    '''Takes a data array and creates an eigenvalue and eigenvector matrix, also returns
    variances as percentages and a factor loading matrix for principle component analysis.
    
    REQUIREMENTS:
    - Must not have any NaN values in the data
    
    INPUTS:
    data: any kind of data that can be converted into an array (must not have NaN's or gaps)
    
    OUTPUTS:
    - The standardized data array
    - The sorted eigenvalue matrix
    - The sorted percentage of variance matrix
    - The sorted eigenvector matrix
    - The factor loading matrix'''
    # Converts data into matrix format
    dat_array = data.values
    
    # Standardizes the data in the array so that mean is 0 with std of 1
    dat_mean = np.mean(dat_array, axis= 0)
    dat_stdev = np.std(dat_array, axis = 0, ddof = 1)
    dat_standardized = (dat_array - dat_mean)/dat_stdev
    
    # Creates a covariance matrix with the standardized data
    cov_mat = np.cov(dat_standardized, rowvar=False)
    
    # Creates a eigenvalue matrix and an eigenvector matrix
    val, vec = linalg.eig(cov_mat)
    
    # Convert eigenvalues into real numbers and sort both val and vec matrices
    val = np.real(val)
    ind = np.argsort(-1*val)
    val_sorted = val[ind]
    vec_sorted = vec[:,ind]
    
    # Calculate the percentage of variance accounted for by the eigenvalues
    val_frac = (val_sorted/np.sum(val_sorted))*100
    
    # Calculate a factor loading matrix for use in principal component analysis
    val_diag = np.diag(val_sorted)
    fac_load = np.matmul(vec_sorted, val_diag**.5)
    
    return dat_array, val_sorted, val_frac, vec_sorted, fac_load

### An experimental version of the PCA function which doesn't standardize the data array beforehand
### I'm not sure if it works properly, but I'm going to keep it on here just in case

# def PCA_ustd(data):
    
#     # Step 1: convert the data into matrix form
#     dat_array = data.values
#     cov_mat = np.cov(dat_array, rowvar=False)
#     val, vec = linalg.eig(cov_mat)
#     # The values in 'val' have a strange format, so I use np.real() to make them regular values. This technique was used in w9 lecture.
#     val = np.real(val)
#     ind = np.argsort(-1*val)
#     val_sorted = val[ind]
#     vec_sorted = vec[:,ind]
#     val_frac = (val_sorted/np.sum(val_sorted))*100
#     val_diag = np.diag(val_sorted)
#     fac_load = np.matmul(vec_sorted, val_diag**.5)
    
#     return val_sorted, val_frac, vec_sorted, fac_load

    