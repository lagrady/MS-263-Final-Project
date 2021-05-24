import numpy as np
import pandas as pd
import data_prep_functions as dat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from scipy.signal import periodogram
from scipy.signal import welch
from scipy.stats import chi2
import os

def depth_pcol(base_df, title):
    depth = ['0m','5m','13m','21m']
    
    #month_int = [0,  22320,  42480,  64800,  86400, 108720, 130320, 152640,
       #174960, 196560, 218880, 240480]
    #month = ['January', 'February', 'March', 'April', 'May','June','July','August','September','October','November','December']
    temp_df = base_df.iloc[:, 1:]
    temp_arr = temp_df.to_numpy()
    temp_arr_t = temp_arr.T
    plt.figure(figsize=(10,4))
    plt.pcolor(base_df['date_time'], len(depth), temp_arr_t)
    plt.tight_layout()
    plt.colorbar(label = 'Temperature (C)')
    plt.gca().invert_yaxis()
    #plt.xticks(ticks = month_int[0:(end_month_num-start_month_num+1)], labels=month[start_month_num-1:end_month_num], rotation = 20)
    plt.xticks(rotation = 20)
    plt.yticks(ticks = np.arange(.5, len(depth)+.5, 1), labels=depth)
    plt.xlabel('Date')
    plt.ylabel('Depth (meters)')
    plt.title(str(title))
    plt.show()

def temp_line(base_df, title):
    plt.figure(figsize=(10,4))
    plt.plot(base_df['date_time'], base_df['temp_c0'], '-y')
    plt.plot(base_df['date_time'], base_df['temp_c5'], '-r')        
    plt.plot(base_df['date_time'], base_df['temp_c13'], '-b')
    plt.plot(base_df['date_time'], base_df['temp_c21'], '-k')
    plt.legend(['0m', '5m', '13m', '21m'],loc='lower right')
    plt.xlabel('Date')
    plt.ylabel('Temperature (C)')
    plt.xticks(rotation = 20)
    plt.title(str(title))
    
def avg_temp_line(base_df, title, start_month_num = 1, end_month_num = 12):
    month_int = [0,  22320,  42480,  64800,  86400, 108720, 130320, 152640,
       174960, 196560, 218880, 240480]
    month = ['January', 'February', 'March', 'April', 'May','June','July','August','September','October','November','December']
    if end_month_num == 12:
        sample_size = np.arange(month_int[start_month_num-1],month_int[end_month_num-1])
        plt.figure(figsize=(10,4))
        plt.plot(sample_size, base_df['temp_c0'].iloc[month_int[start_month_num-1]:month_int[end_month_num-1]], '-y')
        plt.plot(sample_size, base_df['temp_c5'].iloc[month_int[start_month_num-1]:month_int[end_month_num-1]], '-r')        
        plt.plot(sample_size, base_df['temp_c13'].iloc[month_int[start_month_num-1]:month_int[end_month_num-1]], '-b')
        plt.plot(sample_size, base_df['temp_c21'].iloc[month_int[start_month_num-1]:month_int[end_month_num-1]], '-k')
        plt.legend(['0m', '5m', '13m', '21m'],loc='lower right')
        plt.xlabel('Month')
        plt.ylabel('Temperature (C)')
        plt.title(str(title))
        plt.xticks(ticks = month_int[(start_month_num - 1):(end_month_num)], labels=month[start_month_num-1:end_month_num], rotation = 20)
    else:
        sample_size = np.arange(month_int[start_month_num-1],month_int[end_month_num])
        plt.figure(figsize=(10,4))
        plt.plot(sample_size, base_df['temp_c0'].iloc[month_int[start_month_num-1]:month_int[end_month_num]], '-y')
        plt.plot(sample_size, base_df['temp_c5'].iloc[month_int[start_month_num-1]:month_int[end_month_num]], '-r')
        plt.plot(sample_size, base_df['temp_c13'].iloc[month_int[start_month_num-1]:month_int[end_month_num]], '-b')
        plt.plot(sample_size, base_df['temp_c21'].iloc[month_int[start_month_num-1]:month_int[end_month_num]], '-k')
        plt.legend(['0m', '5m', '13m', '21m'],loc='lower right')
        plt.xlabel('Month')
        plt.ylabel('Temperature (C)')
        plt.title(str(title))
        plt.xticks(ticks = month_int[(start_month_num - 1):(end_month_num)], labels=month[start_month_num-1:end_month_num], rotation = 20)
    
def spec_analysis(data, samp_freq = 720, window = 'raw', unit = 'units', title = 'Spectral Analysis', lower_y=.00001, upper_y=1000):
    #data = data.dropna()
    N = len(data)
    fp,Sp = periodogram(data, fs=samp_freq)
    f3,S3 = welch(data, fs=samp_freq,nperseg=N,window='boxcar',detrend='linear')
    f4,S4 = welch(data, fs=samp_freq,nperseg=N/4,window='boxcar',detrend='linear')
    f5,S5 = welch(data, fs=samp_freq,nperseg=N/2,window='hamming',detrend='linear')
    f6,S6 = welch(data, fs=samp_freq,nperseg=N/4,window='hamming',detrend='linear')

    if window == str('raw'):
        # For the confidence interval
        edof = 2
        x2_upper = edof/chi2.ppf(.975, edof)
        x2_lower = edof/chi2.ppf(.025, edof)
    
        plt.figure(figsize=(10,4))
        plt.loglog(fp,Sp,'b')
        x = 10
        plt.plot(np.array([x,x]), 10*np.array([x2_lower,x2_upper]),'k-')
        plt.xlabel('frequency [cpd]')
        plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
        plt.title(str(title))
        plt.ylim([lower_y, upper_y])
    return fp, Sp
        
    elif window == str('boxcar1'):
        edof = 8
        x2_upper = edof/chi2.ppf(.975, edof)
        x2_lower = edof/chi2.ppf(.025, edof)
    
        plt.figure(figsize=(10,4))
        plt.loglog(f3,S3,'r')
        x = 10
        plt.plot(np.array([x,x]), 10*np.array([x2_lower,x2_upper]),'k-')
        plt.xlabel('frequency [cpd]')
        plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
        plt.title(str(title))
        plt.ylim([lower_y, upper_y])
    return f3, S3
        
    elif window == str('boxcar2'):
        edof = 8
        x2_upper = edof/chi2.ppf(.975, edof)
        x2_lower = edof/chi2.ppf(.025, edof)
    
        plt.figure(figsize=(10,4))
        plt.loglog(f4,S4,'g')
        x = 10
        plt.plot(np.array([x,x]), 10*np.array([x2_lower,x2_upper]),'k-')
        plt.xlabel('frequency [cpd]')
        plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
        plt.title(str(title))
        plt.ylim([lower_y, upper_y])
    return f4, S4
        
    elif window == str('hamming1'):
        edof = 8 * 2.5164
        x2_upper = edof/chi2.ppf(.975, edof)
        x2_lower = edof/chi2.ppf(.025, edof)
    
        plt.figure(figsize=(10,4))
        plt.loglog(f5,S5,'m')
        x = 10
        plt.plot(np.array([x,x]), 10*np.array([x2_lower,x2_upper]),'k-')
        plt.xlabel('frequency [cpd]')
        plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
        plt.title(str(title))
        plt.ylim([lower_y, upper_y])
    return f5, S5
        
    elif window == str('hamming2'):
        edof = 8 * 2.5164
        x2_upper = edof/chi2.ppf(.975, edof)
        x2_lower = edof/chi2.ppf(.025, edof)
    
        plt.figure(figsize=(10,4))
        plt.loglog(f3,S3,'r')
        x = 10
        plt.plot(np.array([x,x]), 10*np.array([x2_lower,x2_upper]),'k-')
        plt.xlabel('frequency [cpd]')
        plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
        plt.title(str(title))
        plt.ylim([lower_y, upper_y])
    return f6, S6
        
    else :
        plt.figure(figsize=(10,4))
        plt.loglog(fp,Sp,'b')
        plt.loglog(f3,S3,'r')
        plt.loglog(f4,S4,'g')
        plt.loglog(f5,S5,'m')
        plt.loglog(f6,S6,'y')
        plt.xlabel('frequency [cpd]')
        plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
        plt.title(str(title))
        plt.legend(['raw','pre-whitened','segment length = N/4','segment length = N/2, Hamming','segment length = N/4, Hamming'],loc='lower left')
        plt.ylim([lower_y, upper_y])

def vector_3d(base_df, year):
    fig = plt.figure(figsize= (8,15))

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter3D(base_df['eastward'], base_df['northward'], base_df['upwards'], c=base_df['upwards'], cmap='viridis')
    ax.set_ylabel('Northward velocity [m/s]')
    ax.set_zlabel('Upward velocity [m/s]')
    ax.view_init(0, 0)
    fig.suptitle('Depth averaged current velocity - ' + str(year),x=.5, y=.8, fontsize=16)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter3D(base_df['eastward'], base_df['northward'], base_df['upwards'], c=base_df['upwards'], cmap='viridis')
    ax.set_xlabel('Eastward velocity [m/s]')
    ax.set_zlabel('Upward velocity [m/s]')
    ax.view_init(0, 85)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    p = ax.scatter3D(base_df['eastward'], base_df['northward'], base_df['upwards'], c=base_df['upwards'], cmap='viridis')
    ax.set_xlabel('Eastward velocity [m/s]')
    ax.set_ylabel('Northward velocity [m/s]')
    ax.view_init(90, 90)

    plt.tight_layout()
    plt.show()

def vector_2d(base_df, year):
    plt.figure(figsize=(10,4))
    plt.plot(base_df['date_time'],base_df['eastward'])
    plt.plot(base_df['date_time'],base_df['northward'])
    plt.xlabel('Date')
    plt.ylabel('Current velocity [m/s]')
    plt.xticks(rotation = 20)
    plt.title('Depth averaged current velocity - ' + str(year))
    plt.legend(['eastward','northward','upward'])
    
def princax(u,v=None):
    '''
Principal axes of a vector time series.
Usage:
theta,major,minor = princax(u,v) # if u and v are real-valued vector components
    or
theta,major,minor = princax(w)   # if w is a complex vector
Input:
u,v - 1-D arrays of vector components (e.g. u = eastward velocity, v = northward velocity)
    or
w - 1-D array of complex vectors (u + 1j*v)
Output:
theta - angle of major axis (math notation, e.g. east = 0, north = 90)
major - standard deviation along major axis
minor - standard deviation along minor axis
Reference: Emery and Thomson, 2001, Data Analysis Methods in Physical Oceanography, 2nd ed., pp. 325-328.
Matlab function: http://woodshole.er.usgs.gov/operations/sea-mat/RPSstuff-html/princax.html
    '''

    # if one input only, decompose complex vector
    if v is None:
        w = np.copy(u)
        u = np.real(w)
        v = np.imag(w)

    # only use finite values for covariance matrix
    ii = np.isfinite(u+v)
    uf = u[ii]
    vf = v[ii]

    # compute covariance matrix
    C = np.cov(uf,vf)

    # calculate principal axis angle (ET, Equation 4.3.23b)
    theta = 0.5*np.arctan2(2.*C[0,1],(C[0,0] - C[1,1])) * 180/np.pi

    # calculate variance along major and minor axes (Equation 4.3.24)
    term1 = C[0,0] + C[1,1]
    term2 = ((C[0,0] - C[1,1])**2 + 4*(C[0,1]**2))**0.5
    major = np.sqrt(0.5*(term1 + term2))
    minor = np.sqrt(0.5*(term1 - term2))

    return theta,major,minor

def rot(u,v,theta):
    """
Rotate a vector counter-clockwise OR rotate the coordinate system clockwise.
Usage:
ur,vr = rot(u,v,theta)
Input:
u,v - vector components (e.g. u = eastward velocity, v = northward velocity)
theta - rotation angle (degrees)
Output:
ur,vr - rotated vector components
Example:
rot(1,0,90) returns (0,1)
    """

    w = u + 1j*v             # complex vector
    ang = theta*np.pi/180    # convert angle to radians
    wr = w*np.exp(1j*ang)    # complex vector rotation
    ur = np.real(wr)         # return u and v components
    vr = np.imag(wr)
    return ur,vr

def rot_vector(x, y, year):
    theta,major,minor = princax(x,y)
    rot_x, rot_y = rot(x, y, -theta-90)
    
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.plot(x, y,'.')
    plt.axis('equal')
    plt.axvline(x=0,c='k')
    plt.axhline(y=0,c='k')
    plt.xlabel('Eastward velocity [m/s]')
    plt.ylabel('Northward velocity [m/s]')
    plt.title('Geographic velocity - ' + str(year))

    plt.subplot(122)
    plt.plot(rot_x, rot_y,'.')
    plt.axis('equal')
    plt.axvline(x=0,c='k')
    plt.axhline(y=0,c='k')
    plt.xlabel('Cross shore velocity [m/s]')
    plt.ylabel('Alongshore velocity [m/s]')
    plt.title('Velocity rotated along principle axes')
    plt.tight_layout()
    plt.show()
    
    return base_df