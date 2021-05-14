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

def depth_pcol(base_df, title, start_month_num=1, end_month_num=12):
    depth = ['0m','5m','13m','21m']
    month_int = np.arange(0,len(base_df)+1,(len(base_df)/11), dtype=int)
    month = ['January', 'february', 'march', 'april', 'may','june','july','august','september','october','november','december']
    temp_df = base_df.iloc[month_int[start_month_num-1]:month_int[end_month_num], 1:]
    temp_arr = temp_df.to_numpy()
    temp_arr_t = temp_arr.T
    plt.figure(figsize=(10,4))
    plt.pcolor(temp_arr_t)
    plt.tight_layout()
    plt.colorbar(label = 'Temperature (C)')
    plt.gca().invert_yaxis()
    plt.xticks(ticks = month_int[0:(end_month_num-start_month_num+1)], labels=month[start_month_num-1:end_month_num], rotation = 20)
    plt.yticks(ticks = np.arange(.5, len(depth)+.5, 1), labels=depth)
    plt.xlabel('Month')
    plt.ylabel('Depth (meters)')
    plt.title(str(title))
    plt.show()
    
def depth_line(base_df, title, start=1, end=12):
    month_int = np.arange(0,len(base_df)+1,(len(base_df)/11), dtype=int)
    month = ['January', 'february', 'march', 'april', 'may','june','july','august','september','october','november','december']
    plt.figure(figsize=(10,4))
    plt.plot(base_df['date_time'].iloc[month_int[start-1]:month_int[end]], base_df['temp_c0'].iloc[month_int[start-1]:month_int[end]], '-y')
    plt.plot(base_df['date_time'].iloc[month_int[start-1]:month_int[end]], base_df['temp_c5'].iloc[month_int[start-1]:month_int[end]], '-r')
    plt.plot(base_df['date_time'].iloc[month_int[start-1]:month_int[end]], base_df['temp_c13'].iloc[month_int[start-1]:month_int[end]], '-b')
    plt.plot(base_df['date_time'].iloc[month_int[start-1]:month_int[end]], base_df['temp_c21'].iloc[month_int[start-1]:month_int[end]], '-k')
    plt.legend(['0m', '5m', '13m', '21m'],loc='lower right')
    plt.xlabel('Month')
    plt.ylabel('Temperature (C)')
    plt.title(str(title))
    plt.xticks(ticks = month_int[0:(end-start+1)], labels=month[start-1:end], rotation = 20)
    
def spec_analysis(data, samp_freq = 720, edof = 2, unit = 'units', title = 'Spectral Analysis', lower_y=.00001, upper_y=1000):
    data = data.dropna()
    N = len(data)
    fp,Sp = periodogram(data, fs=samp_freq)
    f3,S3 = welch(data, fs=samp_freq,nperseg=N,window='boxcar',detrend='linear')
    f4,S4 = welch(data, fs=samp_freq,nperseg=N/4,window='boxcar',detrend='linear')
    f5,S5 = welch(data, fs=samp_freq,nperseg=N/2,window='hamming',detrend='linear')
    f6,S6 = welch(data, fs=samp_freq,nperseg=N/4,window='hamming',detrend='linear')

    # For the confidence interval
    x2_upper = edof/chi2.ppf(.975, edof)
    x2_lower = edof/chi2.ppf(.025, edof)
    
    plt.figure(figsize=(10,4))
    plt.loglog(fp,Sp,'b')
    plt.loglog(f3,S3,'r')
    plt.loglog(f4,S4,'g')
    plt.loglog(f5,S5,'m')
    plt.loglog(f6,S6,'y')
    x = 10
    plt.plot(np.array([x,x]), 10*np.array([x2_lower,x2_upper]),'k-')
    plt.xlabel('frequency [cpd]')
    plt.ylabel('PSD [' + str(unit) + '$^2$ cpd$^{-1}$]')
    plt.title(str(title))
    plt.legend(['raw','pre-whitened','segment length = N/4','segment length = N/2, Hamming','segment length = N/4, Hamming'],loc='lower left')
    plt.ylim([lower_y, upper_y])
    