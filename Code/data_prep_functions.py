import numpy as np
from scipy import stats
import pandas as pd
import xarray as xr
import fp_plotting as fpp
from physoce import tseries as ts
import datetime
import os

def therm_data_merge(equipment, year, d):
    '''Take the PISCO thermistor chain data from a designated year and merge all months
    and depths of data into a single, continuous dataframe.

    REQUIREMENTS:
    - Data path for raw data files is Data/SWC001/[equipment]/[year]/[depths of thermistors]
    - 'depth of thermstors' directories must be named '[depth]m' 
        - EX: '0m', '5m', etc.

    ADDITIONAL NOTE: 'depths of thermistors' is the depth of each thermistor along the mooring. For this specific site at Stillwater
    Cove (SWC001), there are thermistors at 0m, 5m, 13m, and 21m. Other sites may have thermistors at different depths. In this case,
    the directories will have to be renamed accordingly, and the 'depth' list will have to be changed as well.

    EXAMPLE: To create a dataframe of all 4 thermistors in the year 2000 at 0m, 5m, 13m, and 21m
    depth = [0, 5, 13, 21]
    df = therm_data_merge('Thermistor', 2000, depth)

    INPUTS:
    equipment - the name of the directory in which the years of data are stored
    year - the name of the directory in which the raw data files are stored
    d - a list of integers that correspond to the depths of the thermistors along the mooring
    
    OUTPUT:
    A pandas dataframe with columns for date_time and temperatures for all values in the 'depth' array'''

    # Create a variable for the list of depths from 'd'
    depth = d

    # Create a dataframe that has an index of datetimes from 1/1/[year] - 12/31/[year]
    base_df = pd.DataFrame({"date_time" : pd.date_range(str(year) + '-01-01', str(year) + '-12-31', tz='UTC', freq = '2T')})

    # A loop that combines all months of raw data files for each depth of 'd'
    for i in depth:
        # Creates a variable 'path' with the filename and then generates a list of all raw data files in that directory, excluding the first one
        path = 'Data/SWC001/' + str(equipment) + '/' + str(year) + '/' + str(i) + 'm/'
        files = os.listdir(path)
        file_len = list(range(1, len(files)))

        # Reads the first raw datafile in the directory and converts it into a pandas dataframe 'df1'
        df1 = pd.read_csv(("Data/SWC001/" + str(equipment) + "/" + str(year) + "/" + str(i) + "m/" + str(files[0])), delimiter='\s+', skiprows=[1],             parse_dates=[[0, 1]])

        # A loop which reads the next raw datafile in the directory, creates a dataframe 'df2', then appends df1 with df2
        for val in file_len:
            df2 = pd.read_csv(("Data/SWC001/" + str(equipment) + "/" + str(year) + "/" + str(i) + "m/" + str(files[val])), delimiter='\s+', skiprows=               [1], parse_dates=[[0, 1]])
            df1 = df1.append(df2, ignore_index=True)

        # Once all raw datafiles for a single depth are combined, the temperature and datetime columns
        # are converted to numeric and datetime types
        df1['temp_c'] = pd.to_numeric(df1['temp_c'])
        df1['date_time'] = pd.to_datetime(df1['date_time'])

        # Due to errors with the thermistors, some temperatures are reported as 9999 degrees, so these datapoints are removed
        df1 = df1[df1['temp_c'] < 9000]

        # Creates a finalized dataframe 'df_final' and merges it with the date index, 'base_df' from before
        df_final = pd.DataFrame({"date_time" : df1['date_time'], "temp_c" + str(i) : df1['temp_c']})
        base_df = pd.merge(base_df, df_final, how='outer', on='date_time')
        base_df = base_df.drop_duplicates(subset='date_time', keep='first')
        # After df_final is merged with base_df, the loop iterates to the next depth from list 'd', and merges that
        # dataframe with the base_df, ensuring that all depths of data are now on a single, consistent timeline in
        # one all-encompassing dataframe
    return base_df

def therm_pd_to_xr(data, depth):
    '''Take the dataframe created from 'therm_data_merge' function and convert it into an analysis friendly xarray dataset.

    EXAMPLE: To convert a dataframe 'df' from adcp_data_merge into an xarray dataset 'ds'
    df = adcp_data_merge('ADCP', 2000)
    ds = adcp_pd_to_xr(df)

    VARIABLES:
    df - the dataframe created from adcp_data_merge'''

    # Create an array of unique values of 'date_time' and 'height' df
    # These arrays are used as the dimensions of the array when df is converted to an xarray dataset
    unique_datetime = np.unique(data['date_time'])
    date_time = pd.to_datetime(unique_datetime, utc= False)
    unique_depth = depth

    # Create an xarray dataset using the pivoted velocity components using the unique date_time and height values as the dimensions
    ds = xr.Dataset({'temperature': (('time', 'depth'), data.iloc[:,1:])},
                       {'time': date_time.values, 'depth':unique_depth})
    return ds

def xr_therm_analysis(ds, time_int, dt, lp_filter=40):
    '''Takes the xarray dataset produced from therm_pd_to_xr and creates filtered temperature variables.
    
    INPUTS:
    ds: dataset produced from adcp_pd_to_xr
    time_int: the desired time interval you wish to average your adcp samples over in od.datetime format 
    - ex: 1M (1 month), 1H (1 hour), 30T (30 minutes)
    dt: the sampling frequency in hours 
    - ex: 30 minute sampling frequency is 1/2
    lp_filter: the lower bound for a low pass filter
    - ex: a value of 40 would filter out any variability under a timescale of 40 hours
    
    OUTPUT:
    An xarray dataset with dimension time and depth with variables:
    - Resampled temperature with specified time interval
    - Temperature with specified low-pass filter'''
    
    ds = ds.resample(time=str(time_int)).mean()
    
    t_filt = ts.pl64(ds.temperature, dt, T = lp_filter)
    t_dat_arr = xr.Dataset({'temperature_filtered': (('time', 'depth'), t_filt)},
                       {'time': ds['time'], 'depth':ds['depth']})
    ds = ds.assign(temperature_filtered = t_dat_arr.temperature_filtered)

    return ds

def adcp_data_merge(equipment, year):
    '''Take the PISCO adcp data from a designated year and merge all months of data into a single, continuous dataframe.

    REQUIREMENTS:
    Data path for raw data files is Data/SWC001/[equipment]/[year]/21m

    ADDITIONAL NOTE: '21m' is the bottom depth of the adcp for this specific site. If another site with a different bottom depth
    is used, then the directory, as well as the final string in the 'path' variable will have to be renamed accordingly. This also
    applies to 'SWC001', which is the name of the mooring from the online data repsitory.

    EXAMPLE: To create a dataframe of adcp data at 21m from the year 2000
    df = adcp_data_merge('ADCP', 2000)

    VARIABLES:
    equipment - the name of the directory in which the years of data are stored
    year - the name of the directory in which the raw data files are stored'''

    # Creates a string for the filepath for ADCP data
    path = 'Data/SWC001/' + str(equipment) + '/' + str(year) + '/' + '21m/'
    files = os.listdir(path)

    # Creates a list of filenames where the first filename in the '21m' directory is omitted
    file_len = list(range(1, len(files)))

    # Creates dataframe 'df1' with the first raw data table in the directory
    # The first file is the first month(s) of data, and serves as a starting point to merge the future data with
    df1 = pd.read_csv((str(path) + str(files[0])), na_values = 9999, skiprows = 1, delimiter=' ', header = None, parse_dates=[[0,1]])

    # A for loop which reads the raw data files in chronological order then creates a new dataframe called 'df2'
    # df1 is appended with df2, and the loop continues until all data files are appended to df1
    for val in file_len:
        df2 = pd.read_csv((str(path) + str(files[val])), na_values = 9999, skiprows = 1, delimiter=' ', header = None, parse_dates=[[0,1]])
        df1 = df1.append(df2, ignore_index=True)

    # With the new, completely merged dataframe, each column is properly renamed, a redundant column known as 'unknown' is removed, and all
    # data that exceeds a depth of 22 (the max depth of the adcp) is removed
    df1 = df1.rename(columns = {'0_1':'date_time', 2:'yearday', 3:'height', 4:'depth', 5:'waterdepth', 6:'temp_c', 7:'pressure', 8:'intensity'
                             , 9:'data_quality', 10:'eastward', 11:'northward', 12:'upwards', 13:'errorvelocity', 14:'flag', 15:'unknown'})
    df1 = df1.drop(columns=['unknown'])
    df1 = df1[(df1['height']<22)]
    df1 = df1.drop_duplicates(subset=['date_time', 'height'], keep='first')

    return df1

def adcp_pd_to_xr(data):
    '''Take the dataframe created from 'adcp_data_merge' function and convert it into an analysis friendly xarray dataset.

    EXAMPLE: To convert a dataframe 'df' from adcp_data_merge into an xarray dataset 'ds'
    df = adcp_data_merge('ADCP', 2000)
    ds = adcp_pd_to_xr(df)

    VARIABLES:
    df - the dataframe created from adcp_data_merge'''

    # Create an array of unique values of 'date_time' and 'height' df
    # These arrays are used as the dimensions of the array when df is converted to an xarray dataset
    unique_datetime = np.unique(data['date_time'])
    date_time = pd.to_datetime(unique_datetime, utc= False)
    unique_depth = np.unique(data['depth'])

    # Use pd.pivot to manipulate df so that all velocity components (eastward, northward, upwards) are properly sorted
    # into their associated height in the water column
    pivoted_eastward = data.pivot(index='date_time',columns='depth',values='eastward')
    pivoted_northward = data.pivot(index='date_time',columns='depth',values='northward')
    pivoted_upwards = data.pivot(index='date_time',columns='depth',values='upwards')

    # Create an xarray dataset using the pivoted velocity components using the unique date_time and height values as the dimensions
    ds = xr.Dataset({'northward': (('time', 'depth'), pivoted_northward),
                         'eastward': (('time', 'depth'), pivoted_eastward)},
                       {'time': date_time.values, 'depth':unique_depth})
    return ds

def xr_adcp_analysis(ds, time_int, dt, lp_filter=40):
    '''Takes the dataset output by adcp_pd_to_xr and produces necessary variables for principal component analysis. Filters and averages out data,
    converts vectors to alongshore and across shore velocities, also creates depth averaged velocity variables.
    
    INPUTS:
    ds: dataset produced from adcp_pd_to_xr
    time_int: the desired time interval you wish to average your adcp samples over in od.datetime format 
    - ex: 1M (1 month), 1H (1 hour), 30T (30 minutes)
    dt: the sampling frequency in hours 
    - ex: 30 minute sampling frequency is 1/2
    lp_filter: the lower bound for a low pass filter
    - ex: a value of 40 would filter out any variability under a timescale of 40 hours
    
    OUTPUT:
    - An xarray dataset with dimensions of time and depth and variables:
        - low pass filtered northward and eastward velocity
        - depth averaged northward and eastward velocity
        - alongshore and across shore velocities
        - depth averaged alongshore and across shore velocities'''
    
    ds = ds.resample(time=str(time_int)).mean()
    
    n_filt = ts.pl64(ds.northward, dt, T = lp_filter)
    nf_dat_arr = xr.Dataset({'northward_filtered': (('time', 'depth'), n_filt)},
                       {'time': ds['time'], 'depth':ds['depth']})
    ds = ds.assign(northward_filtered = nf_dat_arr.northward_filtered)
    e_filt = ts.pl64(ds.eastward, dt, T = lp_filter)
    ef_dat_arr = xr.Dataset({'eastward_filtered': (('time', 'depth'), e_filt)},
                       {'time': ds['time'], 'depth':ds['depth']})
    ds = ds.assign(eastward_filtered = ef_dat_arr.eastward_filtered)
    
    north_da = ds.northward_filtered.mean(axis=1)
    ds = ds.assign(north_da = north_da)
    east_da = ds.eastward_filtered.mean(axis=1)
    ds = ds.assign(east_da = east_da)
    
    theta,major,minor = fpp.princax(ds.east_da, ds.north_da)
    crossshore,alongshore = fpp.rot(ds.eastward_filtered, ds.northward_filtered,-theta-90)
    
    ds = ds.assign(alongshore = alongshore)
    ds = ds.assign(crossshore = crossshore)
    
    da_v = ds.alongshore.mean(axis=1)
    ds = ds.assign(alongshore_da = da_v)
    da_u = ds.crossshore.mean(axis=1)
    ds = ds.assign(crossshore_da = da_u)
    
    return ds


def df_date_sel(data, start_date, end_date):
    '''Select specific dates to extract from a pandas dataframe.
    
    REQUIREMENTS:
    - Must have a column called date_time which is a type datetime[ns]
    - Must be a pandas dataframe (xarray has its own method for extracting specific dates)
    
    INPUTS:
    data: a pandas dataframe with a date_time column
    start_date/end_date: the start and end dates that you want extracted from your dataframe
    
    OUTPUT:
    - A new pandas dataframe which has all the same columns as the original dataframe, but it's
    only the range of dates you specified'''
    
    mask = (data['date_time'] >= str(start_date)) & (data['date_time'] <= str(end_date))
    new_df = data.iloc[:,:].loc[mask]
    
    return new_df

