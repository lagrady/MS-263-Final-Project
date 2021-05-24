This is Logan Grady's final project for MS 263, spring 2021

Data information:

All data is from the PISCO SWC001 mooring placed in Stillwater Cove. This data and all metadata is publicy accessable on https://search.dataone.org/data. To find the specific 
PISCO sensors used in this project, locate Carmel Bay on the map and zoom in so that only Carmel Bay is visible. The dataone website will automatically display all 
available data files in the scroll hud to the left of the map. Above the hud, there is a dropdown menu labeled 'Sort by', choose 'Identifier (a-z)'. This will automatically 
sort all sensors by their name, at which point SWC001 is easily findable in the scroll hud.

All ADCP and Thermistor data used in this project from SWC001 is also available in this google drive folder:
https://drive.google.com/drive/folders/10_3WSimUFLbI_mWnKSYleNXRFCC2vJGl?usp=sharing
You will need to be logged into your Moss Landing email in order to access it. More information on proper importation of the data folders below in 'Data importation' section


Data importation:

For the code in this project to work properly, there is a specific filepath that must be used in your personal directories. From your notebook's directory, you must have the
filepath 'Data/SWC001/', then two separate directories labeled 'ADCP' and 'Thermistor' within the 'SWC001' folder. Once these folders have been created, transfer the desired 
year of data from the google drive into the associated ADCP or Thermistor folder in your personal workbook repository. Each year must be placed in the correct folder and in its 
entirety, or the data import functions will not work properly.

EX: This project uses ADCP and Thermistor data from 2002. Go into the google drive folder, select the '2002' folder from 'ADCP' and transfer it into your local directory, 
'Data/SWC001/ADCP/'. I would then do the same for desired 'Thermistor' data, 'Data/SWC001/Thermistor/'

Once data is properly imported into your personal notebook, it can be read as a pandas dataframe or xarray dataset. There are the relevant functions to do so located in 
'data_prep_functions.py'


Data analysis:

All functions for data importation, plotting, and analysis are located in the files 'data_prep_functions.py', 'fp_plotting.py', and 'data_analysis.py', respectively. Other
functions used in this project are from public packages such as numpy, matplotlib.pyplot, pandas, xarray, and scipy. There are also functions from a custom package used in
order to perform low-pass filtration on the data.

This package is called 'physoce', created by Tom Connolly at Moss Landing Marine Labs. In order to use 'physoce':
- Open whatever command prompt/terminal you use for python and enter: 
pip install git+https://github.com/physoce/physoce-py

- Once physoce is installed, it can be imported into your project with: 
from physoce import tseries as ts
