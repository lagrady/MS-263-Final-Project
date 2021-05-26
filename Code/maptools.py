import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def make_map(projection=ccrs.PlateCarree()):
    '''
    Return axis object for cartopy map with labeled gridlines.
    
    Input: Cartopy projection, default = cartopy.crs.PlateCarree()
    Output: Cartopy axis object
    
    Based on blog post by Filipe Fernandes:
    https://ocefpaf.github.io/python4oceanographers/blog/2015/06/22/osm/
    License: Creative Commons Attribution-ShareAlike 4.0
    https://creativecommons.org/licenses/by-sa/4.0/
    
    Example code:
    
    import cartopy.crs as ccrs
    plt.figure()
    ax = make_map(projection=ccrs.Mercator())
    ax.coastlines()
    '''
    ax = plt.axes(projection=projection)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax