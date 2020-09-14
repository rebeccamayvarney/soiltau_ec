#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:46:43 2020

@author: rmv203
"""

#%%

# Analysis imports
import numpy as np
import numpy.ma as ma

import csv

# netcdf4 imports
import netCDF4
from netCDF4 import Dataset

# Iris imports
import iris
import iris.coord_categorisation
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes

# My functions
from rmv_cmip_analysis import combine_netCDF_time_overlap
from rmv_cmip_analysis import combine_netCDF_observations
from rmv_cmip_analysis import combine_netCDF_observations_temp
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import annual_average
from rmv_cmip_analysis import numpy_to_cube
from rmv_cmip_analysis import regrid_model
from rmv_cmip_analysis import area_average
from rmv_cmip_analysis import global_total
from rmv_cmip_analysis import global_total_percentage

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams, colors
from matplotlib import gridspec as gspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpat
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D




#%% Loading the observational datasets

# cube for regridding
regrid_cube = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
regrid_cube = time_average(regrid_cube)
regrid_cube.coord('latitude').guess_bounds()
regrid_cube.coord('longitude').guess_bounds()
regrid_modelcube = regrid_cube
# correct lat and lon dimensions
n_lat = regrid_cube.coord('latitude').points
n_lon = regrid_cube.coord('longitude').points


## MODIS NPP
npp_file = Dataset('/home/links/rmv203/obs_datasets/MOD17A3_Science_NPP_mean_00_14_regridhalfdegree.nc')
npp_data = npp_file.variables['npp'][:]*1e-3


### CARD

cubes = iris.load('/home/links/rmv203/obs_datasets/CARDAMOM_2001_2010_FL_RHE.nc')
for cube in cubes:
    if cube.var_name == 'longitude':
        lon = cube
    if cube.var_name == 'latitude':
        lat = cube
    if cube.var_name == 'Mean':
        mean_cube = cube

#### Takes the latitude and longitude ‘cubes’ and makes them in to coordinates
lat_aux = iris.coords.AuxCoord(lat.data, standard_name=lat.name(), units=lat.units)
lon_aux = iris.coords.AuxCoord(lon.data, standard_name=lon.name(), units=lon.units)
 
### Add latitude and longitude as coordinates. If you print cube after this it wont show up on the top line as they still aren’t dimension
mean_cube.add_aux_coord(lat_aux, data_dims=(0))
mean_cube.add_aux_coord(lon_aux, data_dims=(1))
 
### This is to promote the aux coords to dim coords where you should be able to conduct all analysis needed
iris.util.promote_aux_coord_to_dim_coord(mean_cube, 'latitude')
iris.util.promote_aux_coord_to_dim_coord(mean_cube, 'longitude')

rh_cube = regrid_model(mean_cube, regrid_modelcube)
rh_data_regridded = rh_cube.data
rh_data_regridded = rh_data_regridded*1e-3*365
rh_data_regridded = ma.masked_invalid(rh_data_regridded)
rh_data_regridded = np.ma.masked_where(rh_data_regridded<=0, rh_data_regridded)
card_rh = rh_data_regridded.copy()

npp_data_new = np.ma.masked_where(rh_data_regridded==0, npp_data)
npp_data_new = np.ma.masked_where(npp_data_new<=0, npp_data_new)


# heterotrophic respiration (original dataset)
rs_file_old = Dataset('/home/links/rmv203/obs_datasets/sr_daily_0.5degree.nc')
rs_data_old = rs_file_old.variables['soil_respiration'][:].mean(axis=0)
rs_data_old = np.flip(rs_data_old, axis=0)
rs_data_old = rs_data_old*1e-3*365


# heterotrophic respiration (new dataset)
rh_file_bad = Dataset('/home/links/rmv203/obs_datasets/RH_yr_Hashimoto2015.nc')
rh_data_bad = rh_file_bad.variables['co2'][:].mean(axis=0)
rh_data_bad = np.squeeze(rh_data_bad)
rh_data_bad = rh_data_bad*1e-3
rh_data_bad = np.ma.masked_array(rh_data_bad, mask= rh_data_bad<0.1)
rh_data_bad = np.ma.masked_array(rh_data_bad, mask= rh_data_bad>1e2)


#%% Setting up the figure

# SUBPLOT FIGURE
fig_figure1 = plt.figure(1, figsize=(48,28))
gs = gspec.GridSpec(2, 3, figure=fig_figure1, hspace=0.2)
column = 0
row = 0

mpl.rcParams['xtick.direction'] = 'out'       # set 'ticks' pointing inwards
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True             # add ticks to top and right hand axes  
mpl.rcParams['ytick.right'] = True           # of plot 

params = {
    'lines.linewidth':2,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}

plt.rcParams.update(params)


#%% Plotting for comparison

min_axis_value = 0
max_axis_value = 2

# 1
ax = fig_figure1.add_subplot(gs[row, column])

x1 = np.ma.ravel(card_rh)
y1 = np.ma.ravel(npp_data_new)

plt.plot(x1, y1, 'ko', markersize=1)

# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'r', linewidth=1)

plt.xlabel(r'CARDAMOM $R_\mathrm{h}$')
plt.ylabel(r'MODIS NPP')
plt.xlim((0,2))
plt.ylim((0,2))
plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,0.5,1,1.5,2])

# Calculating r-squared value
correlation_matrix = ma.corrcoef(x1, y1)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('1:', r_squared, correlation_matrix)
plt.text(1.5, 0.2, r'r$^2$ = %0.2f' % r_squared, fontsize=34)


# 2
column += 1
ax = fig_figure1.add_subplot(gs[row, column])

x2 = np.ma.ravel(rs_data_old)
y2 = np.ma.ravel(npp_data_new)

plt.plot(x2, y2, 'ko', markersize=1)

# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'r', linewidth=1)

plt.xlabel(r'Raich $R_\mathrm{s}$')
plt.ylabel(r'MODIS NPP')
plt.xlim((0,2))
plt.ylim((0,2))
plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,0.5,1,1.5,2])

# Calculating r-squared value
correlation_matrix = ma.corrcoef(x2, y2)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('2:', r_squared, correlation_matrix)
plt.text(1.5, 0.2, r'r$^2$ = %0.2f' % r_squared, fontsize=34)


# 3
column += 1
ax = fig_figure1.add_subplot(gs[row, column])

x3 = np.ma.ravel(rh_data_bad)
y3 = np.ma.ravel(npp_data_new)

plt.plot(x3, y3, 'ko', markersize=1)

# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'r', linewidth=1)

plt.xlabel(r'Hashimoto $R_\mathrm{h}$')
plt.ylabel(r'MODIS NPP')
plt.xlim((0,2))
plt.ylim((0,2))
plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,0.5,1,1.5,2])

# Calculating r-squared value
correlation_matrix = ma.corrcoef(x3, y3)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('3:', r_squared, correlation_matrix)
plt.text(1.5, 0.2, r'r$^2$ = %0.2f' % r_squared, fontsize=34)


# 4
column = 0
row += 1
ax = fig_figure1.add_subplot(gs[row, column])

x4 = np.ma.ravel(card_rh)
y4 = np.ma.ravel(rs_data_old)

plt.plot(x4, y4, 'ko', markersize=1)

# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'r', linewidth=1)

plt.xlabel(r'CARDAMOM $R_\mathrm{h}$')
plt.ylabel(r'Raich $R_\mathrm{s}$')
plt.xlim((0,2))
plt.ylim((0,2))
plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,0.5,1,1.5,2])

# Calculating r-squared value
correlation_matrix = ma.corrcoef(x4, y4)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('4:', r_squared, correlation_matrix)
plt.text(1.5, 0.2, r'r$^2$ = %0.2f' % r_squared, fontsize=34)


# 5
column += 1
ax = fig_figure1.add_subplot(gs[row, column])

x5 = np.ma.ravel(rh_data_bad)
y5 = np.ma.ravel(rs_data_old)

plt.plot(x5, y5, 'ko', markersize=1)

# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'r', linewidth=1)

plt.xlabel(r'Hashimoto $R_\mathrm{h}$')
plt.ylabel(r'Raich $R_\mathrm{s}$')
plt.xlim((0,2))
plt.ylim((0,2))
plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,0.5,1,1.5,2])

# Calculating r-squared value
correlation_matrix = ma.corrcoef(x5, y5)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('5:', r_squared, correlation_matrix)
plt.text(1.5, 0.2, r'r$^2$ = %0.2f' % r_squared, fontsize=34)


# 6
column += 1
ax = fig_figure1.add_subplot(gs[row, column])

x6 = np.ma.ravel(rh_data_bad)
y6 = np.ma.ravel(card_rh)

plt.plot(x6, y6, 'ko', markersize=1)

# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'r', linewidth=1)

plt.xlabel(r'Hashimoto $R_\mathrm{h}$')
plt.ylabel(r'CARDAMOM $R_\mathrm{h}$')
plt.xlim((0,2))
plt.ylim((0,2))
plt.xticks([0,0.5,1,1.5,2])
plt.yticks([0,0.5,1,1.5,2])

# Calculating r-squared value
correlation_matrix = ma.corrcoef(x6, y6)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('6:', r_squared, correlation_matrix)
plt.text(1.5, 0.2, r'r$^2$ = %0.2f' % r_squared, fontsize=34)


#%%
fig_figure1.savefig('paper_figures/SupplementaryFigure4.pdf', bbox_inches='tight')
plt.close()