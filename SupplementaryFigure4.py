#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script produces Supplementary Figure 4 in Varney et al. 2020 Nature Communications
- Observational sensitivity study to test robustness of emergent constraint to choice of heterotrophic respiration dataset;
one-to-one comparisons of observational datasets: CARDAMOM Rh, MODIS NPP, Raich Rs, Hashimoto Rh
"""

#%%

# Analysis imports
import numpy as np
import numpy.ma as ma
import csv
import netCDF4
from netCDF4 import Dataset
import iris
import iris.coord_categorisation
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes

# My functions
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import regrid_model

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


#%%
# cube for regridding

regrid_cube = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
regrid_cube = time_average(regrid_cube)
regrid_cube.coord('latitude').guess_bounds()
regrid_cube.coord('longitude').guess_bounds()
regrid_modelcube = regrid_cube.copy()
# correct lat and lon dimensions
n_lat = regrid_cube.coord('latitude').points
n_lon = regrid_cube.coord('longitude').points


#%%
# observational datasets

# CARDAMOM Heterotrophic Respiration (Rh)
cubes = iris.load('/home/links/rmv203/obs_datasets/CARDAMOM_2001_2010_FL_RHE.nc')
for cube in cubes:
    if cube.var_name == 'longitude':
        lon = cube
    if cube.var_name == 'latitude':
        lat = cube
    if cube.var_name == 'Mean':
        mean_cube = cube
# Takes the latitude and longitude ‘cubes’ and makes them in to coordinates
lat_aux = iris.coords.AuxCoord(lat.data, standard_name=lat.name(), units=lat.units)
lon_aux = iris.coords.AuxCoord(lon.data, standard_name=lon.name(), units=lon.units)
# Add latitude and longitude as coordinates
mean_cube.add_aux_coord(lat_aux, data_dims=(0))
mean_cube.add_aux_coord(lon_aux, data_dims=(1))
iris.util.promote_aux_coord_to_dim_coord(mean_cube, 'latitude')
iris.util.promote_aux_coord_to_dim_coord(mean_cube, 'longitude')
# regrid cube
rh_cube = regrid_model(mean_cube, regrid_modelcube)
rh_data_regridded = rh_cube.data
rh_data_regridded = rh_data_regridded*1e-3*365
rh_data_regridded = ma.masked_invalid(rh_data_regridded)
rh_data_regridded = np.ma.masked_where(rh_data_regridded<=0, rh_data_regridded)
card_rh = rh_data_regridded.copy()


# MODIS Net Primary Production (NPP)
npp_file = Dataset('/home/links/rmv203/obs_datasets/MOD17A3_Science_NPP_mean_00_14_regridhalfdegree.nc')
npp_data_new = npp_file.variables['npp'][:]*1e-3
#npp_data_new = np.ma.masked_where(rh_data_regridded==0, npp_data)
#npp_data_new = np.ma.masked_where(npp_data_new<=0, npp_data_new)


# Raich 2002 Soil Respiration (Rs)
raich_rs_file = Dataset('/home/links/rmv203/obs_datasets/sr_daily_0.5degree.nc')
raich_rs_data = raich_rs_file.variables['soil_respiration'][:].mean(axis=0)
raich_rs_data = np.flip(raich_rs_data, axis=0)
raich_rs_data = raich_rs_data*1e-3*365


# Hashimoto 2015 Heterotrophic Respiration (Rh)
rh_file_hashimoto = Dataset('/home/links/rmv203/obs_datasets/RH_yr_Hashimoto2015.nc')
rh_data_hashimoto = rh_file_hashimoto.variables['co2'][:].mean(axis=0)
rh_data_hashimoto = np.squeeze(rh_data_hashimoto)
rh_data_hashimoto = rh_data_hashimoto*1e-3
rh_data_hashimoto = np.ma.masked_array(rh_data_hashimoto, mask= rh_data_hashimoto<0.1)
rh_data_hashimoto = np.ma.masked_array(rh_data_hashimoto, mask= rh_data_hashimoto>1e2)


#%%
# Setting up the figure

fig_figure1 = plt.figure(1, figsize=(48,28))
gs = gspec.GridSpec(2, 3, figure=fig_figure1, hspace=0.2)
column = 0
row = 0
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
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


#%%
# plotting

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

x2 = np.ma.ravel(raich_rs_data)
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

x3 = np.ma.ravel(rh_data_hashimoto)
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
y4 = np.ma.ravel(raich_rs_data)
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

x5 = np.ma.ravel(rh_data_hashimoto)
y5 = np.ma.ravel(raich_rs_data)
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

x6 = np.ma.ravel(rh_data_hashimoto)
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
fig_figure1.savefig('paper_figures/SupplementaryFigure4_v1.pdf', bbox_inches='tight')
plt.close()