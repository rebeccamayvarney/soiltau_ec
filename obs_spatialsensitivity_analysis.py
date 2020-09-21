#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis Python Script for Varney et al. 2020 Nature Communications
- script plots the spatial temperature sensivity of soil carbon turnover time (log(tau_s)),
calculated using observational datasets, against observational near surface air temperature (T).
- script finds a quadratic fit to represent the observational 'real world' spatial
temperature sensitivty of soil carbon turnover time.
- observational sensitivity study: define 'observational_rh' to be the observational dataset being investigated
(CARDAMOM Heterotrophic Respiration, MODIS Net Primary Production, Raich 2002 Soil Respiration, Hashimoto Heterotrophic Respiration)
- saves useful variables as numpy arrays:
observational soil carbon, observational temperature, observational heterotrophic respiration (changes depending on dataset),
and the observational quadratic fit derived from log(tau) v T plot
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
from rmv_analysis_functions import select_time
from rmv_analysis_functions import time_average
from rmv_analysis_functions import numpy_to_cube
from rmv_analysis_functions import regrid_model

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams, colors
from matplotlib import gridspec as gspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpat
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


#%%
# regrid cube
regrid_cube = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
regrid_cube = time_average(regrid_cube)
regrid_cube.coord('latitude').guess_bounds()
regrid_cube.coord('longitude').guess_bounds()
regrid_modelcube = regrid_cube.copy()


#%%
# Observational datasets

# Observational temperature
obs_temp_file = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
obs_temp_file = select_time(obs_temp_file, 2001, 2010)
obs_temp_file = time_average(obs_temp_file)
obs_temp = obs_temp_file.data
obs_temp = np.ma.masked_where(obs_temp>60, obs_temp)
current_temp = obs_temp.copy()

# Observational soil carbon
# ncscd
ncscd_file = Dataset('/home/links/rmv203/obs_datasets/NCSCDV22_soilc_0.5x0.5.nc')
ncscd_data = ncscd_file.variables['soilc'][:]
n_lat_ncscd = ncscd_file.variables['lat'][:]
n_lon_ncscd = ncscd_file.variables['lon'][:]
# hwsd
hwsd_file = Dataset('/home/links/rmv203/obs_datasets/HWSD_soilc_0.5x0.5.nc')
hwsd_data = hwsd_file.variables['soilc'][:]
# merging the soil carbon observational datasets
merged_hwsd_ncscd = np.copy(hwsd_data)
merged_hwsd_ncscd[ncscd_data[:] > 0.] = ncscd_data[ncscd_data[:] > 0.]
merged_hwsd_ncscd_masked = ma.masked_where(np.logical_or(merged_hwsd_ncscd < 0, merged_hwsd_ncscd > 998), merged_hwsd_ncscd)
observational_cSoil = merged_hwsd_ncscd_masked.copy()


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
# regridding
rh_cube = regrid_model(mean_cube, regrid_modelcube)
rh_data_regridded = rh_cube.data
rh_data_regridded = rh_data_regridded*1e-3*365
rh_data_regridded = ma.masked_invalid(rh_data_regridded)
rh_data_regridded = np.ma.masked_where(rh_data_regridded<=0, rh_data_regridded)

# MODIS Net Primary Production (NPP)
npp_file = Dataset('/home/links/rmv203/obs_datasets/MOD17A3_Science_NPP_mean_00_14_regridhalfdegree.nc')
npp_data = npp_file.variables['npp'][:]*1e-3
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
# Observational sensitivity study:

observational_rh = rh_data_regridded.copy() # CARDAMOM Heterotrophic Respiration (Rh)
# observational_rh = npp_data.copy() # MODIS Net Primary Production (NPP)
# observational_rh = raich_rs_data.copy() # Raich 2002 Soil Respiration (Rs)
# observational_rh = rh_data_hashimoto.copy() # Hashimoto Heterotrophic Respiration (Rh)


#%%
# calculation of soil carbon turnover time

tau_s = observational_cSoil / observational_rh
tau_s_masked = ma.masked_where(np.logical_or(tau_s < 1, tau_s > 1e4), tau_s)


#%%
# plotting spatial log(tau_s) against T

fig = plt.figure(1, figsize=(16,12))
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':2,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 32,
    'xtick.labelsize':32,
    'ytick.labelsize':32,
    'font.size':26,
}
plt.rcParams.update(params)


# x
obs_temp = ma.masked_where(np.logical_or(tau_s < 1, tau_s > 1e4), obs_temp)
x = np.ma.ravel(obs_temp)
# y
y_nonlog = np.ma.ravel(tau_s_masked)
y = ma.log(y_nonlog)

plt.plot(x, y, 'ko', markersize=1)

p_obs = np.ma.polyfit(x, y, 2)
poly_relationship_obs = np.poly1d(p_obs)
sorted_x = np.sort(x)
plt.plot(sorted_x, poly_relationship_obs(sorted_x), 'b-',label="Quadratic fit")
plt.legend()


plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.ylabel(r'log($\tau_{s}$) (log(yrs))')
plt.ylim((1,9))
plt.xlim((-23,30))


#%%
# save figure
fig.savefig('additional_figures/observational_spatial_quadratic_CARDrh', bbox_inches='tight')
plt.close()


#%%
# Saving useful data
np.save('saved_variables/observational_rh_data.npy', observational_rh.data)
np.save('saved_variables/observational_rh_mask.npy', observational_rh.mask)
np.save('saved_variables/observational_temperature_data.npy', current_temp.data)
np.save('saved_variables/observational_temperature_mask.npy', current_temp.mask)
np.save('saved_variables/observational_Cs_data.npy', merged_hwsd_ncscd_masked.data)
np.save('saved_variables/observational_Cs_mask.npy', merged_hwsd_ncscd_masked.mask)
np.save('saved_variables/poly_relationship_obs.npy', poly_relationship_obs)

