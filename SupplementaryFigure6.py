#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script produces Supplementary Figure 6 in Varney et al. 2020 Nature Communications
- Observational sensitivity study to test robustness of emergent constraint to choice of heterotrophic respiration dataset;
comparison of the quadratic log(tau_s) v T fit, derived for tau_s calculated using the following observational datasets
to represent heterotrophic respiration (Rh): CARDAMOM Rh, MODIS NPP, Raich Rs
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


#%%
# observational datasets


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


# CARDAMON Rh
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
rh_data_card = np.ma.masked_where(rh_data_regridded<=0, rh_data_regridded)


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


# Observational temperature
obs_temp_file = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
obs_temp_file = select_time(obs_temp_file, 2001, 2010)
obs_temp_file = time_average(obs_temp_file)
obs_temp = obs_temp_file.data
obs_temp = np.ma.masked_where(obs_temp>60, obs_temp)
current_temp = obs_temp.copy()


#%%
# turnover time calculations

# Raich 2002 Rs
tau_s_1 = merged_hwsd_ncscd_masked / raich_rs_data
tau_s_masked_1 = ma.masked_where(np.logical_or(tau_s_1 < 1, tau_s_1 > 1e4), tau_s_1)
obs_temp_1 = ma.masked_where(np.logical_or(tau_s_1 < 1, tau_s_1 > 1e4), obs_temp)

# MODIS NPP
tau_s_2 = merged_hwsd_ncscd_masked / npp_data
tau_s_masked_2 = ma.masked_where(np.logical_or(tau_s_2 < 1, tau_s_2 > 1e4), tau_s_2)
obs_temp_2 = ma.masked_where(np.logical_or(tau_s_2 < 1, tau_s_2 > 1e4), obs_temp)

# CARDAMOM Rh
tau_s_3 = merged_hwsd_ncscd_masked / rh_data_card
tau_s_masked_3 = ma.masked_where(np.logical_or(tau_s_3 < 1, tau_s_3 > 1e4), tau_s_3)
obs_temp_3 = ma.masked_where(np.logical_or(tau_s_3 < 1, tau_s_3 > 1e4), obs_temp)

# Hashimoto Rh
tau_s_4 = merged_hwsd_ncscd_masked / rh_data_hashimoto
tau_s_masked_4 = ma.masked_where(np.logical_or(tau_s_4 < 1, tau_s_4 > 1e4), tau_s_4)
obs_temp_4 = ma.masked_where(np.logical_or(tau_s_4 < 1, tau_s_4 > 1e4), obs_temp)


#%%
# plotting

# set up the figure
fig = plt.figure(1, figsize=(16,12))
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':5,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 30,
    'xtick.labelsize':30,
    'ytick.labelsize':30,
    'font.size':30,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


# Raich 2002 Rs
x1 = np.ma.ravel(obs_temp_1)
y1 = np.ma.ravel(tau_s_masked_1)
y1 = ma.log(y1)
p_obs = np.ma.polyfit(x1, y1, 2)
poly_relationship_obs = np.poly1d(p_obs)
sorted_x1 = np.sort(x1)
plt.plot(sorted_x1, poly_relationship_obs(sorted_x1), 'b-',label=r'RAICH $R_\mathrm{s}$')

# MODIS NPP
x2 = np.ma.ravel(obs_temp_2)
y2 = np.ma.ravel(tau_s_masked_2)
y2 = ma.log(y2)
p_obs = np.ma.polyfit(x2, y2, 2)
poly_relationship_obs = np.poly1d(p_obs)
sorted_x2 = np.sort(x2)
plt.plot(sorted_x2, poly_relationship_obs(sorted_x2), 'r-',label="MODIS NPP")

# CARDAMOM Rh
x3 = np.ma.ravel(obs_temp_3)
y3 = np.ma.ravel(tau_s_masked_3)
y3 = ma.log(y3)
p_obs = np.ma.polyfit(x3, y3, 2)
poly_relationship_obs = np.poly1d(p_obs)
sorted_x3 = np.sort(x3)
plt.plot(sorted_x3, poly_relationship_obs(sorted_x3), 'g-', label=r'CARDAMOM $R_\mathrm{h}$')

# Hashimoto Rh
x4 = np.ma.ravel(obs_temp_4)
y4 = np.ma.ravel(tau_s_masked_4)
y4 = ma.log(y4)
p_obs = np.ma.polyfit(x4, y4, 2)
poly_relationship_obs = np.poly1d(p_obs)
sorted_x4 = np.sort(x4)
#plt.plot(sorted_x4, poly_relationship_obs(sorted_x4), 'm-',label=r'Hashimoto $R_\mathrm{h}$')


#
plt.legend()
plt.xlabel(r'$T$ ($^{\circ}$C)')
plt.ylabel(r'$\mathrm{log}(\tau_\mathrm{s})$')
plt.ylim((1,9))
plt.xlim((-23,30))


#%%
#  save figure
fig.savefig('paper_figures/SupplementaryFigure6_v1.pdf', bbox_inches='tight')
plt.close()
