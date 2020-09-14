#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script produces Figure 2 in Varney et al. 2020 Nature Communications
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
from rmv_cmip_analysis import combine_netCDF_cmip5
from rmv_cmip_analysis import combine_netCDF_observations
from rmv_cmip_analysis import combine_netCDF_time_overlap
from rmv_cmip_analysis import combine_netCDF
from rmv_cmip_analysis import combine_netCDF_merging_time_runs
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import annual_average
from rmv_cmip_analysis import decadal_average
from rmv_cmip_analysis import numpy_to_cube
from rmv_cmip_analysis import regrid_model
from rmv_cmip_analysis import area_average
from rmv_cmip_analysis import global_total
from rmv_cmip_analysis import global_total_percentage
from rmv_cmip_analysis import obtaining_fractional_deltaCs_q10

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
# Set up the subplot figure

fig_figure2 = plt.figure(1, figsize=(56,18))
gs = gspec.GridSpec(1, 2, figure=fig_figure2, hspace=5, wspace=0.5)
column = 0
row = 0
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
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
# Inputs

# historical time considered
lower_historical = 1995
upper_historical = 2005

# global region used for global averages
region_global = [0, 360, -90,  90]


#%% CMIP5

# CMIP5 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
model_colours_cmip5 = ['olive', 'gold', 'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5', '#bebada']
n_models_cmip5 = len(cmip5_models)

# RCP Scenarios
rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)


# Figure 2a
ax = fig_figure2.add_subplot(gs[row, column])

# Loop through each rcp run being considered
for rcp_option in range(0, rcp_options_length):
    rcp = rcp_options[rcp_option] # selecting the rcp scenario

    # for loop for each CMIP5 model
    for model_i in range(0, n_models_cmip5):
        model = cmip5_models[model_i] # seleting the models

        print(rcp, model)
        

        #%% modelled historical

        # Heterotrophic Respiration (RH)
        rh_historical_cube_new = combine_netCDF_cmip5('/home/rmv203/cmip5_data/rh_Lmon_'+model+'_historical*', 'heterotrophic_respiration_carbon_flux', model)
        rh_historical_cube_new = open_netCDF(rh_historical_cube_new)
        # Soil Carbon (cSoil)
        cSoil_historical_cube_new = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_historical*', 'soil_carbon_content', model)
        cSoil_historical_cube_new = open_netCDF(cSoil_historical_cube_new)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_historical*', 'air_temperature', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)
        # Select historical time period
        rh_historical_cube_new = select_time(rh_historical_cube_new, lower_historical, upper_historical)
        cSoil_historical_cube_new = select_time(cSoil_historical_cube_new, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # Time average
        rh_historical_time_av_cube_new = time_average(rh_historical_cube_new)
        cSoil_historical_time_av_cube_new = time_average(cSoil_historical_cube_new)
        tas_historical_cube = time_average(tas_historical_cube)
        tas_historical_data = tas_historical_cube.data

        # Converting from cubes to numpy_arrays
        rh_historical_time_av_data_new = rh_historical_time_av_cube_new.data
        cSoil_historical_time_av_data_new = cSoil_historical_time_av_cube_new.data
        historical_rh_save_data = rh_historical_time_av_data_new*86400.*365. # save historical rh data to use later

        # Calculating Soil Turnover Time
        tau_s_data_historical_new = cSoil_historical_time_av_data_new / (rh_historical_time_av_data_new*86400.*365.)
        tau_s_masked_data_historical_new = ma.masked_where(np.logical_or(tau_s_data_historical_new < 1, tau_s_data_historical_new > 1e4), tau_s_data_historical_new)
        historical_modelled_tau_new = tau_s_masked_data_historical_new.copy()


        #%%  Modelled Future

        # Heterotrophic Respiration (RH)
        rh_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/rh_Lmon_'+model+'_'+rcp+'_*', 'heterotrophic_respiration_carbon_flux', model)
        rh_cube = open_netCDF(rh_cube)
        # Soil Carbon (cSoil)
        cSoil_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_'+rcp+'_*', 'soil_carbon_content', model)
        cSoil_cube = open_netCDF(cSoil_cube)
        # Near Surface Air Temperature (tas)
        tas_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_'+rcp+'_*', 'air_temperature', model)
        tas_cube = open_netCDF(tas_cube)
        # Select future time period
        rh_cube = select_time(rh_cube, 2010, 2100)
        cSoil_cube = select_time(cSoil_cube, 2010, 2100)
        tas_cube = select_time(tas_cube, 2010, 2100)
        # time average
        rh_cube = annual_average(rh_cube)
        cSoil_cube = annual_average(cSoil_cube)
        tas_cube = annual_average(tas_cube)
        time_dimension = rh_cube.coord('year').points
        cube_save = cSoil_cube.copy()

        # Converting from cubes to numpy_arrays
        rh_time_av_data = rh_cube.data
        cSoil_time_av_data = cSoil_cube.data
        tas_data = tas_cube.data

        # Calculating Soil Turnover Time
        tau_s_data = cSoil_time_av_data / (rh_time_av_data*86400.*365.)
        tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
        future_modelled_tau = tau_s_masked_data.copy()


        #%% Finding the timeseries data

        # Modelled delta tau_s
        delta_tau_actual = future_modelled_tau - historical_modelled_tau_new
        # calculating delta soil carbon actual (initial rh)
        delta_c_soil_actual = delta_tau_actual*historical_rh_save_data

        # Calculating the global averaged values delta Cs
        delta_c_soil_actual = np.ma.masked_invalid(delta_c_soil_actual) # Masking invalid values
        # convert numpy array to cube
        delta_c_soil_actual_cube = numpy_to_cube(delta_c_soil_actual, cube_save, 3)
        landfraction = combine_netCDF_cmip5('/home/rmv203/cmip5_data/sftlf_fx_'+model+'_historical_r0i0p0.nc', 'land_area_fraction', model)
        actual_delta_cSoil_global = global_total_percentage(delta_c_soil_actual_cube, landfrac=landfraction, latlon_cons=None)
        actual_delta_cSoil_global_data = actual_delta_cSoil_global.data

        # global temperature change
        future_global_temperature_cube = area_average(tas_cube, region_global)
        historical_global_temp_cube = area_average(tas_historical_cube, region_global)
        future_global_temperature_data = future_global_temperature_cube.data
        historical_global_temperature_data = historical_global_temp_cube.data
        global_temperature_change = future_global_temperature_data - historical_global_temperature_data


        #%% Plotting
        if rcp == 'rcp26':
            ax.scatter(global_temperature_change, actual_delta_cSoil_global_data, color=model_colours_cmip5[model_i], marker='o', s=60)
        if rcp == 'rcp45':
            ax.scatter(global_temperature_change, actual_delta_cSoil_global_data, color=model_colours_cmip5[model_i], marker='^', s=60)
        if rcp == 'rcp85':
            ax.scatter(global_temperature_change, actual_delta_cSoil_global_data, color=model_colours_cmip5[model_i], marker='s', s=60)


#%% CMIP6

# CMIP6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
model_colours = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen']
n_models = len(cmip6_models)

# ssp Scenarios
ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)


# Loop through each ssp run being considered
for ssp_option in range(0, ssp_options_length):
    ssp = ssp_options[ssp_option] # selecting the ssp scenario

    # for loop for each CMIP5 model
    for model_i in range(0, n_models):
        model = cmip6_models[model_i] # seleting the models

        print(ssp, model)
        

        #%% modelled historical

        # Heterotrophic Respiration (RH)
        rh_historical_cube_new = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_historical*', model)
        rh_historical_cube_new = open_netCDF(rh_historical_cube_new)
        # Soil Carbon (cSoil)
        cSoil_historical_cube_new = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_historical*', model)
        cSoil_historical_cube_new = open_netCDF(cSoil_historical_cube_new)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_historical*', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)

        # Select historical time period
        rh_historical_cube_new = select_time(rh_historical_cube_new, lower_historical, upper_historical)
        cSoil_historical_cube_new = select_time(cSoil_historical_cube_new, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # Time average
        rh_historical_time_av_cube_new = time_average(rh_historical_cube_new)
        cSoil_historical_time_av_cube_new = time_average(cSoil_historical_cube_new)
        tas_historical_cube = time_average(tas_historical_cube)
        tas_historical_data = tas_historical_cube.data

        # Converting from cubes to numpy_arrays
        rh_historical_time_av_data_new = rh_historical_time_av_cube_new.data
        historical_rh_save_data = rh_historical_time_av_data_new*86400.*365. # save historical rh data to use later
        cSoil_historical_time_av_data_new = cSoil_historical_time_av_cube_new.data


        # Calculating Soil Turnover Time
        tau_s_data_historical_new = cSoil_historical_time_av_data_new / (rh_historical_time_av_data_new*86400.*365.)
        tau_s_masked_data_historical_new = ma.masked_where(np.logical_or(tau_s_data_historical_new < 1, tau_s_data_historical_new > 1e4), tau_s_data_historical_new)
        historical_modelled_tau_new = tau_s_masked_data_historical_new.copy()


        #%%  Modelled Future

        # Heterotrophic Respiration (RH)
        rh_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_'+ssp+'_*', model)
        rh_cube = open_netCDF(rh_cube)
        # Soil Carbon (cSoil)
        cSoil_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_'+ssp+'_*', model)
        cSoil_cube = open_netCDF(cSoil_cube)
        # Near Surface Air Temperature (tas)
        tas_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_'+ssp+'_*', model)
        tas_cube = open_netCDF(tas_cube)
        # Select future time period
        rh_cube = select_time(rh_cube, 2010, 2100)
        cSoil_cube = select_time(cSoil_cube, 2010, 2100)
        tas_cube = select_time(tas_cube, 2010, 2100)
        # time average
        rh_cube = annual_average(rh_cube)
        cSoil_cube = annual_average(cSoil_cube)
        tas_cube = annual_average(tas_cube)
        time_dimension = rh_cube.coord('year').points
        cube_save = cSoil_cube.copy()

        # Converting from cubes to numpy_arrays
        rh_time_av_data = rh_cube.data
        cSoil_time_av_data = cSoil_cube.data
        tas_data = tas_cube.data

        # Calculating Soil Turnover Time
        tau_s_data = cSoil_time_av_data / (rh_time_av_data*86400.*365.)
        tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
        future_modelled_tau = tau_s_masked_data.copy()


        #%% Finding the timeseries data

        # Modelled delta tau_s
        delta_tau_actual = future_modelled_tau - historical_modelled_tau_new
        # calculating delta soil carbon actual (initial rh)
        delta_c_soil_actual = delta_tau_actual*historical_rh_save_data

        # Calculating the global averaged values delta Cs 
        delta_c_soil_actual = np.ma.masked_invalid(delta_c_soil_actual) # Masking invalid values
        # convert numpy array to cube
        delta_c_soil_actual_cube = numpy_to_cube(delta_c_soil_actual, cube_save, 3)
        landfraction = combine_netCDF('/home/rmv203/cmip6_data/sftlf_fx_'+model+'_historical*', model)
        actual_delta_cSoil_global = global_total_percentage(delta_c_soil_actual_cube, landfrac=landfraction, latlon_cons=None)
        actual_delta_cSoil_global_data = actual_delta_cSoil_global.data

        # global temperature change
        future_global_temperature_cube = area_average(tas_cube, region_global)
        historical_global_temp_cube = area_average(tas_historical_cube, region_global)
        future_global_temperature_data = future_global_temperature_cube.data
        historical_global_temperature_data = historical_global_temp_cube.data
        global_temperature_change = future_global_temperature_data - historical_global_temperature_data


        #%% Plotting ###
        if ssp == 'ssp126':
            ax.scatter(global_temperature_change, actual_delta_cSoil_global_data, color=model_colours[model_i], marker='o', s=60)
        if ssp == 'ssp245':
            ax.scatter(global_temperature_change, actual_delta_cSoil_global_data, color=model_colours[model_i], marker='^', s=60)
        if ssp == 'ssp585':
            ax.scatter(global_temperature_change, actual_delta_cSoil_global_data, color=model_colours[model_i], marker='s', s=60)


ax.set_xlabel(r'$\Delta T$ ($^{\circ}$C)')
ax.set_ylabel(r'Model $\Delta C_{\mathrm{s, \tau}}$ (PgC)')
ax.set_ylim((-850,250))
ax.set_xlim((0,5))

# legend
handels = []
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkblue', label='ACCESS-ESM1-5')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='dodgerblue', label='BCC-CSM2-MR')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#80b1d3', label='CanESM5')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkcyan', label='CNRM-ESM2-1')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#8dd3c7', label='IPSL-CM6A-LR')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkseagreen', label='MIROC-ES2L')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkgreen', label='UKESM1-0-LL')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='olive', label='BNU-ESM')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='gold', label='CanESM2')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='orange', label='CESM1-CAM5')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='peachpuff', label='GFDL-ESM2G')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#fb8072', label='GISS-E2-R')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='red', label='HadGEM2-ES')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='hotpink', label='IPSL-CM5A-LR')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#fccde5', label='MIROC-ESM')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#bebada', label='NorESM1-M')])
labels = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL', 'BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
leg1 = ax.legend(handels, labels, loc='center right', borderaxespad=0.2, bbox_to_anchor=(1.345, 0.5), title='Model Colours', fontsize=34)
plt.gca().add_artist(leg1)

ax.text(-0.13, 0.9999, 'a',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


#%%
# Figure 2b
column += 1
ax = fig_figure2.add_subplot(gs[row, column])

#%% CMIP5

# Loop through each rcp run being considered
for rcp_option in range(0, rcp_options_length):
    rcp = rcp_options[rcp_option] # selecting the rcp scenario

    # for loop for each CMIP5 model
    for model_i in range(0, n_models_cmip5):
        model = cmip5_models[model_i] # seleting the models

        print(rcp, model)


        #%% modelled historical

        # Heterotrophic Respiration (RH)
        rh_historical_cube_new = combine_netCDF_cmip5('/home/rmv203/cmip5_data/rh_Lmon_'+model+'_historical*', 'heterotrophic_respiration_carbon_flux', model)
        rh_historical_cube_new = open_netCDF(rh_historical_cube_new)
        # Soil Carbon (cSoil)
        cSoil_historical_cube_new = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_historical*', 'soil_carbon_content', model)
        cSoil_historical_cube_new = open_netCDF(cSoil_historical_cube_new)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_historical*', 'air_temperature', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)
        # Select historical time period
        rh_historical_cube_new = select_time(rh_historical_cube_new, lower_historical, upper_historical)
        cSoil_historical_cube_new = select_time(cSoil_historical_cube_new, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # Time average
        rh_historical_time_av_cube_new = time_average(rh_historical_cube_new)
        cSoil_historical_time_av_cube_new = time_average(cSoil_historical_cube_new)
        tas_historical_cube = time_average(tas_historical_cube)
        tas_historical_data = tas_historical_cube.data

        # Converting from cubes to numpy_arrays
        rh_historical_time_av_data_new = rh_historical_time_av_cube_new.data
        historical_rh_save_data = rh_historical_time_av_data_new*86400.*365. # save historical rh data to use later
        cSoil_historical_time_av_data_new = cSoil_historical_time_av_cube_new.data

        # Calculating Soil Turnover Time
        tau_s_data_historical_new = cSoil_historical_time_av_data_new / (rh_historical_time_av_data_new*86400.*365.)
        tau_s_masked_data_historical_new = ma.masked_where(np.logical_or(tau_s_data_historical_new < 1, tau_s_data_historical_new > 1e4), tau_s_data_historical_new)
        historical_modelled_tau_new = tau_s_masked_data_historical_new.copy()


        #%% Modelled Future

        # Heterotrophic Respiration (RH)
        rh_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/rh_Lmon_'+model+'_'+rcp+'_*', 'heterotrophic_respiration_carbon_flux', model)
        rh_cube = open_netCDF(rh_cube)
        # Soil Carbon (cSoil)
        cSoil_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_'+rcp+'_*', 'soil_carbon_content', model)
        cSoil_cube = open_netCDF(cSoil_cube)
        # Near Surface Air Temperature (tas)
        tas_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_'+rcp+'_*', 'air_temperature', model)
        tas_cube = open_netCDF(tas_cube)
        # Select future time period
        rh_cube = select_time(rh_cube, 2010, 2100)
        cSoil_cube = select_time(cSoil_cube, 2010, 2100)
        tas_cube = select_time(tas_cube, 2010, 2100)
        # time average
        rh_cube = annual_average(rh_cube)
        cSoil_cube = annual_average(cSoil_cube)
        tas_cube = annual_average(tas_cube)
        time_dimension = rh_cube.coord('year').points
        cube_save = cSoil_cube.copy()

        # Converting from cubes to numpy_arrays
        rh_time_av_data = rh_cube.data
        cSoil_time_av_data = cSoil_cube.data
        tas_data = tas_cube.data

        # Calculating Soil Turnover Time
        tau_s_data = cSoil_time_av_data / (rh_time_av_data*86400.*365.)
        tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
        future_modelled_tau = tau_s_masked_data.copy()


        #%% Finding the timeseries data

        # Modelled delta tau_s
        delta_tau_actual = future_modelled_tau - historical_modelled_tau_new
        # calculating delta soil carbon actual (initial rh)
        delta_c_soil_actual = delta_tau_actual*historical_rh_save_data

        # Calculating the global averaged values delta Cs
        delta_c_soil_actual = np.ma.masked_invalid(delta_c_soil_actual) # Masking invalid values
        # convert numpy array to cube
        delta_c_soil_actual_cube = numpy_to_cube(delta_c_soil_actual, cube_save, 3)
        landfraction = combine_netCDF_cmip5('/home/rmv203/cmip5_data/sftlf_fx_'+model+'_historical_r0i0p0.nc', 'land_area_fraction', model)
        actual_delta_cSoil_global = global_total_percentage(delta_c_soil_actual_cube, landfrac=landfraction, latlon_cons=None)
        actual_delta_cSoil_global_data = actual_delta_cSoil_global.data

        # Calculating global totals of hitorical Soil Carbon (cSoil)
        cSoil_inital_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_historical*', 'soil_carbon_content', model)
        cSoil_inital_cube = open_netCDF(cSoil_inital_cube)
        cSoil_inital_cube = select_time(cSoil_inital_cube, 1995, 2005)
        cSoil_inital_cube = time_average(cSoil_inital_cube)
        landfraction = combine_netCDF_cmip5('/home/rmv203/cmip5_data/sftlf_fx_'+model+'_*', 'land_area_fraction', model)
        total_cSoil_inital_cube = global_total_percentage(cSoil_inital_cube, landfrac=landfraction, latlon_cons=None)
        total_cSoil_inital_data = total_cSoil_inital_cube.data

        # fractional cSoil
        fractional_deltacsoil = actual_delta_cSoil_global_data / total_cSoil_inital_data


        # global temperature change
        future_global_temperature_cube = area_average(tas_cube, region_global)
        global_temperature_historical_cube = area_average(tas_historical_cube, region_global)
        future_temperature_change = future_global_temperature_cube.data
        global_temperature_change_historical = global_temperature_historical_cube.data
        global_temperature_change = future_temperature_change - global_temperature_change_historical


        # plotting
        if rcp == 'rcp26':
            ax.scatter(global_temperature_change, fractional_deltacsoil, color=model_colours_cmip5[model_i], marker='o', s=60)
        if rcp == 'rcp45':
            ax.scatter(global_temperature_change, fractional_deltacsoil, color=model_colours_cmip5[model_i], marker='^', s=60)
        if rcp == 'rcp85':
            ax.scatter(global_temperature_change, fractional_deltacsoil, color=model_colours_cmip5[model_i], marker='s', s=60)


#%% CMIP6

# Loop through each ssp run being considered
for ssp_option in range(0, ssp_options_length):
    ssp = ssp_options[ssp_option] # selecting the ssp scenario

    # for loop for each CMIP5 model
    for model_i in range(0, n_models):
        model = cmip6_models[model_i] # seleting the models

        print(ssp, model)


        #%% modelled historical

        # Heterotrophic Respiration (RH)
        rh_historical_cube_new = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_historical*', model)
        rh_historical_cube_new = open_netCDF(rh_historical_cube_new)
        # Soil Carbon (cSoil)
        cSoil_historical_cube_new = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_historical*', model)
        cSoil_historical_cube_new = open_netCDF(cSoil_historical_cube_new)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_historical*', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)
        # Select historical time period
        rh_historical_cube_new = select_time(rh_historical_cube_new, lower_historical, upper_historical)
        cSoil_historical_cube_new = select_time(cSoil_historical_cube_new, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # time average
        rh_historical_time_av_cube_new = time_average(rh_historical_cube_new)
        cSoil_historical_time_av_cube_new = time_average(cSoil_historical_cube_new)
        tas_historical_cube = time_average(tas_historical_cube)
        tas_historical_data = tas_historical_cube.data

        # Converting from cubes to numpy_arrays
        rh_historical_time_av_data_new = rh_historical_time_av_cube_new.data
        historical_rh_save_data = rh_historical_time_av_data_new*86400.*365. # save historical rh data to use later
        cSoil_historical_time_av_data_new = cSoil_historical_time_av_cube_new.data

        # Calculating Soil Turnover Time
        tau_s_data_historical_new = cSoil_historical_time_av_data_new / (rh_historical_time_av_data_new*86400.*365.)
        tau_s_masked_data_historical_new = ma.masked_where(np.logical_or(tau_s_data_historical_new < 1, tau_s_data_historical_new > 1e4), tau_s_data_historical_new)
        historical_modelled_tau_new = tau_s_masked_data_historical_new.copy()


        #%% Modelled Future

        # Heterotrophic Respiration (RH)
        rh_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_'+ssp+'_*', model)
        rh_cube = open_netCDF(rh_cube)
        # Soil Carbon (cSoil)
        cSoil_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_'+ssp+'_*', model)
        cSoil_cube = open_netCDF(cSoil_cube)
        # Near Surface Air Temperature (tas)
        tas_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_'+ssp+'_*', model)
        tas_cube = open_netCDF(tas_cube)
        # Select future time period
        rh_cube = select_time(rh_cube, 2010, 2100)
        cSoil_cube = select_time(cSoil_cube, 2010, 2100)
        tas_cube = select_time(tas_cube, 2010, 2100)
        # time average
        rh_cube = annual_average(rh_cube)
        cSoil_cube = annual_average(cSoil_cube)
        tas_cube = annual_average(tas_cube)
        time_dimension = rh_cube.coord('year').points
        cube_save = cSoil_cube.copy()

        # Converting from cubes to numpy_arrays
        rh_time_av_data = rh_cube.data
        cSoil_time_av_data = cSoil_cube.data
        tas_data = tas_cube.data

        # Calculating Soil Turnover Time
        tau_s_data = cSoil_time_av_data / (rh_time_av_data*86400.*365.)
        tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
        future_modelled_tau = tau_s_masked_data.copy()


        #%% Finding the timeseries data

        # Modelled delta tau_s
        delta_tau_actual = future_modelled_tau - historical_modelled_tau_new
        # calculating delta soil carbon actual (initial rh)
        delta_c_soil_actual = delta_tau_actual*historical_rh_save_data

        # Calculating the global averaged values delta Cs
        delta_c_soil_actual = np.ma.masked_invalid(delta_c_soil_actual) # Masking invalid values
        # convert numpy array to cube
        delta_c_soil_actual_cube = numpy_to_cube(delta_c_soil_actual, cube_save, 3)
        landfraction = combine_netCDF('/home/rmv203/cmip6_data/sftlf_fx_'+model+'_historical*', model)
        actual_delta_cSoil_global = global_total_percentage(delta_c_soil_actual_cube, landfrac=landfraction, latlon_cons=None)
        actual_delta_cSoil_global_data = actual_delta_cSoil_global.data

        # Calculating global totals of hitorical Soil Carbon (cSoil)
        cSoil_inital_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_historical*', model)
        cSoil_inital_cube = open_netCDF(cSoil_inital_cube)
        cSoil_inital_cube = select_time(cSoil_inital_cube, 1995, 2005)
        cSoil_inital_cube = time_average(cSoil_inital_cube)
        landfraction = combine_netCDF('/home/rmv203/cmip6_data/sftlf_fx_'+model+'_historical*', model)
        total_cSoil_inital_cube = global_total_percentage(cSoil_inital_cube, landfrac=landfraction, latlon_cons=None)
        total_cSoil_inital_data = total_cSoil_inital_cube.data

        # fractional cSoil
        fractional_deltacsoil = actual_delta_cSoil_global_data / total_cSoil_inital_data


        # global temperature change
        future_global_temperature_cube = area_average(tas_cube, region_global)
        global_temperature_historical_cube = area_average(tas_historical_cube, region_global)
        future_temperature_change = future_global_temperature_cube.data
        global_temperature_change_historical = global_temperature_historical_cube.data
        global_temperature_change = future_temperature_change - global_temperature_change_historical


        if ssp == 'ssp126':
            ax.scatter(global_temperature_change, fractional_deltacsoil, color=model_colours[model_i], marker='o', s=60)
        if ssp == 'ssp245':
            ax.scatter(global_temperature_change, fractional_deltacsoil, color=model_colours[model_i], marker='^', s=60)
        if ssp == 'ssp585':
            ax.scatter(global_temperature_change, fractional_deltacsoil, color=model_colours[model_i], marker='s', s=60)


# legend
handels_2 = []
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='SSP126 / RCP2.6')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='SSP245 / RCP4.5')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='SSP585 / RCP8.5')])
label_2 = ['SSP126 / RCP2.6', 'SSP245 / RCP4.5', 'SSP585 / RCP8.5']
leg3 = ax.legend(handels_2, label_2, loc=1, fontsize=34)
plt.gca().add_artist(leg3)


#%%
# q10 comparison

new_temperature_change_options = [0, 1, 2, 3, 4, 5]
new_temperature_change_options_length = len(new_temperature_change_options)
q10_parameters = [1.5, 2, 3, 6]
temp_degree_q10s = np.zeros((len(q10_parameters), len(new_temperature_change_options)))

for q10 in range(0, len(q10_parameters)):
    q10_parameter = q10_parameters[q10]

    for new_temp_option in range(0, new_temperature_change_options_length):
        temp_change = new_temperature_change_options[new_temp_option] # selecting the temperature change
        
        if temp_change == 0:
            temp_degree_q10s[q10, new_temp_option] = 0
        else:
            temp_degree_q10s[q10, new_temp_option] = obtaining_fractional_deltaCs_q10(q10_parameter, temp_change)


q10_colours = ['lightgrey', 'darkgrey', 'dimgrey', 'k']
labels = [r'q$_{10}$=1.5', r'q$_{10}$=2', r'q$_{10}$=3', r'q$_{10}$=6']

for q10 in range(0, len(q10_parameters)):
    q10_parameter = q10_parameters[q10]
    
    ax.scatter(new_temperature_change_options, temp_degree_q10s[q10, :], color=q10_colours[q10], s=200, label=labels[q10])
    legq = ax.legend(loc=3, fontsize=34)
    # fit
    best_trend_q10 = np.ma.polyfit(new_temperature_change_options, temp_degree_q10s[q10, :], 3)
    best_trend_equation_q10 = np.poly1d(best_trend_q10)
    ax.plot(new_temperature_change_options, best_trend_equation_q10(new_temperature_change_options), linewidth=5, color=q10_colours[q10])

plt.gca().add_artist(legq)


ax.set_xlabel(r'$\Delta T$ ($^{\circ}$C)')
ax.set_ylabel(r'Model fractional change ($\Delta C_{\mathrm{s, \tau}}$/$C_{\mathrm{s,0}}$)')
ax.set_xlim((0,5))
ax.set_ylim((-0.55,0.05))


plt.text(-0.14, 0.9999, 'b',transform=ax.transAxes,va = 'top',fontweight = 'bold', fontsize=34)


#%%
fig_figure2.savefig('paper_figures/Figure2.pdf', bbox_inches='tight')
plt.close()