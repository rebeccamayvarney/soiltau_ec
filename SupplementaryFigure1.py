#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script that calculates and plots a timeseries of total soil carbon change (delta cSoil)
against temperature change (delta T) for a selection of CMIP5 and CMIP6 models,
for different RCP / SSP future scenarios.
"""

#%%

# Analysis imports
import numpy as np
import numpy.ma as ma
import csv
import iris
import iris.coord_categorisation
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes

# My functions
from rmv_cmip_analysis import combine_netCDF_cmip5
from rmv_cmip_analysis import combine_netCDF_time_overlap
from rmv_cmip_analysis import combine_netCDF
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import annual_average
from rmv_cmip_analysis import numpy_to_cube
from rmv_cmip_analysis import area_average
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


#%% Set up figure

fig_figure1 = plt.figure(1, figsize=(24,18))
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


#%% Inputs

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


# Loop through each rcp run being considered
for rcp_option in range(0, rcp_options_length):
    rcp = rcp_options[rcp_option] # selecting the rcp scenario

    # for loop for each CMIP5 model
    for model_i in range(0, n_models_cmip5):
        model = cmip5_models[model_i] # seleting the models

        print(rcp, model)
        

        #%% modelled historical

        # Soil Carbon (cSoil)
        cSoil_historical_cube_new = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_historical*', 'soil_carbon_content', model)
        cSoil_historical_cube_new = open_netCDF(cSoil_historical_cube_new)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_historical*', 'air_temperature', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)
        # Select historical time period
        cSoil_historical_cube_new = select_time(cSoil_historical_cube_new, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # Time average
        cSoil_historical_time_av_cube_new = time_average(cSoil_historical_cube_new)
        tas_historical_cube = time_average(tas_historical_cube)
        tas_historical_data = tas_historical_cube.data

        # Converting from cubes to numpy_arrays
        cSoil_historical_time_av_data_new = cSoil_historical_time_av_cube_new.data


        #%%  Modelled Future

        # Soil Carbon (cSoil)
        cSoil_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_'+rcp+'_*', 'soil_carbon_content', model)
        cSoil_cube = open_netCDF(cSoil_cube)
        # Near Surface Air Temperature (tas)
        tas_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_'+rcp+'_*', 'air_temperature', model)
        tas_cube = open_netCDF(tas_cube)
        # Select future time period
        cSoil_cube = select_time(cSoil_cube, 2010, 2100)
        tas_cube = select_time(tas_cube, 2010, 2100)
        # annual average
        cSoil_cube = annual_average(cSoil_cube)
        tas_cube = annual_average(tas_cube)
        time_dimension = cSoil_cube.coord('year').points
        cube_save = cSoil_cube.copy()

        # Converting from cubes to numpy_arrays
        cSoil_time_av_data = cSoil_cube.data
        tas_data = tas_cube.data


        #%% Finding the timeseries data

        # Modelled delta tau_s
        delta_cSoil_actual_cmip5 = cSoil_time_av_data - cSoil_historical_time_av_data_new

        # Calculating the global averaged values delta Cs
        delta_cSoil_actual_cmip5 = np.ma.masked_invalid(delta_cSoil_actual_cmip5) # Masking invalid values
        # convert numpy array to cube
        delta_cSoil_actual_cmip5_cube = numpy_to_cube(delta_cSoil_actual_cmip5, cube_save, 3)
        landfraction = combine_netCDF_cmip5('/home/rmv203/cmip5_data/sftlf_fx_'+model+'_historical_r0i0p0.nc', 'land_area_fraction', model)
        actual_delta_cSoil_global_cmip5 = global_total_percentage(delta_cSoil_actual_cmip5_cube, landfrac=landfraction, latlon_cons=None)
        actual_delta_cSoil_global_cmip5 = actual_delta_cSoil_global_cmip5.data

        # global temperature change
        future_global_temperature_cube = area_average(tas_cube, region_global)
        historical_global_temp_cube = area_average(tas_historical_cube, region_global)
        future_global_temperature_data = future_global_temperature_cube.data
        historical_global_temperature_data = historical_global_temp_cube.data
        global_temperature_change = future_global_temperature_data - historical_global_temperature_data


        #%% Plotting
        if rcp == 'rcp26':
            plt.scatter(global_temperature_change, actual_delta_cSoil_global_cmip5, color=model_colours_cmip5[model_i], marker='o', s=60)
        if rcp == 'rcp45':
            plt.scatter(global_temperature_change, actual_delta_cSoil_global_cmip5, color=model_colours_cmip5[model_i], marker='^', s=60)
        if rcp == 'rcp85':
            plt.scatter(global_temperature_change, actual_delta_cSoil_global_cmip5, color=model_colours_cmip5[model_i], marker='s', s=60)


#%% CMIP6

# CMIP6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
model_colours = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen']
n_models = len(cmip6_models)

# ssp scenarios
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

        # Soil Carbon (cSoil)
        cSoil_historical_cube_new = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_historical*', model)
        cSoil_historical_cube_new = open_netCDF(cSoil_historical_cube_new)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_historical*', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)
        # Select historical time period
        cSoil_historical_cube_new = select_time(cSoil_historical_cube_new, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # Time average
        cSoil_historical_time_av_cube_new = time_average(cSoil_historical_cube_new)
        tas_historical_cube = time_average(tas_historical_cube)
        tas_historical_data = tas_historical_cube.data
        # Converting from cubes to numpy_arrays
        cSoil_historical_time_av_data_new = cSoil_historical_time_av_cube_new.data
        historical_modelled_cSoil_new = cSoil_historical_time_av_data_new.copy()


        #%%  Modelled Future

        # Soil Carbon (cSoil)
        cSoil_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_'+ssp+'_*', model)
        cSoil_cube = open_netCDF(cSoil_cube)
        # Near Surface Air Temperature (tas)
        tas_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_'+ssp+'_*', model)
        tas_cube = open_netCDF(tas_cube)
        # Select future time period
        cSoil_cube = select_time(cSoil_cube, 2010, 2100)
        tas_cube = select_time(tas_cube, 2010, 2100)
        # annual average
        cSoil_cube = annual_average(cSoil_cube)
        tas_cube = annual_average(tas_cube)
        time_dimension = cSoil_cube.coord('year').points
        cube_save = cSoil_cube.copy()

        # Converting from cubes to numpy_arrays
        cSoil_time_av_data = cSoil_cube.data
        future_modelled_cSoil = cSoil_time_av_data.copy()
        tas_data = tas_cube.data


        #%% Finding the timeseries data

        # Modelled delta cSoil
        delta_cSoil_actual_cmip6 = future_modelled_cSoil - historical_modelled_cSoil_new

        # Calculating the global averaged values delta Cs
        delta_cSoil_actual_cmip6 = np.ma.masked_invalid(delta_cSoil_actual_cmip6) # Masking invalid values
        # convert numpy array to cube
        delta_cSoil_actual_cmip6_cube = numpy_to_cube(delta_cSoil_actual_cmip6, cube_save, 3)
        landfraction = combine_netCDF('/home/rmv203/cmip6_data/sftlf_fx_'+model+'_historical*', model)
        actual_delta_cSoil_global_cmip6_cube = global_total_percentage(delta_cSoil_actual_cmip6_cube, landfrac=landfraction, latlon_cons=None)
        actual_delta_cSoil_global_cmip6_data = actual_delta_cSoil_global_cmip6_cube.data

        # global temperature change
        future_global_temperature_cube = area_average(tas_cube, region_global)
        historical_global_temp_cube = area_average(tas_historical_cube, region_global)
        future_global_temperature_data = future_global_temperature_cube.data
        historical_global_temperature_data = historical_global_temp_cube.data
        global_temperature_change = future_global_temperature_data - historical_global_temperature_data


        #%% Plotting
        if ssp == 'ssp126':
            plt.scatter(global_temperature_change, actual_delta_cSoil_global_cmip6_data, color=model_colours[model_i], marker='o', s=60)
        if ssp == 'ssp245':
            plt.scatter(global_temperature_change, actual_delta_cSoil_global_cmip6_data, color=model_colours[model_i], marker='^', s=60)
        if ssp == 'ssp585':
            plt.scatter(global_temperature_change, actual_delta_cSoil_global_cmip6_data, color=model_colours[model_i], marker='s', s=60)


plt.xlabel(r'$\Delta T$ ($^{\circ}$C)')
plt.ylabel(r'Model $\Delta C_{\mathrm{s}}$ (PgC)')
plt.ylim((-200, 250))
plt.xlim((0,5))


# legends
handels_2 = []
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='SSP126 / RCP2.6')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='SSP245 / RCP4.5')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='SSP585 / RCP8.5')])
label_2 = ['SSP126 / RCP2.6', 'SSP245 / RCP4.5', 'SSP585 / RCP8.5']
leg3 = plt.legend(handels_2, label_2, loc=4, fontsize=34)
plt.gca().add_artist(leg3)

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
leg1 = plt.legend(handels, labels, loc='center right', borderaxespad=0.2, bbox_to_anchor=(1.325, 0.5), title='Model Colours', fontsize=34)
plt.gca().add_artist(leg1)


#%%
fig_figure1.savefig('paper_figures/SupplementaryFigure1_v1.pdf', bbox_inches='tight')
plt.close()
