#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis and Plotting Python Script for Varney et al. 2020 Nature Communications
- script uses relationship-derived deltaCs,tau (x-axis) and model deltaCs,tau (y-axis) calculated in 'parisagreement_cmip5_analysis'
and plots the values against one another for each CMIP5 model considered in this study
- the observational derived constraint is plotted on the x-axis
- the emergent constraint is plotted on the y-axis
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

# loading saved observational variables
poly_relationship_obs = np.poly1d(np.load('saved_variables/poly_relationship_obs.npy'))
#
observational_temperature_data = np.load('saved_variables/observational_temperature_data.npy')
observational_temperature_mask = np.load('saved_variables/observational_temperature_mask.npy')
observational_temperature = np.ma.masked_array(observational_temperature_data, mask=observational_temperature_mask)
#
observational_rh_data = np.load('saved_variables/observational_rh_data.npy')
observational_rh_mask = np.load('saved_variables/observational_rh_mask.npy')
observational_rh = np.ma.masked_array(observational_rh_data, mask=observational_rh_mask)


# CMIP5 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)
model_shapes = ['o', '^', 'v', '1', 's', '*', 'x', '+', 'd']

# RCP senarios
rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)

# global mean temperature changes
temperature_change_options = [2]
temperature_change_options_length = len(temperature_change_options)


#%%
# loop through each global mean temperature change
for temp_option in range(0, temperature_change_options_length):
    min_temperature = temperature_change_options[temp_option] # selecting the temperature change
    
    #%%
    # set up the figure
    fig = plt.figure(1, figsize=(24,18))
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 30,
    'xtick.labelsize':30,
    'ytick.labelsize':30,
    'font.size':30,
    }
    plt.rcParams.update(params)

    xmin_limit = 750
    xmax_limit = 0


    #%%
    # loading data
    x_data = np.loadtxt('saved_data/x_'+str(min_temperature)+'_degree_warming_cmip5.csv',  delimiter=',')
    y_data = np.loadtxt('saved_data/y_'+str(min_temperature)+'_degree_warming_cmip5.csv',  delimiter=',')
    obs_data = np.loadtxt('saved_data/obs_constraint_'+str(min_temperature)+'_degree_warming_cmip5.csv', delimiter=',')
    xfit = np.loadtxt("saved_data/EC_xfit_"+str(min_temperature)+"degreewarming_cmip5.csv", delimiter=',')
    yfit = np.loadtxt("saved_data/EC_yfit_"+str(min_temperature)+"degreewarming_cmip5.csv", delimiter=',')

    #%%
    # Loop through each rcp run being considered
    for rcp_option in range(0, rcp_options_length):
        rcp = rcp_options[rcp_option] # selecting the rcp scenario

        # for loop for each cmip5 model
        for model_i in range(0, n_models):
            model = cmip5_models[model_i] # seleting the models
            
            print(rcp, model)


            # plotting
            if rcp == 'rcp85':
                    plt.plot(x_data[rcp_option, model_i], y_data[rcp_option, model_i], marker=model_shapes[model_i], color='r', markersize=20, mew=5)
            elif rcp == 'rcp45':
                    plt.plot(x_data[rcp_option, model_i], y_data[rcp_option, model_i], marker=model_shapes[model_i], color='g', markersize=20, mew=5)
            elif rcp == 'rcp26':
                    plt.plot(x_data[rcp_option, model_i], y_data[rcp_option, model_i], marker=model_shapes[model_i], color='b', markersize=20, mew=5)


            # observational constraint
            obs_data_model = obs_data[model_i+(rcp_option*n_models)]
            
            
    # saving x_data and y_data
    flat_x_array = x_data.flatten()
    flat_y_array = y_data.flatten()
    flat_x_array = flat_x_array[flat_x_array==flat_x_array]
    flat_y_array = flat_y_array[flat_y_array==flat_y_array]
            
    # std of constrained values         
    x_obs = np.nanmean(obs_data)
    dx_obs = np.nanstd(obs_data)


    # creating constrained data line
    x_line = np.linspace(-xmin_limit, xmax_limit, 100)
    global_array = np.zeros([100,1])
    global_array = np.squeeze(global_array)
    for b in range(0,100):
        global_array[b] = x_obs
    plt.plot(global_array, x_line, color='darkgreen', linewidth=2, alpha=1)
        
    # plotting
    plt.axvspan(x_obs-dx_obs, x_obs+dx_obs, color='lightgreen', alpha=0.8, zorder=20)

    # calculating the constrained values
    x_values = np.loadtxt("saved_data/combined_x_"+str(min_temperature)+"_degree_warming_cmip5.csv", delimiter=",")
    y_values = np.loadtxt("saved_data/combined_y_"+str(min_temperature)+"_degree_warming_cmip5.csv", delimiter=",")
    new_xobs = np.loadtxt("saved_data/x_obs_"+str(min_temperature)+"_degree_warming_cmip5.csv", delimiter=",")
    new_dxobs = np.loadtxt("saved_data/dx_obs_"+str(min_temperature)+"_degree_warming_cmip5.csv", delimiter=",")
    # Plotting the y axis constrained values
    mean_ec_y_value, lower_ec_limit, upper_ec_limit = EC_pdf_UU_reduced(x_values, y_values, new_xobs.item(), new_dxobs.item())

    # creating constrained data line
    y_line = np.linspace(-xmin_limit, x_obs-dx_obs, 100)
    ec_array = np.zeros([100,1])
    ec_array = np.squeeze(ec_array)
    for b in range(0,100):
        ec_array[b] = mean_ec_y_value
    plt.plot(y_line, ec_array, color='b', linewidth=2, alpha=1)

    print('new mean:', ec_array[0])
    print('new std:', upper_ec_limit-ec_array[0])

    xmax = (xmin_limit+(x_obs-dx_obs))/(xmin_limit+xmax_limit)
    plt.axhspan(lower_ec_limit, upper_ec_limit, xmin=0, xmax=xmax, color='lightblue', alpha=0.8, zorder=20)
    plt.plot(xfit, yfit, color='k', linewidth=2)

    # one to one line
    one_to_one_line = np.linspace(-xmin_limit, xmax_limit, 100)
    plt.plot(one_to_one_line, one_to_one_line, 'darkgrey', linewidth=0.25)


    plt.xlim((-xmin_limit, xmax_limit))
    plt.ylim((-xmin_limit, xmax_limit))
    plt.xlabel(r'Relationship-derived $\Delta C_{\mathrm{s, \tau}}$ (PgC)')
    plt.ylabel(r'Model $\Delta C_{\mathrm{s, \tau}}$ (PgC)')


    # legends

    handels_1 = []
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='b', label='RCP2.6')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='g', label='RCP4.5')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='r', label='RCP8.5')])
    label_1 = ['RCP2.6', 'RCP4.5', 'RCP8.5']
    leg_1 = plt.legend(handels_1, label_1, loc=4, fontsize=30)
    plt.gca().add_artist(leg_1)
    
    handels = []
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='BNU-ESM')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='CanESM2')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='v', markersize=20, color='k', label='CESM1-CAM5')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='1', markersize=20, color='k', label='GFDL-ESM2G')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='GISS-E2-R')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='*', markersize=20, color='k', label='HadGEM2-ES')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='x', markersize=20, color='k', label='IPSL-CM5A-LR')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='+', markersize=20, color='k', label='MIROC-ESM')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='d', markersize=20, color='k', label='NorESM1-M')])
    handels.extend([Line2D([0,0],[0,0], linewidth=20, color='lightgreen', alpha=0.8, label='Observational Constraint')])
    handels.extend([Line2D([0,0],[0,0], linewidth=20, color='lightblue', alpha=0.8, label='Emergent Constraint')])
    label = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M', 'Observational Constraint', 'Emergent Constraint']
    leg = plt.legend(handels, label, loc=3, fontsize=30)
    plt.gca().add_artist(leg)


    #%% 
    # save figure
    fig.savefig('additional_figures/EC_cmip5_'+str(min_temperature)+'_CARDrh.pdf', bbox_inches='tight')
    plt.close()
            