#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis and Plotting Python Script for Varney et al. 2020 Nature Communications
- script uses relationship-derived deltaCs,tau (x-axis) and model deltaCs,tau (y-axis) calculated in 'parisagreement_cmip5_analysis'
and 'parisagreement_cmip6_analysis', and plots the values against one another for each model considered in this study
- script combines the data to consider models from CMIP6 and CMIP5 as one model ensemble
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
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


#%%

# CMIP6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
n_models_cmip6 = len(cmip6_models)
# SSP senarios
ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)

# CMIP5 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)
# RCP senarios
rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)

model_shapes = ['o', '^', 's', '*', 'x', '+', 'd', 'p', 'H', 'X', 'D', '|', '_', '>', 'v', '1']


# global mean temperature change
temperature_change_options = [1, 2, 3]
temperature_change_options_length = len(temperature_change_options)


#%%
# loop through each global mean temperature change
for temp_option in range(0, temperature_change_options_length):
    min_temperature = temperature_change_options[temp_option] # selecting the temperature change
    

    # set up figure for each temperature change
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
    'axes.labelsize': 22,
    'xtick.labelsize':22,
    'ytick.labelsize':22,
    'font.size':22,
    }
    plt.rcParams.update(params)


    min_axis_value = -850
    max_axis_value = 50


    #%%
    # Loop through each ssp run being considered
    for ssp_option in range(0, ssp_options_length):
        ssp = ssp_options[ssp_option] # selecting the SSP scenario
        rcp = rcp_options[ssp_option] # selecting the RCP scenario
    
        # for loop for each cmip6 model
        for model_i in range(0, n_models):
            if model_i > 6:
                model_cmip5 = cmip5_models[model_i] # seleting the CMIP5 model     
                print(min_temperature, rcp, model_cmip5)

                # loading data
                x_data_cmip5 = np.loadtxt('saved_data/x_'+str(min_temperature)+'_degree_warming_cmip5.csv',  delimiter=',')
                y_data_cmip5 = np.loadtxt('saved_data/y_'+str(min_temperature)+'_degree_warming_cmip5.csv',  delimiter=',')
                obs_data_cmip5 = np.loadtxt('saved_data/obs_constraint_'+str(min_temperature)+'_degree_warming_cmip5.csv')
                    
                
                #%%
                # model data
                rcp_option = ssp_option
                cmip5_modelshape = model_i+7
                # plotting
                if rcp == 'rcp85':
                        plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker=model_shapes[cmip5_modelshape], color='r', markersize=20, mew=5)
                elif rcp == 'rcp45':
                        plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker=model_shapes[cmip5_modelshape], color='g', markersize=20, mew=5)
                elif rcp == 'rcp26':
                        plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker=model_shapes[cmip5_modelshape], color='b', markersize=20, mew=5)


                #%%
                # observational constraint
                
                obs_data_model_cmip5 = obs_data_cmip5[model_i+(rcp_option*n_models)]
                
                # plotting constrained data line
                x_line = np.linspace(min_axis_value, max_axis_value, 100)
                global_array = np.zeros([100,1])
                global_array = np.squeeze(global_array)
                for b in range(0,100):
                    global_array[b] = obs_data_model_cmip5
                plt.plot(global_array, x_line, color='b', linewidth=2, alpha=0.5)


            else:

                #%% CMIP6
                model_cmip6 = cmip6_models[model_i] # seleting the models
                print(min_temperature, ssp, model_cmip6)

                # loading data
                x_data_cmip6 = np.loadtxt('saved_data/x_'+str(min_temperature)+'_degree_warming_cmip6.csv',  delimiter=',')
                y_data_cmip6 = np.loadtxt('saved_data/y_'+str(min_temperature)+'_degree_warming_cmip6.csv',  delimiter=',')
                obs_data_cmip6 = np.loadtxt('saved_data/obs_constraint_'+str(min_temperature)+'_degree_warming_cmip6.csv')
                    
                # plotting
                if ssp == 'ssp585':
                        plt.plot(x_data_cmip6[ssp_option, model_i], y_data_cmip6[ssp_option, model_i], marker=model_shapes[model_i], color='r', markersize=20, mew=5)
                elif ssp == 'ssp245':
                        plt.plot(x_data_cmip6[ssp_option, model_i], y_data_cmip6[ssp_option, model_i], marker=model_shapes[model_i], color='g', markersize=20, mew=5)
                elif ssp == 'ssp126':
                        plt.plot(x_data_cmip6[ssp_option, model_i], y_data_cmip6[ssp_option, model_i], marker=model_shapes[model_i], color='b', markersize=20, mew=5)


                #%%
                # observational constraint
                obs_data_model_cmip6 = obs_data_cmip6[model_i+(ssp_option*n_models_cmip6)]
                
                # plotting constrained data line
                x_line = np.linspace(min_axis_value, max_axis_value, 100)
                global_array = np.zeros([100,1])
                global_array = np.squeeze(global_array)
                for b in range(0,100):
                    global_array[b] = obs_data_model_cmip6
                plt.plot(global_array, x_line, color='b', linewidth=2, alpha=0.5)


                #%% CMIP5
                model_cmip5 = cmip5_models[model_i] # seleting the models
                print(min_temperature, ssp, model_cmip5)

                # loading data
                x_data_cmip5 = np.loadtxt('saved_data/x_'+str(min_temperature)+'_degree_warming_cmip5.csv',  delimiter=',')
                y_data_cmip5 = np.loadtxt('saved_data/y_'+str(min_temperature)+'_degree_warming_cmip5.csv',  delimiter=',')
                obs_data_cmip5 = np.loadtxt('saved_data/obs_constraint_'+str(min_temperature)+'_degree_warming_cmip5.csv')
                    
                # plotting
                rcp_option = ssp_option
                cmip5_modelshape = model_i+7
                if rcp == 'rcp85':
                        plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker=model_shapes[cmip5_modelshape], color='r', markersize=20, mew=5)
                elif rcp == 'rcp45':
                        plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker=model_shapes[cmip5_modelshape], color='g', markersize=20, mew=5)
                elif rcp == 'rcp26':
                        plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker=model_shapes[cmip5_modelshape], color='b', markersize=20, mew=5)


                #%%
                # observational constraint
                obs_data_model_cmip5 = obs_data_cmip5[model_i+(rcp_option*n_models)]
                
                # creating constrained data line
                x_line = np.linspace(min_axis_value, max_axis_value, 100)
                global_array = np.zeros([100,1])
                global_array = np.squeeze(global_array)
                for b in range(0,100):
                    global_array[b] = obs_data_model_cmip5
                plt.plot(global_array, x_line, color='b', linewidth=2, alpha=0.5)
            
    
    #%%
    # combining CMIP5 and CMIP6 models

    # saving x_data and y_data for CMIP5
    flat_x_array_cmip5 = x_data_cmip5.flatten()
    flat_y_array_cmip5 = y_data_cmip5.flatten()
    flat_x_array_cmip5 = flat_x_array_cmip5[flat_x_array_cmip5==flat_x_array_cmip5]
    flat_y_array_cmip5 = flat_y_array_cmip5[flat_y_array_cmip5==flat_y_array_cmip5]

    # saving x_data and y_data for CMIP6
    flat_x_array_cmip6 = x_data_cmip6.flatten()
    flat_y_array_cmip6 = y_data_cmip6.flatten()
    flat_x_array_cmip6 = flat_x_array_cmip6[flat_x_array_cmip6==flat_x_array_cmip6]
    flat_y_array_cmip6 = flat_y_array_cmip6[flat_y_array_cmip6==flat_y_array_cmip6]

    # CONCATENATE CMIP5 and CMIP6
    flat_x_array = np.concatenate((flat_x_array_cmip5, flat_x_array_cmip6), axis=0)
    flat_y_array = np.concatenate((flat_y_array_cmip5, flat_y_array_cmip6), axis=0)
    obs_data = np.concatenate((obs_data_cmip5, obs_data_cmip6), axis=0)


    #%%
    # unconstrained values
    old_ensemble_mean = np.nanmean(flat_y_array)
    old_ensemble_std = np.std(flat_y_array)
    print('original mean plus uncertainty:', old_ensemble_mean, old_ensemble_std)
    r_coeffient = ma.corrcoef(flat_x_array, flat_y_array)
    print('Combined CMIP r-coefficent:', r_coeffient)
    

    #%%
    # observational constraint       
    x_obs = np.nanmean(obs_data)
    dx_obs = np.nanstd(obs_data)
    plt.axvspan(x_obs-dx_obs, x_obs+dx_obs, color='lightblue', alpha=0.8, zorder=20)


    # one to one line         
    one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100) # one to one line
    plt.plot(one_to_one_line, one_to_one_line, 'grey', linewidth=1)
    

    # legends

    handels_1 = []
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='b', label='ssp126')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='g', label='ssp245')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='r', label='ssp585')])
    label_1 = ['ssp126', 'ssp245', 'ssp585']
    leg_1 = plt.legend(handels_1, label_1, loc=4)
    plt.gca().add_artist(leg_1)
    
    handels = []
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='ACCESS-ESM1-5')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='BCC-CSM2-MR')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='CanESM5')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='*', markersize=20, color='k', label='CNRM-ESM2-1')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='x', markersize=20, color='k', label='IPSL-CM6A-LR')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='+', markersize=20, color='k', label='MIROC-ES2L')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='d', markersize=20, color='k', label='UKESM1-0-LL')])
    handels.extend([Line2D([0,0],[0,0], linewidth=1, color='grey', label='one to one line')])
    handels.extend([Line2D([0,0],[0,0], linewidth=10, color='b', alpha=0.5, label='Observational-derived mean')])
    handels.extend([Line2D([0,0],[0,0], linewidth=20, color='lightblue', alpha=0.8, label='CMIP6 Standard Deviation')])
    label = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL', 'one to one line', 'Observational-derived mean', 'CMIP6 Standard Deviation']
    leg = plt.legend(handels, label, loc=2)
    plt.gca().add_artist(leg)
    

    # axis limits
    plt.xlim((min_axis_value, max_axis_value))
    plt.ylim((min_axis_value, max_axis_value))
    # axis labels
    plt.xlabel(r'Estimated $\Delta C_{\mathrm{s, \tau}}$ (PgC)')
    plt.ylabel(r'Model $\Delta C_{\mathrm{s, \tau}}$ (PgC)')
    

    #%%
    fig.savefig('additional_figures/obs_constraint_cmip6cmip5_'+str(min_temperature)+'_CARDrh.pdf', bbox_inches='tight')
    plt.close()
        
        
    #%%
    # saving x_obs and dx_obs values, and x and y values
    x_obs = np.array([x_obs])
    dx_obs = np.array([dx_obs])
    np.savetxt("saved_data/x_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", x_obs, delimiter=",")
    np.savetxt("saved_data/dx_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", dx_obs, delimiter=",")
    np.savetxt("saved_data/combined_x_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", flat_x_array, delimiter=",")
    np.savetxt("saved_data/combined_y_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", flat_y_array, delimiter=",")
