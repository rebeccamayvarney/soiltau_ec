#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:48:19 2020

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
from rmv_cmip_analysis import combine_netCDF_rh_cmip6
from rmv_cmip_analysis import combine_netCDF_cSoil_cmip6
from rmv_cmip_analysis import combine_netCDF_observations
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


#%%
# inputs

# cmip6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
n_models_cmip6 = len(cmip6_models)

ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)

# cmip6 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)

rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)


model_shapes = ['o', '^', 's', '*', 'x', '+', 'd', 'p', 'H', 'X', 'D', '|', '_', '>', 'v', '1']


temperature_change_options = [1, 2, 3]
temperature_change_options_length = len(temperature_change_options)


min_axis_value = -850
max_axis_value = 50


#%%

for temp_option in range(0, temperature_change_options_length):
    min_temperature = temperature_change_options[temp_option] # selecting the temperature change
    
    # Figure
    fig = plt.figure(1, figsize=(24,18))
    
    mpl.rcParams['xtick.direction'] = 'out'       # set 'ticks' pointing inwards
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.top'] = True             # add ticks to top and right hand axes  
    mpl.rcParams['ytick.right'] = True           # of plot 
    
    params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 22,
    'xtick.labelsize':22,
    'ytick.labelsize':22,
    'font.size':22,
    'text.usetex': False,
    "svg.fonttype": 'none'
    }
    
    plt.rcParams.update(params)


    #%%
    # Loop through each ssp run being considered
    for ssp_option in range(0, ssp_options_length):
        ssp = ssp_options[ssp_option] # selecting the ssp scenario
        rcp = rcp_options[ssp_option]
    
        # for loop for each cmip6 model
        for model_i in range(0, n_models):
            if model_i > 6:
                model_cmip5 = cmip5_models[model_i] # seleting the models       
                print(min_temperature, rcp, model_cmip5)

                # loading data
                if min_temperature == 0.5:
                    x_data_cmip5 = np.loadtxt('saved_data/x_05_degree_warming_cmip5.csv', delimiter=',')
                    y_data_cmip5 = np.loadtxt('saved_data/y_05_degree_warming_cmip5.csv', delimiter=',')
                    obs_data_cmip5 = np.loadtxt('saved_data/obs_constraint_05_degree_warming_cmip5.csv')
                else:
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
                
                # creating constrained data line
                x_line = np.linspace(min_axis_value, max_axis_value, 100)
                global_array = np.zeros([100,1])
                global_array = np.squeeze(global_array)
                for b in range(0,100):
                    global_array[b] = obs_data_model_cmip5
                    
                # plotting
                plt.plot(global_array, x_line, color='b', linewidth=2, alpha=0.5)

            else:
                model_cmip6 = cmip6_models[model_i] # seleting the models
                model_cmip5 = cmip5_models[model_i] # seleting the models
                
                print(min_temperature, ssp, model_cmip6)

                # loading data
                if min_temperature == 0.5:
                    x_data_cmip6 = np.loadtxt('saved_data/x_05_degree_warming_cmip6.csv', delimiter=',')
                    y_data_cmip6 = np.loadtxt('saved_data/y_05_degree_warming_cmip6.csv', delimiter=',')
                    obs_data_cmip6 = np.loadtxt('saved_data/obs_constraint_05_degree_warming_cmip6.csv')
                else:
                    x_data_cmip6 = np.loadtxt('saved_data/x_'+str(min_temperature)+'_degree_warming_cmip6.csv',  delimiter=',')
                    y_data_cmip6 = np.loadtxt('saved_data/y_'+str(min_temperature)+'_degree_warming_cmip6.csv',  delimiter=',')
                    obs_data_cmip6 = np.loadtxt('saved_data/obs_constraint_'+str(min_temperature)+'_degree_warming_cmip6.csv')
                    
                
                #%%
                # model data
                
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
                
                # creating constrained data line
                x_line = np.linspace(min_axis_value, max_axis_value, 100)
                global_array = np.zeros([100,1])
                global_array = np.squeeze(global_array)
                for b in range(0,100):
                    global_array[b] = obs_data_model_cmip6
                    
                # plotting
                plt.plot(global_array, x_line, color='b', linewidth=2, alpha=0.5)


                
                
                
                print(min_temperature, rcp, model_cmip5)

                # loading data
                if min_temperature == 0.5:
                    x_data_cmip5 = np.loadtxt('saved_data/x_05_degree_warming_cmip5.csv', delimiter=',')
                    y_data_cmip5 = np.loadtxt('saved_data/y_05_degree_warming_cmip5.csv', delimiter=',')
                    obs_data_cmip5 = np.loadtxt('saved_data/obs_constraint_05_degree_warming_cmip5.csv')
                else:
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
                
                # creating constrained data line
                x_line = np.linspace(min_axis_value, max_axis_value, 100)
                global_array = np.zeros([100,1])
                global_array = np.squeeze(global_array)
                for b in range(0,100):
                    global_array[b] = obs_data_model_cmip5
                    
                # plotting
                plt.plot(global_array, x_line, color='b', linewidth=2, alpha=0.5)
            
            
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


    ## CONCATENATE CMIP5 and CMIP6
    flat_x_array = np.concatenate((flat_x_array_cmip5, flat_x_array_cmip6), axis=0)
    flat_y_array = np.concatenate((flat_y_array_cmip5, flat_y_array_cmip6), axis=0)
    obs_data = np.concatenate((obs_data_cmip5, obs_data_cmip6), axis=0)

    old_ensemble_mean = np.nanmean(flat_y_array)
    old_ensemble_std = np.std(flat_y_array)
    print('original mean plus uncertainty:', old_ensemble_mean, old_ensemble_std)
    r_coeffient = ma.corrcoef(flat_x_array, flat_y_array)
    print('Combined CMIP r-coefficent:', r_coeffient)
    
    # std of constrained values         
    x_obs = np.nanmean(obs_data)
    dx_obs = np.nanstd(obs_data)
    
    
    # plotting
    plt.axvspan(x_obs-dx_obs, x_obs+dx_obs, color='lightblue', alpha=0.8, zorder=20)


    #%%
       
    # one to one line         
    one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100) # one to one line
    plt.plot(one_to_one_line, one_to_one_line, 'grey', linewidth=1)
    
    # legend
    handels_1 = []
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='b', label='ssp126')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='g', label='ssp245')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='r', label='ssp585')])
    label_1 = ['ssp126', 'ssp245', 'ssp585']
    
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
    
    leg1 = plt.legend(handels, label, loc=2)
    leg2 = plt.legend(handels_1, label_1, loc=4)
    
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)
    
    # axis limits
    plt.xlim((min_axis_value, max_axis_value))
    plt.ylim((min_axis_value, max_axis_value))
    
    # axis labels
    plt.xlabel(r'Estimated $\Delta C_{s, \tau}$ ($PgC$)')
    plt.ylabel(r'Modelled $\Delta C_{s, \tau}$ ($PgC$)')
    
    # # save figure
    # if min_temperature == 0.5:
    #     fig.savefig('final_plots/cmip6_parisagreement_05degreeswarming_oldSR.pdf', bbox_inches='tight')
    #     plt.close()
    # else:
    #     fig.savefig('second_reviewercomments_plots/testing_cmip6_cmip5_'+str(min_temperature)+'degreeswarming_CARDrh_Above1m.pdf', bbox_inches='tight')
    #     plt.close()
        
        
    #%%
    # saving x_obs and dx_obs values, and x and y values
    
    # converting to numpy arrays
    x_obs = np.array([x_obs])
    dx_obs = np.array([dx_obs])
    print(x_obs, 'std:', dx_obs)
    
    if min_temperature == 0.5:
        np.savetxt("saved_data/x_obs_05_degree_warming_cmip6cmip5.csv", x_obs, delimiter=",")
        np.savetxt("saved_data/dx_obs_05_degree_warming_cmip6cmip5.csv", dx_obs, delimiter=",")
        np.savetxt("saved_data/combined_x_05_degree_warming_cmip6cmip5.csv", flat_x_array, delimiter=",")
        np.savetxt("saved_data/combined_y_05_degree_warming_cmip6cmip5.csv", flat_y_array, delimiter=",")
    else:
        np.savetxt("saved_data/x_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", x_obs, delimiter=",")
        np.savetxt("saved_data/dx_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", dx_obs, delimiter=",")
        np.savetxt("saved_data/combined_x_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", flat_x_array, delimiter=",")
        np.savetxt("saved_data/combined_y_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", flat_y_array, delimiter=",")