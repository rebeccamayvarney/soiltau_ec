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
from rmv_cmip_analysis import EC_pdf_UU_reduced

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

# observational data
poly_relationship_obs = np.poly1d(np.load('saved_variables/poly_relationship_obs.npy'))
observational_temperature_data = np.load('saved_variables/observational_temperature_data.npy')
observational_temperature_mask = np.load('saved_variables/observational_temperature_mask.npy')
observational_temperature = np.ma.masked_array(observational_temperature_data, mask=observational_temperature_mask)
observational_rh_data = np.load('saved_variables/observational_rh_data.npy')
observational_rh_mask = np.load('saved_variables/observational_rh_mask.npy')
observational_rh = np.ma.masked_array(observational_rh_data, mask=observational_rh_mask)


# cmip6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'NorESM2-LM', 'UKESM1-0-LL']
n_models = len(cmip6_models)
model_shapes = ['o', '^', 'v', '1', 's', '*', 'x', '+', 'd']


ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)

temperature_change_options = [2]
temperature_change_options_length = len(temperature_change_options)


xmin_limit = 750
xmax_limit = 0


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
    'axes.labelsize': 30,
    'xtick.labelsize':30,
    'ytick.labelsize':30,
    'font.size':30,
    'text.usetex': False,
    "svg.fonttype": 'none'
    }
    
    plt.rcParams.update(params)




    # loading data
    x_data = np.loadtxt('saved_data/x_'+str(min_temperature)+'_degree_warming_cmip6.csv',  delimiter=',')
    y_data = np.loadtxt('saved_data/y_'+str(min_temperature)+'_degree_warming_cmip6.csv',  delimiter=',')
    obs_data = np.loadtxt('saved_data/obs_constraint_'+str(min_temperature)+'_degree_warming_cmip6.csv', delimiter=',')

    xfit = np.loadtxt("saved_data/EC_xfit_"+str(min_temperature)+"degreewarming_cmip6.csv", delimiter=',')
    yfit = np.loadtxt("saved_data/EC_yfit_"+str(min_temperature)+"degreewarming_cmip6.csv", delimiter=',')


    # Loop through each ssp run being considered
    for ssp_option in range(0, ssp_options_length):
        ssp = ssp_options[ssp_option] # selecting the ssp scenario

        # for loop for each cmip6 model
        for model_i in range(0, n_models):
            model = cmip6_models[model_i] # seleting the models
            
            print(ssp, model)

            # plotting
            if ssp == 'ssp585':
                    plt.plot(x_data[ssp_option, model_i], y_data[ssp_option, model_i], marker=model_shapes[model_i], color='r', markersize=20, mew=5)
            elif ssp == 'ssp245':
                    plt.plot(x_data[ssp_option, model_i], y_data[ssp_option, model_i], marker=model_shapes[model_i], color='g', markersize=20, mew=5)
            elif ssp == 'ssp126':
                    plt.plot(x_data[ssp_option, model_i], y_data[ssp_option, model_i], marker=model_shapes[model_i], color='b', markersize=20, mew=5)


            # observational constraint
            obs_data_model = obs_data[model_i+(ssp_option*n_models)]
            
            
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
    # plotting
    plt.plot(global_array, x_line, color='darkgreen', linewidth=2, alpha=1)
        
    # plotting
    plt.axvspan(x_obs-dx_obs, x_obs+dx_obs, color='lightgreen', alpha=0.8, zorder=20)
    # ax.axvspan(obs_min, obs_max, color='lightblue', label='cmip6 Range', alpha=0.2, zorder=20)

    # calculating the constrained values
    x_values = np.loadtxt("saved_data/combined_x_"+str(min_temperature)+"_degree_warming_cmip6.csv", delimiter=",")
    y_values = np.loadtxt("saved_data/combined_y_"+str(min_temperature)+"_degree_warming_cmip6.csv", delimiter=",")
    new_xobs = np.loadtxt("saved_data/x_obs_"+str(min_temperature)+"_degree_warming_cmip6.csv", delimiter=",")
    new_dxobs = np.loadtxt("saved_data/dx_obs_"+str(min_temperature)+"_degree_warming_cmip6.csv", delimiter=",")
    # Plotting the y axis constrained values
    mean_ec_y_value, lower_ec_limit, upper_ec_limit = EC_pdf_UU_reduced(x_values, y_values, new_xobs.item(), new_dxobs.item())

    # creating constrained data line
    y_line = np.linspace(-xmin_limit, x_obs-dx_obs, 100)
    ec_array = np.zeros([100,1])
    ec_array = np.squeeze(ec_array)
    for b in range(0,100):
        ec_array[b] = mean_ec_y_value
    # plotting
    plt.plot(y_line, ec_array, color='b', linewidth=2, alpha=1)

    print('new mean:', ec_array[0])
    print('new std:', upper_ec_limit-ec_array[0])

    xmax = (xmin_limit+(x_obs-dx_obs))/(xmin_limit+xmax_limit)
    plt.axhspan(lower_ec_limit, upper_ec_limit, xmin=0, xmax=xmax, color='lightblue', alpha=0.8, zorder=20)

    plt.plot(xfit, yfit, color='k', linewidth=2)

    one_to_one_line = np.linspace(-xmin_limit, xmax_limit, 100)
    plt.plot(one_to_one_line, one_to_one_line, 'darkgrey', linewidth=0.25)


    handels_1 = []
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='b', label='SSP126')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='g', label='SSP245')])
    handels_1.extend([Line2D([0,0],[0,0], linewidth=20, color='r', label='SSP585')])
    label_1 = ['SSP126', 'SSP245', 'SSP585']

    # handels_2 = []
    # handels_2.extend([Line2D([0,0],[0,0], linewidth=4, color='k', label='Emergent Relationship')])
    # handels_2.extend([Line2D([0,0],[0,0], linewidth=4, color='b', alpha=0.5, label='Observational-derived mean')])
    # handels_2.extend([Line2D([0,0],[0,0], linewidth=20, color='lightblue', alpha=0.8, label='Standard Deviation')])
    # handels_2.extend([Line2D([0,0],[0,0], linewidth=4, color='darkgreen', alpha=0.5, label='Emergent Constraint')])
    # handels_2.extend([Line2D([0,0],[0,0], linewidth=20, color='lightgreen', alpha=0.8, label='Standard Deviation')])
    # label_2 = ['Emergent Relationship', 'Observational-derived mean', 'Standard Deviation', 'Emergent Constraint', 'Standard Deviation']

    handels = []
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='ACCESS-ESM1-5')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='BCC-CSM2-MR')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='v', markersize=20, color='k', label='CanESM5')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='1', markersize=20, color='k', label='CESM2')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='CNRM-ESM2-1')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='*', markersize=20, color='k', label='IPSL-CM6A-LR')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='x', markersize=20, color='k', label='MIROC-ES2L')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='+', markersize=20, color='k', label='NorESM2-LM')])
    handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='d', markersize=20, color='k', label='UKESM1-0-LL')])
    handels.extend([Line2D([0,0],[0,0], linewidth=20, color='lightgreen', alpha=0.8, label='Observational Constraint')])
    handels.extend([Line2D([0,0],[0,0], linewidth=20, color='lightblue', alpha=0.8, label='Emergent Constraint')])
    label = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'NorESM2-LM', 'UKESM1-0-LL', 'Observational Constraint', 'Emergent Constraint']

    leg1 = plt.legend(handels, label, loc=3, fontsize=30)
    leg2 = plt.legend(handels_1, label_1, loc=4, fontsize=30)
    #leg3 = ax.legend(handels_2, label_2, loc=2, fontsize=30)

    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)

    #ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks())))

    plt.xlim((-xmin_limit, xmax_limit))
    plt.ylim((-xmin_limit, xmax_limit))

    #plt.text(-0.17, 0.9999, 'a',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize = 44)

    plt.xlabel(r'Relationship derived $\Delta C_{s, \tau}$')
    plt.ylabel(r'Model calculated $\Delta C_{s, \tau}$')
        
    # save figure
    if min_temperature == 0.5:
        fig.savefig('final_plots/cmip6_classicEC_05degreeswarming_CARDrh_new.pdf', bbox_inches='tight')
        plt.close()
    else:
        fig.savefig('paper_plots/cmip6_'+str(min_temperature)+'degreeswarming_CARDrh_quadratic_new.pdf', bbox_inches='tight')
        plt.close()
            