#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis Python Script for Varney et al. 2020 Nature Communications
- script finds model soil turnover time (tau_s) calculated using model output and then subsequent
modelled change in soil carbon (deltaCs,tau), and relationship-derived deltaCs,tau, which is calculated
using the model-specific spatial temperature sensitivities of tau (quadratic fits) and model temperature.
- calculated for CMIP5 models
"""

#%%

# Analysis imports
import numpy as np
import numpy.ma as ma

# My functions
from rmv_cmip_analysis import combine_netCDF_cmip5
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import numpy_to_cube
from rmv_cmip_analysis import global_total_percentage


#%%
#inputs

# historical / present day dates
lower_historical = 1995
upper_historical = 2005
# future dates
lower = 2090
upper = 2100


# CMIP5 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)
model_shapes = ['o', '^', 'v', '1', 's', '*', 'x', '+', 'd']

# RCP senarios
rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)


# defining empty numpy array to save values
x_array = ma.zeros((len(rcp_options), len(cmip5_models)))
y_array = ma.zeros((len(rcp_options), len(cmip5_models)))


#%%
# Loop through each rcp run being considered
for rcp_option in range(0, rcp_options_length):
    rcp = rcp_options[rcp_option] # selecting the rcp scenario

    # for loop for each CMIP5 model
    for model_i in range(0, n_models):
        model = cmip5_models[model_i] # seleting the models

        print(rcp, model)


        #%% historical soil turnover time

        # Heterotrophic Respiration (RH)
        rh_historical_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/rh_Lmon_'+model+'_historical*', 'heterotrophic_respiration_carbon_flux', model)
        rh_historical_cube = open_netCDF(rh_historical_cube)
        # Soil Carbon (cSoil)
        cSoil_historical_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_historical*', 'soil_carbon_content', model)
        cSoil_historical_cube = open_netCDF(cSoil_historical_cube)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_historical*', 'air_temperature', model)
        tas_historical_cube = open_netCDF(tas_historical_cube)

        # Select historical time period
        rh_historical_cube = select_time(rh_historical_cube, lower_historical, upper_historical)
        cSoil_historical_cube = select_time(cSoil_historical_cube, lower_historical, upper_historical)
        tas_historical_cube = select_time(tas_historical_cube, lower_historical, upper_historical)
        # Time average
        rh_historical_cube = time_average(rh_historical_cube)
        cSoil_historical_cube = time_average(cSoil_historical_cube)
        tas_historical_cube = time_average(tas_historical_cube)
        # Converting from cubes to numpy_arrays
        rh_historical_data = rh_historical_cube.data
        cSoil_historical_data = cSoil_historical_cube.data
        tas_historical_data = tas_historical_cube.data

        # save to use later
        historical_tas_save_data = tas_historical_data - 273.15
        cSoil_historical_save_cube = cSoil_historical_cube.copy()
        historical_rh_save_data = rh_historical_data*86400.*365.


        # Calculating Soil Turnover Time (tau_s)
        tau_s_data_historical = cSoil_historical_data / (rh_historical_data*86400.*365.)
        tau_s_masked_data_historical = ma.masked_where(np.logical_or(tau_s_data_historical < 1, tau_s_data_historical > 1e4), tau_s_data_historical)


        #%% finding the spatial relationship

        # masking tas data with corresponding mask
        tas_historical_data = ma.masked_where(np.logical_or(tau_s_data_historical < 1, tau_s_data_historical > 1e4), tas_historical_data)
        # changing the x variable air temperature to celcius from kelvin
        xvar_historical = tas_historical_data - 273.15

        # define x and y and flatten
        x = xvar_historical.flatten()
        y = tau_s_masked_data_historical.flatten()
        y = ma.log(y) # numpy masked log of y

        # model-specific quadratic relationship
        p = np.ma.polyfit(x, y, 2)
        poly_relationship = np.poly1d(p)


        #%% finding estimated Cs

        # historical
        tau_s_historical_estimated = poly_relationship(historical_tas_save_data)

        # future
        tas_future_cube = combine_netCDF_cmip5('/home/links/rmv203/cmip5_data/tas_Amon_'+model+'_'+rcp+'_*', 'air_temperature', model)
        tas_future_cube = open_netCDF(tas_future_cube)
        # select time
        tas_future_cube = select_time(tas_future_cube, lower, upper)
        # time average
        tas_future_cube = time_average(tas_future_cube)
        # cube to numpy array
        tas_future_data = tas_future_cube.data
        # K to C degrees
        tas_future_data = tas_future_data - 273.15
        # estimating future tau_s with polynomial relationship
        tau_s_future_estimated = poly_relationship(tas_future_data)

        # estimated delta tau_s
        delta_tau_estimated = ma.exp(tau_s_future_estimated) - ma.exp(tau_s_historical_estimated)
        # estimated delta soil carbon (relationship-derived deltaCs,tau)
        delta_c_soil_estimated = delta_tau_estimated*historical_rh_save_data


        #%% finding model Cs

        # historical
        tau_historical_model = tau_s_masked_data_historical.copy()

        # future
        # Heterotrophic Respiration (RH)
        rh_future_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/rh_Lmon_'+model+'_'+rcp+'*', 'heterotrophic_respiration_carbon_flux', model)
        rh_future_cube = open_netCDF(rh_future_cube)
        # Soil Carbon (cSoil)
        cSoil_future_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/cSoil_Lmon_'+model+'_'+rcp+'*', 'soil_carbon_content', model)
        cSoil_future_cube = open_netCDF(cSoil_future_cube)
        # Select future time period
        rh_future_cube = select_time(rh_future_cube, lower, upper)
        cSoil_future_cube = select_time(cSoil_future_cube, lower, upper)
        # Time average
        rh_future_cube = time_average(rh_future_cube)
        cSoil_future_cube = time_average(cSoil_future_cube)
        # Converting from cubes to numpy_arrays
        rh_future_data = rh_future_cube.data
        cSoil_future_data = cSoil_future_cube.data
        # Calculating future soil turnover time
        tau_s_data = cSoil_future_data / (rh_future_data*86400.*365.)
        tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
        tau_future_model = tau_s_masked_data.copy()

        # Modelled delta tau_s
        delta_tau_model = tau_future_model - tau_historical_model
        # calculating delta soil carbon (model deltaCs,tau)
        delta_c_soil_model = delta_tau_model*historical_rh_save_data


        #%% Calculating the global averaged value of both delta Cs

        # Masking invalid values
        delta_c_soil_estimated = np.ma.masked_invalid(delta_c_soil_estimated)
        delta_c_soil_model = np.ma.masked_invalid(delta_c_soil_model)
        # convert numpy array to cube
        delta_c_soil_model_cube = numpy_to_cube(delta_c_soil_model, cSoil_historical_save_cube, 2)
        delta_c_soil_estimated_cube = numpy_to_cube(delta_c_soil_estimated, cSoil_historical_save_cube, 2)
        # landfracs
        landfraction = combine_netCDF_cmip5('/home/rmv203/cmip5_data/sftlf_fx_'+model+'_*', 'land_area_fraction', model)
        # global totals
        model_delta_cSoil_global = global_total_percentage(delta_c_soil_model_cube, landfrac=landfraction, latlon_cons=None)
        model_delta_cSoil_global_data = model_delta_cSoil_global.data
        estimate_delta_cSoil_global = global_total_percentage(delta_c_soil_estimated_cube, landfrac=landfraction, latlon_cons=None)
        estimate_delta_cSoil_global_data = estimate_delta_cSoil_global.data


        # saving delta Cs values
        x_array[rcp_option, model_i] = estimate_delta_cSoil_global_data
        y_array[rcp_option, model_i] = model_delta_cSoil_global_data


        #%%
        # saving variables
        np.save('saved_variables/historical_tas_data_'+model+'.npy', historical_tas_save_data.data)
        np.save('saved_variables/historical_tas_mask_'+model+'.npy', historical_tas_save_data.mask)
        np.save('saved_variables/historical_rh_data_'+model+'.npy', historical_rh_save_data.data)
        np.save('saved_variables/historical_rh_mask_'+model+'.npy', historical_rh_save_data.mask)
        np.save('saved_variables/historical_modelled_tau_data_'+model+'.npy', tau_historical_model.data)
        np.save('saved_variables/historical_modelled_tau_mask_'+model+'.npy', tau_historical_model.mask)
        np.save('saved_variables/poly_relationship_'+model+'.npy', poly_relationship)


#%%
# saving data
        
# looping through each rcp
for j in range(0, rcp_options_length):
    rcp = rcp_options[j]
    
    # saving x_array and y_array for each rcp
    np.savetxt("saved_data/x_"+str(rcp)+"_cmip5.csv",  x_array[j,:], delimiter=",")
    np.savetxt("saved_data/y_"+str(rcp)+"_cmip5.csv",  y_array[j,:], delimiter=",")
        
    # saving the r coefficient for x_array and y_array for each rcp
    r_coeffient = ma.corrcoef(x_array[j,:], y_array[j,:])
    print('CMIP5 r-coefficent:', rcp, r_coeffient)
    np.savetxt("saved_data/cmip5_xy_rcoefficient_"+str(rcp)+".csv",  r_coeffient, delimiter=",")
    
    # saving mean delta Cs from CMIP5 models for each rcp
    mean_delta_Cs_cmip5 = np.nanmean(y_array[j,:])
    mean_delta_Cs_cmip5 = np.array([mean_delta_Cs_cmip5])
    print('CMIP5 delta Cs mean:', rcp, mean_delta_Cs_cmip5)
    np.savetxt("saved_data/cmip5_mean_model_deltaCs_"+str(rcp)+".csv",  mean_delta_Cs_cmip5, delimiter=",")
    
    # saving std in delta Cs from CMIP5 models for each rcp
    std_delta_Cs_cmip5 = np.nanstd(y_array[j,:])
    std_delta_Cs_cmip5 = np.array([std_delta_Cs_cmip5])
    print('CMIP5 delta Cs std:', rcp, std_delta_Cs_cmip5)
    np.savetxt("saved_data/cmip5_std_model_deltaCs_"+str(rcp)+".csv",  std_delta_Cs_cmip5, delimiter=",")
            

# saving over all rcp runs
    
# saving the r coefficient for x_array and y_array
x_array_flatten = x_array.flatten()
y_array_flatten = y_array.flatten()
r_coeffient = ma.corrcoef(x_array_flatten, y_array_flatten)
print('CMIP5 all rcps r-coefficent:', r_coeffient)
np.savetxt("saved_data/cmip5_xy_rcoefficient_allrcps.csv",  r_coeffient, delimiter=",")

# saving mean delta Cs from CMIP5 models
mean_delta_Cs_cmip5 = np.nanmean(y_array)
mean_delta_Cs_cmip5 = np.array([mean_delta_Cs_cmip5])
print('CMIP5 delta Cs mean (all rcps):', mean_delta_Cs_cmip5)
np.savetxt("saved_data/cmip5_mean_model_deltaCs_allrcps.csv",  mean_delta_Cs_cmip5, delimiter=",")

# saving std in delta Cs from CMIP5 models
std_delta_Cs_cmip5 = np.nanstd(y_array)
std_delta_Cs_cmip5 = np.array([std_delta_Cs_cmip5])
print('CMIP5 delta Cs std (all rcps):', std_delta_Cs_cmip5)
np.savetxt("saved_data/cmip5_std_model_deltaCs_allrcps.csv",  std_delta_Cs_cmip5, delimiter=",")

