##!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis Python Script for Varney et al. 2020 Nature Communications
- script finds model soil turnover time (tau_s) calculated using model output and then subsequent modelled change in
soil carbon (model deltaCs,tau), and relationship-derived change in soil carbon (relationship-derived deltaCs,tau),
which is calculated using the model-specific spatial temperature sensitivities of tau (quadratic fits) and model temperature
- calculated for CMIP6 models
- investigating if relationship-derived deltaCs,tau and model deltaCs,tau are similar (on one-to-one line)
for each model considered in this study
- pofp_analysis: change is considered between time averaged historical (1995-2005) and then time averaged at the
 end of a future SSP scenario (2090-2100)
"""

#%%

# Analysis imports
import numpy as np
import numpy.ma as ma

# My functions
from rmv_cmip_analysis import combine_netCDF_model
from rmv_cmip_analysis import combine_netCDF_cmip6
from rmv_cmip_analysis import combine_netCDF_rh_cmip6
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


# CMIP6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
n_models = len(cmip6_models)
model_shapes = ['o', '^', '+', 's', '*', 'd', 'x']

# SSP scenarios
ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)


# defining empty numpy array to save values
x_array = ma.zeros((len(ssp_options), len(cmip6_models)))
y_array = ma.zeros((len(ssp_options), len(cmip6_models)))


#%%
# Loop through each ssp run being considered
for ssp_option in range(0, ssp_options_length):
    ssp = ssp_options[ssp_option] # selecting the ssp scenarios

    # for loop for each CMIP5 model
    for model_i in range(0, n_models):
        model = cmip6_models[model_i] # seleting the models

        print(ssp, model)


        #%% historical soil turnover time

        # Heterotrophic Respiration (RH)
        rh_historical_cube = combine_netCDF_rh_cmip6('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_historical*', model)
        rh_historical_cube = open_netCDF(rh_historical_cube)
        # Soil Carbon (cSoil)
        cSoil_historical_cube = combine_netCDF_cmip6('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_historical*', model)
        cSoil_historical_cube = open_netCDF(cSoil_historical_cube)
        # Near Surface Air Temperature (tas)
        tas_historical_cube = combine_netCDF_cmip6('/home/rmv203/cmip6_data/tas_Amon_'+model+'_historical*', model)
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

        # saving for later
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
        x = np.ma.ravel(xvar_historical)
        y = np.ma.ravel(tau_s_masked_data_historical)
        y = ma.log(y) # numpy masked log of y

        # model-specific quadratic relationship
        p = np.ma.polyfit(x, y, 2)
        poly_relationship = np.poly1d(p)


        #%% finding estimated Cs

        # historical
        tau_s_historical_estimated = poly_relationship(historical_tas_save_data)

        # future
        tas_future_cube = combine_netCDF_cmip6('/home/links/rmv203/cmip6_data/tas_Amon_'+model+'_'+ssp+'_*', model)
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

        # Estimated delta tau_s
        delta_tau_estimated = ma.exp(tau_s_future_estimated) - ma.exp(tau_s_historical_estimated)
        # estimated delta soil carbon (relationship-derived deltaCs,tau)
        delta_c_soil_estimated = delta_tau_estimated*historical_rh_save_data


        #%%
        # finding model Cs

        # historical
        tau_historical_model = tau_s_masked_data_historical.copy()

        # future
        # Heterotrophic Respiration (RH)
        rh_future_cube = combine_netCDF_rh_cmip6('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_'+ssp+'*', model)
        rh_future_cube = open_netCDF(rh_future_cube)
        # Soil Carbon (cSoil)
        cSoil_future_cube = combine_netCDF_cmip6('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_'+ssp+'*', model)
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

        # Calculating Soil Turnover Time
        tau_s_data = cSoil_future_data / (rh_future_data*86400.*365.)
        tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
        tau_future_model = tau_s_masked_data.copy()

        # Modelled delta tau_s
        delta_tau_model = tau_future_model - tau_historical_model
        # calculating model delta soil carbon (model deltaCs,tau)
        delta_c_soil_model = delta_tau_model*historical_rh_save_data


        #%%
        # Calculating the global averaged value of both delta Cs

        # Masking invalid values
        delta_c_soil_estimated = np.ma.masked_invalid(delta_c_soil_estimated)
        delta_c_soil_model = np.ma.masked_invalid(delta_c_soil_model)
        # convert numpy array to cube
        delta_c_soil_model_cube = numpy_to_cube(delta_c_soil_model, cSoil_historical_save_cube, 2)
        delta_c_soil_estimated_cube = numpy_to_cube(delta_c_soil_estimated, cSoil_historical_save_cube, 2)
        # landfrac
        landfraction = combine_netCDF_model('/home/rmv203/cmip6_data/sftlf_fx_'+model+'_historical*', model)
        # Global totals
        model_delta_cSoil_global = global_total_percentage(delta_c_soil_model_cube, landfrac=landfraction, latlon_cons=None)
        model_delta_cSoil_global_data = model_delta_cSoil_global.data
        estimate_delta_cSoil_global = global_total_percentage(delta_c_soil_estimated_cube, landfrac=landfraction, latlon_cons=None)
        estimate_delta_cSoil_global_data = estimate_delta_cSoil_global.data


        # saving delta Cs values
        x_array[ssp_option, model_i] = estimate_delta_cSoil_global_data
        y_array[ssp_option, model_i] = model_delta_cSoil_global_data


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

# looping through each ssp
for j in range(0, ssp_options_length):
    ssp = ssp_options[j]
    
    # saving x_array and y_array for each ssp
    np.savetxt("saved_data/x_"+str(ssp)+"_cmip6.csv",  x_array[j,:], delimiter=",")
    np.savetxt("saved_data/y_"+str(ssp)+"_cmip6.csv",  y_array[j,:], delimiter=",")
        
    # saving the r coefficient for x_array and y_array for each ssp
    r_coeffient = ma.corrcoef(x_array[j,:], y_array[j,:])
    print('CMIP6 r-coefficent:', ssp, r_coeffient)
    np.savetxt("saved_data/cmip6_xy_rcoefficient_"+str(ssp)+".csv",  r_coeffient, delimiter=",")
        
    # saving mean delta Cs from CMIP5 models for each ssp
    mean_delta_Cs_cmip6 = np.nanmean(y_array[j,:])
    mean_delta_Cs_cmip6 = np.array([mean_delta_Cs_cmip6])
    print('CMIP6 delta Cs mean:', ssp, mean_delta_Cs_cmip6)
    np.savetxt("saved_data/cmip6_mean_model_deltaCs_"+str(ssp)+".csv",  mean_delta_Cs_cmip6, delimiter=",")
    
    # saving std in delta Cs from CMIP5 models for each ssp
    std_delta_Cs_cmip6 = np.nanstd(y_array[j,:])
    std_delta_Cs_cmip6 = np.array([std_delta_Cs_cmip6])
    print('CMIP6 delta Cs std:', ssp, std_delta_Cs_cmip6)
    np.savetxt("saved_data/cmip6_std_model_deltaCs_"+str(ssp)+".csv",  std_delta_Cs_cmip6, delimiter=",")
            

# saving over all ssp runs
    
# saving the r coefficient for x_array and y_array
x_array_flatten = x_array.flatten()
y_array_flatten = y_array.flatten()
r_coeffient = ma.corrcoef(x_array_flatten, y_array_flatten)
print('CMIP6 all ssps r-coefficent:', r_coeffient)
np.savetxt("saved_data/cmip6_xy_rcoefficient_allssps.csv",  r_coeffient, delimiter=",")

# saving mean delta Cs from CMIP5 models
mean_delta_Cs_cmip6 = np.nanmean(y_array)
mean_delta_Cs_cmip6 = np.array([mean_delta_Cs_cmip6])
print('CMIP6 delta Cs mean (all ssps):', mean_delta_Cs_cmip6)
np.savetxt("saved_data/cmip6_mean_model_deltaCs_allssps.csv",  mean_delta_Cs_cmip6, delimiter=",")

# saving std in delta Cs from CMIP5 models
std_delta_Cs_cmip6 = np.nanstd(y_array)
std_delta_Cs_cmip6 = np.array([std_delta_Cs_cmip6])
print('CMIP6 delta Cs std (all ssps):', std_delta_Cs_cmip6)
np.savetxt("saved_data/cmip6_std_model_deltaCs_allssps.csv",  std_delta_Cs_cmip6, delimiter=",")
