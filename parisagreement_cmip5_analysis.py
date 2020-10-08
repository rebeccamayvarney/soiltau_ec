#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis Python Script for Varney et al. 2020 Nature Communications
- script finds model soil turnover time (tau_s) calculated using model output and then subsequent modelled change in
soil carbon (model deltaCs,tau), and relationship-derived change in soil carbon (relationship-derived deltaCs,tau),
which is calculated using the model-specific spatial temperature sensitivities of tau (quadratic fits) and model temperature
- calculated for CMIP5 models
- investigating if relationship-derived deltaCs,tau and model deltaCs,tau are similar (on one-to-one line)
for each model considered in this study
- parisagreement_analysis: change is considered between time averaged historical (1995-2005) and then a degree
(1 degree, 2 degrees, 3 degrees) of global mean warming
"""

#%%

# Analysis imports
import numpy as np
import numpy.ma as ma
import iris
import iris.coord_categorisation
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes

# My functions
from rmv_cmip_analysis import combine_netCDF_variable
from rmv_cmip_analysis import combine_netCDF_cmip5
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import annual_average
from rmv_cmip_analysis import numpy_to_cube
from rmv_cmip_analysis import regrid_model
from rmv_cmip_analysis import area_average
from rmv_cmip_analysis import global_total
from rmv_cmip_analysis import global_total_percentage


#%%
# Importing saved variables

poly_relationship_obs = np.poly1d(np.load('saved_variables/poly_relationship_obs.npy'))
#
observational_temperature_data = np.load('saved_variables/observational_temperature_data.npy')
observational_temperature_mask = np.load('saved_variables/observational_temperature_mask.npy')
observational_temperature = np.ma.masked_array(observational_temperature_data, mask=observational_temperature_mask)
#
observational_rh_data = np.load('saved_variables/observational_rh_data.npy')
observational_rh_mask = np.load('saved_variables/observational_rh_mask.npy')
observational_rh = np.ma.masked_array(observational_rh_data, mask=observational_rh_mask)

# loading observational land fraction
landfraction_obs = combine_netCDF_variable('/home/links/rmv203/obs_datasets/luc4c_landmask.nc', 'mask')


#%%

# Loading regrid cube
regrid_cube = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
regrid_cube = time_average(regrid_cube)
regrid_cube.coord('latitude').guess_bounds()
regrid_cube.coord('longitude').guess_bounds()
regrid_modelcube = regrid_cube.copy()
# correct lat and lon dimensions
n_lat = regrid_cube.coord('latitude').points
n_lon = regrid_cube.coord('longitude').points


#%%
# inputs

lower_historical = 1995
upper_historical = 2005

region_global = [0, 360, -90,  90]

# CMIP5 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)
model_shapes = ['o', '^', 'v', '1', 's', '*', 'x', '+', 'd']

# RCP scenarios
rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)

# Global mean temperature change
temperature_change_options = [1, 2, 3]
temperature_change_options_length = len(temperature_change_options)


#%%
# defining array to save values
x_array = ma.zeros([len(temperature_change_options), len(rcp_options), len(cmip5_models)])
y_array = ma.zeros([len(temperature_change_options), len(rcp_options), len(cmip5_models)])
obs_array = ma.zeros([len(temperature_change_options), len(rcp_options)*len(cmip5_models)])


#%%
# loop through each global mean temperature change
for temp_option in range(0, temperature_change_options_length):
    min_temperature = temperature_change_options[temp_option] # selecting the temperature change    

    # Loop through each rcp run being considered
    for rcp_option in range(0, rcp_options_length):
        rcp = rcp_options[rcp_option] # selecting the rcp scenario

        # for loop for each cmip5 model
        for model_i in range(0, n_models):
            model = cmip5_models[model_i] # seleting the models

            print(min_temperature, rcp, model)
            
            #%%
            
            #%%
            # finding spatial profile for future temperature

            # time averaged, area averaged historical/present day temperature
            tas_preindustrial_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_data/tas_Amon_'+model+'_historical*', 'air_temperature', model)
            tas_preindustrial_cube = open_netCDF(tas_preindustrial_cube)        
            tas_preindustrial_cube = select_time(tas_preindustrial_cube, 1995, 2005)
            tas_preindustrial_cube = time_average(tas_preindustrial_cube)
            tas_preindustrial_cube = area_average(tas_preindustrial_cube, region_global)
            tas_preindustrial_data = tas_preindustrial_cube.data
    
            # cube to find future temperature change
            tas_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_historical_'+rcp+'/tas_Amon_'+model+'_*', 'air_temperature', model)
            tas_cube = open_netCDF(tas_cube)
            tas_test_cube = annual_average(tas_cube)
            # defining the time variable for years
            tas_test_years = tas_test_cube.coord('year').points
            tas_test_cube = area_average(tas_test_cube, region_global)
            tas_test_data = tas_test_cube.data
            
            # finding year global mean temperature change is reached
            for i in range(5, len(tas_test_data)-5):
                    decadal_average = np.mean(tas_test_data[i-5:i+5])
                    temp_change = decadal_average - tas_preindustrial_data
                    if temp_change >= min_temperature:
                            temp_change_year = tas_test_years[i]
                            print(temp_change_year)
                            tas_new_cube = select_time(tas_cube, temp_change_year-5, temp_change_year+5)
                            break

            # if not reached, continue
            if temp_change < min_temperature:
                    print('model too cold', model)
                    x_array[temp_option, rcp_option, model_i] = np.nan
                    y_array[temp_option, rcp_option, model_i] = np.nan
                    obs_array[temp_option, model_i+(rcp_option*n_models)] = np.nan
                    continue
    
            # saving future spatial temperature
            tas_new_cube = time_average(tas_new_cube)
            save_model_temp_cube = tas_new_cube.copy()
            save_model_temp_data = tas_new_cube.data - 273.15
            
            
            #%%
            # loading pre-found model specific data

            historical_tas_data = np.load('saved_variables/historical_tas_data_'+model+'.npy')
            historical_tas_mask = np.load('saved_variables/historical_tas_mask_'+model+'.npy')
            historical_tas = np.ma.masked_array(historical_tas_data, mask=historical_tas_mask)
            #
            historical_rh_data = np.load('saved_variables/historical_rh_data_'+model+'.npy')
            historical_rh_mask = np.load('saved_variables/historical_rh_mask_'+model+'.npy')
            historical_rh = np.ma.masked_array(historical_rh_data, mask=historical_rh_mask)
            #       
            historical_modelled_tau_data = np.load('saved_variables/historical_modelled_tau_data_'+model+'.npy')
            historical_modelled_tau_mask = np.load('saved_variables/historical_modelled_tau_mask_'+model+'.npy')
            historical_modelled_tau = np.ma.masked_array(historical_modelled_tau_data, mask=historical_modelled_tau_mask)
            #
            poly_relationship = np.poly1d(np.load('saved_variables/poly_relationship_'+model+'.npy'))
            
            
            #%%
            # estimated deltaCs,tau
            
            # finding historical estimated tau for model
            historical_estimated_tau = poly_relationship(historical_tas)
            
            # finding future estimated tau for model
            future_estimated_tau = poly_relationship(save_model_temp_data)
            
            # estimated spatial delta Cs
            estimated_delta_tau = ma.exp(future_estimated_tau) - ma.exp(historical_estimated_tau)
            estimated_delta_Cs = estimated_delta_tau*historical_rh
            
            
            #%%
            # model deltaCs,tau

            # finding future model tau for model
            rh_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_historical_'+rcp+'/rh_Lmon_'+model+'_*', 'heterotrophic_respiration_carbon_flux', model)
            rh_cube = open_netCDF(rh_cube)
            cSoil_cube = combine_netCDF_cmip5('/home/rmv203/cmip5_historical_'+rcp+'/cSoil_Lmon_'+model+'_*', 'soil_carbon_content', model)
            cSoil_cube = open_netCDF(cSoil_cube)
            # Select future time period
            rh_cube = select_time(rh_cube, temp_change_year-5, temp_change_year+5)
            cSoil_cube = select_time(cSoil_cube, temp_change_year-5, temp_change_year+5)
            # Time average
            rh_cube = time_average(rh_cube)
            cSoil_cube = time_average(cSoil_cube)
            # Converting from cubes to numpy_arrays
            rh_data = rh_cube.data
            cSoil_data = cSoil_cube.data
            # Calculating Soil Turnover Time
            tau_s_data = cSoil_data / (rh_data*86400.*365.)
            tau_s_masked_data = ma.masked_where(np.logical_or(tau_s_data < 1, tau_s_data > 1e4), tau_s_data)
            
            # model spatial delta Cs
            model_delta_tau = tau_s_masked_data - historical_modelled_tau
            model_delta_Cs = model_delta_tau*historical_rh


            #%%
            # global totals

            # model land fraction
            landfraction = combine_netCDF_cmip5('/home/rmv203/cmip5_data/sftlf_fx_'+model+'*', 'land_area_fraction', model)
    
            # Masking invalid values
            estimated_delta_Cs = np.ma.masked_invalid(estimated_delta_Cs)
            model_delta_Cs = np.ma.masked_invalid(model_delta_Cs)
            # calculating global totals
            estimated_delta_Cs_cube = numpy_to_cube(estimated_delta_Cs, cSoil_cube, 2)
            model_delta_Cs_cube = numpy_to_cube(model_delta_Cs, cSoil_cube, 2)
            estimated_delta_Cs_cube = global_total_percentage(estimated_delta_Cs_cube, landfrac=landfraction, latlon_cons=None)
            model_delta_Cs_cube = global_total_percentage(model_delta_Cs_cube, landfrac=landfraction, latlon_cons=None)
            estimated_delta_Cs_data = estimated_delta_Cs_cube.data
            model_delta_Cs_data = model_delta_Cs_cube.data

            #%%
            # saving delta Cs values
            x_array[temp_option, rcp_option, model_i] = estimated_delta_Cs_data
            y_array[temp_option, rcp_option, model_i] = model_delta_Cs_data


            #%%
            # finding the observational derived constraint
            
            # historical model temperature
            historical_tas_cube = numpy_to_cube(historical_tas, cSoil_cube, 2)
            historical_tas_cube_regrid = regrid_model(historical_tas_cube, regrid_cube)
            historical_tas_data_regrid = historical_tas_cube_regrid.data

            # future model temperature
            model_future_temp_cube_regrid = regrid_model(save_model_temp_cube, regrid_cube)
            model_future_temp_data_regrid = model_future_temp_cube_regrid.data - 273.15
            
            # deriving future 'real world' temperature (model change + observations)
            observational_future_temp = (model_future_temp_data_regrid - historical_tas_data_regrid) + observational_temperature
            
            # Calculating new tau_s with observational relationship
            historical_tau_obs = poly_relationship_obs(observational_temperature)
            future_tau_obs = poly_relationship_obs(observational_future_temp)
        
            # calculating deltaCs,tau
            delta_tau_obs = ma.exp(future_tau_obs) - ma.exp(historical_tau_obs)
            delta_cSoil_obs = delta_tau_obs*observational_rh
            
            # global totals
            delta_cSoil_obs = np.ma.masked_invalid(delta_cSoil_obs)
            delta_Cs_obs_cube = numpy_to_cube(delta_cSoil_obs, regrid_cube, 2)
            delta_Cs_obs_cube = global_total(delta_Cs_obs_cube, landfrac=landfraction_obs, latlon_cons=None)
            delta_Cs_obs_data = delta_Cs_obs_cube.data
        

            # saving delta Cs observational derived values
            obs_array[temp_option, model_i+(rcp_option*n_models)] = delta_Cs_obs_data
            

#%%
# saving data
      
# looping through for each temperature change option
for j in range(0, temperature_change_options_length):
    t = temperature_change_options[j]
    
    # saving x_array and y_array for each temperature change
    if temperature_change_options[j] == 0.5:
        np.savetxt("saved_data/x_05_degree_warming_cmip5.csv",  x_array[j,:,:], delimiter=",")
        np.savetxt("saved_data/y_05_degree_warming_cmip5.csv",  y_array[j,:,:], delimiter=",")
    else:
        np.savetxt("saved_data/x_"+str(t)+"_degree_warming_cmip5.csv",  x_array[j,:,:], delimiter=",")
        np.savetxt("saved_data/y_"+str(t)+"_degree_warming_cmip5.csv",  y_array[j,:,:], delimiter=",")
        
    # saving the r coefficient for x_array and y_array at each temperature change
    x_array_flatten = x_array[j,:,:]
    x_array_flatten = x_array_flatten.flatten()
    y_array_flatten = y_array[j,:,:]
    y_array_flatten = y_array_flatten.flatten()
    r_coeffient = ma.corrcoef(ma.masked_invalid(x_array_flatten), ma.masked_invalid(y_array_flatten))
    print('CMIP5 r-coefficient (all rcps)', t, r_coeffient)
    if temperature_change_options[j] == 0.5:
        np.savetxt("saved_data/cmip5_xy_rcoefficient_05_degree_warming.csv",  r_coeffient, delimiter=",")
    else:
        np.savetxt("saved_data/cmip5_xy_rcoefficient_"+str(t)+"_degree_warming.csv",  r_coeffient, delimiter=",")
        
        
    # saving the observational derived constrained values
    if temperature_change_options[j] == 0.5:
        np.savetxt("saved_data/obs_constraint_05_degree_warming_cmip5.csv", obs_array[j,:], delimiter=",")
    else:
        np.savetxt("saved_data/obs_constraint_"+str(t)+"_degree_warming_cmip5.csv", obs_array[j,:], delimiter=",")
    
    
    # saving mean delta Cs from CMIP5 models
    select_temp = ma.masked_invalid(y_array[j,:,:])
    select_temp = select_temp.flatten()
    mean_delta_Cs_cmip5 = np.nanmean(select_temp)
    mean_delta_Cs_cmip5 = np.array([mean_delta_Cs_cmip5])
    print('CMIP5 delta Cs mean at degrees:', t, mean_delta_Cs_cmip5)
    if temperature_change_options[j] == 0.5:
        np.savetxt("saved_data/cmip5_mean_model_deltaCs_05_degree_warming.csv",  mean_delta_Cs_cmip5, delimiter=",")
    else:
        np.savetxt("saved_data/cmip5_mean_model_deltaCs_"+str(t)+"_degree_warming.csv",  mean_delta_Cs_cmip5, delimiter=",")
    
    # saving std in delta Cs from CMIP5 models
    std_delta_Cs_cmip5 = np.nanstd(select_temp)
    std_delta_Cs_cmip5 = np.array([std_delta_Cs_cmip5])
    print('CMIP5 delta Cs std at degrees:', t, std_delta_Cs_cmip5)
    if temperature_change_options[j] == 0.5:
        np.savetxt("saved_data/cmip5_std_model_deltaCs_05_degree_warming.csv",  std_delta_Cs_cmip5, delimiter=",")
    else:
        np.savetxt("saved_data/cmip5_std_model_deltaCs_"+str(t)+"_degree_warming.csv",  std_delta_Cs_cmip5, delimiter=",")
