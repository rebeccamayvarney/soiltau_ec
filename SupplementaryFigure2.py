#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script produces Supplementary Figure 2 in Varney et al. 2020 Nature Communications
- finding the quadratic fit for tau v T (spatial temperature sensitivity of soil carbon turnover time)
for each CMIP6 model in this study
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
from rmv_cmip_analysis import combine_netCDF_cmip6
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import define_attributes
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import annual_average

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
# set up of figure
fig_cmip6 = plt.figure(1, figsize=(18,14))
gs = gspec.GridSpec(3, 3, figure=fig_cmip6, hspace=0.2)
n = 9
column_1 = 0
row_1 = 0
n_columns_1 = 3
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':2,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 26,
    'xtick.labelsize':26,
    'ytick.labelsize':26,
    'font.size':26,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


#%%
# inputs

lower_historical = 1995
upper_historical = 2005

# CMIP6 Models
cmip6_data_in = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
n_models = len(cmip6_data_in)


# for loop for each cmip6 model
for model_i, a, in zip(range(n_models), range(n)):

    # subplot pannel for each model
    ax = fig_cmip6.add_subplot(gs[row_1, column_1])

    # seleting the models
    model = cmip6_data_in[model_i]
    print(model)


    # Heterotrophic Respiration (RH)
    rh_historical_cube = combine_netCDF_cmip6('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_historical*', model)
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
    rh_historical_time_av_cube = time_average(rh_historical_cube)
    cSoil_historical_time_av_cube = time_average(cSoil_historical_cube)
    tas_historical_time_av_cube = time_average(tas_historical_cube)
    # Converting from cubes to numpy_arrays
    rh_historical_time_av_data = rh_historical_time_av_cube.data
    cSoil_historical_time_av_data = cSoil_historical_time_av_cube.data
    tas_historical_time_av_data = tas_historical_time_av_cube.data


    # Calculating Soil Turnover Time
    tau_s_data_historical = cSoil_historical_time_av_data / (rh_historical_time_av_data* 86400.*365.)
    tau_s_masked_data_historical = ma.masked_where(np.logical_or(tau_s_data_historical < 1, tau_s_data_historical > 1e4), tau_s_data_historical)
    

    #%%
    # Plotting soil turnover time aganist T

    # masking tas data with corresponding mask
    tas_historical_time_av_data = ma.masked_where(np.logical_or(tau_s_data_historical < 1, tau_s_data_historical > 1e4), tas_historical_time_av_data)
    # changing the x variable air temperature to celcius from kelvin
    xvar_historical = tas_historical_time_av_data - 273.15

    x = np.ma.ravel(xvar_historical)
    y = np.ma.ravel(tau_s_masked_data_historical)
    y = ma.log(y)
    ax.plot(x, y, 'ko', markersize=1)

    p = np.ma.polyfit(x, y, 2)
    poly_relationship = np.poly1d(p)
    sorted_x = np.sort(x)
    ax.plot(sorted_x, poly_relationship(sorted_x), 'b-', label='Quadratic fit')

    ax.title.set_text(str(cmip6_data_in[model_i]))
    ax.set_xlim((-22, 30))
    ax.set_ylim((1, 9.9))
    # labelling only the far left figures
    if column_1 == 0:
        ax.set_ylabel(r'$\mathrm{log}(\tau_\mathrm{s})$')
        ax.tick_params(labelleft=True)
    else:
        ax.tick_params(labelleft=False)
    # label bottom axes in bottom row only
    if row_1 == 2:
        ax.set_xlabel(r'$T$ ($^{\circ}$C)')
        ax.tick_params(labelbottom=True)
    else:
        ax.tick_params(labelbottom=False)

    if (row_1 == 0) & (column_1 == 2):
        ax.legend(fontsize=26)


    # increase row and column 
    row_1 += 1 
    if (a-column_1*n_columns_1)==n_columns_1-1: 
        column_1 += 1 
        row_1 = 0


#%%
fig_cmip6.savefig('paper_figures/SupplementaryFigure2_v1.pdf', bbox_inches='tight')
plt.close()
