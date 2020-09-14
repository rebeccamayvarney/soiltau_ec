
# -*- coding: utf-8 -*-

# Analysis
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from scipy import stats
import scipy as sp
import sys

# from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")

# Iris imports
import iris
import iris.coord_categorisation
import iris.quickplot as qplt
import iris.plot as iplt
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes

# My functions
from rmv_cmip_analysis import combine_netCDF
from rmv_cmip_analysis import combine_netCDF_time_overlap
from rmv_cmip_analysis import open_netCDF
from rmv_cmip_analysis import define_attributes
from rmv_cmip_analysis import select_time
from rmv_cmip_analysis import time_average
from rmv_cmip_analysis import annual_average

# Plotting
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams, colors
import matplotlib as mpl
from matplotlib import gridspec as gspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpat
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


#%%

# SUBPLOT FIGURE
fig_historical = plt.figure(1, figsize=(18,14))
gs = gspec.GridSpec(3, 3, figure=fig_historical, hspace=0.2)
n = 9 # (3*3)

# set rows and column
n_columns_1 = 3 # for figure 1
n_row_1 = 3
column_1 = 0 # for figure 1
row_1 = 0


mpl.rcParams['xtick.direction'] = 'out'       # set 'ticks' pointing inwards
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True             # add ticks to top and right hand axes  
mpl.rcParams['ytick.right'] = True           # of plot 


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

# cmip6 Models
cmip6_data_in = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
# number of models
n_models = len(cmip6_data_in)


# Inputs
lower_historical = 1995
upper_historical = 2005
region_global = [0, 360, -90,  90] # global average


# for loop for each cmip6 model
for model_i, a, in zip(range(n_models), range(n)):

    print('Historical, Model: '+cmip6_data_in[model_i]) # print the name of the model being considered
    model = cmip6_data_in[model_i] # seleting the models


    # Heterotrophic Respiration (RH)
    rh_historical_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/rh_Lmon_'+model+'_historical*', model)
    rh_historical_cube = open_netCDF(rh_historical_cube)

    # Soil Carbon (cSoil)
    cSoil_historical_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/cSoil_Emon_'+model+'_historical*', model)
    cSoil_historical_cube = open_netCDF(cSoil_historical_cube)

    # Near Surface Air Temperature (tas)
    tas_historical_cube = combine_netCDF_time_overlap('/home/rmv203/cmip6_data/tas_Amon_'+model+'_historical*', model)
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

    # masking data (trans tt)
    tau_s_masked_data_historical = ma.masked_where(np.logical_or(tau_s_data_historical < 1, tau_s_data_historical > 1e4), tau_s_data_historical)
    # masking tas data with corresponding mask
    tas_historical_time_av_data = ma.masked_where(np.logical_or(tau_s_data_historical < 1, tau_s_data_historical > 1e4), tas_historical_time_av_data)

    # changing the x variable air temperature to celcius from kelvin
    xvar_historical = tas_historical_time_av_data - 273.15

    
    #%%
    # Plotting soil turnover time aganist tas

    x = np.ma.ravel(xvar_historical)
    y = np.ma.ravel(tau_s_masked_data_historical)
    y = ma.log(y)

    p = np.ma.polyfit(x, y, 2)
    poly_relationship = np.poly1d(p)


    ax = fig_historical.add_subplot(gs[row_1, column_1])

    ax.plot(x, y, 'ko', markersize=1)

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
fig_historical.savefig('paper_figures/SupplementaryFigure2_v1.pdf', bbox_inches='tight')
plt.close()
