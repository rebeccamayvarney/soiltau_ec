#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script produces Figure 4 in Varney et al. 2020 Nature Communications
(a) Emergent Constraint Plot
- x-axis: estimated deltaCs,tau / observational constraint, y axis: model deltaCs,tau / emergent constraint
(b) Probability Density Function of Emergent Constraint
(c) Emergent Constrained values of global mean deltaCs,tau against global mean temperature change (delta T)
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

# My functions
from rmv_analysis_functions import combine_netCDF_observations
from rmv_analysis_functions import time_average
from rmv_analysis_functions import numpy_to_cube
from rmv_analysis_functions import regrid_model
from rmv_analysis_functions import global_total
from rmv_analysis_functions import lin_reg_UU
from rmv_analysis_functions import obtaining_Cs_q10
from rmv_analysis_functions import EC_pdf_UU_reduced

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
# Set up subplot figure

fig_figure4 = plt.figure(1, figsize=(76,18))
gs = gspec.GridSpec(1, 3, figure=fig_figure4, hspace=0.2)
column = 0
row = 0
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


#%% FIGURE 4a
ax = fig_figure4.add_subplot(gs[row, column])


# loading observational data
poly_relationship_obs = np.poly1d(np.load('saved_variables/poly_relationship_obs.npy'))
observational_temperature_data = np.load('saved_variables/observational_temperature_data.npy')
observational_temperature_mask = np.load('saved_variables/observational_temperature_mask.npy')
observational_temperature = np.ma.masked_array(observational_temperature_data, mask=observational_temperature_mask)
observational_rh_data = np.load('saved_variables/observational_rh_data.npy')
observational_rh_mask = np.load('saved_variables/observational_rh_mask.npy')
observational_rh = np.ma.masked_array(observational_rh_data, mask=observational_rh_mask)


# CMIP6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
# SSP scenarios
ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)

# CMIP5 models
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)
# RCP scenarios
rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)

# model colours
model_colours = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen', 'olive', 'gold', 'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5', '#bebada']


# loading saved data
x_data_cmip5 = np.loadtxt('saved_data/x_2_degree_warming_cmip5.csv',  delimiter=',')
y_data_cmip5 = np.loadtxt('saved_data/y_2_degree_warming_cmip5.csv',  delimiter=',')
x_data_cmip6 = np.loadtxt('saved_data/x_2_degree_warming_cmip6.csv',  delimiter=',')
y_data_cmip6 = np.loadtxt('saved_data/y_2_degree_warming_cmip6.csv',  delimiter=',')

xfit = np.loadtxt("saved_data/EC_xfit_2degreewarming_cmip6cmip5.csv", delimiter=',')
yfit = np.loadtxt("saved_data/EC_yfit_2degreewarming_cmip6cmip5.csv", delimiter=',')


#%%
#  Loop through each rcp run being considered
for rcp_option in range(0, rcp_options_length):
    rcp = rcp_options[rcp_option] # selecting the rcp scenario
    ssp = ssp_options[rcp_option]

    # for loop for each cmip6 model
    for model_i in range(0, n_models):

        if model_i > 6:
            # seleting the models
            model_cmip5 = cmip5_models[model_i]       
            print(rcp, model_cmip5)
                
            cmip5_modelshape = model_i+7
            # Plotting CMIP5
            if rcp == 'rcp85':
                    plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker='s', color=model_colours[cmip5_modelshape], markersize=20, alpha=1)
            elif rcp == 'rcp45':
                    plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker='^', color=model_colours[cmip5_modelshape], markersize=20, alpha=1)
            elif rcp == 'rcp26':
                    plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker='o', color=model_colours[cmip5_modelshape], markersize=20, alpha=1)


        else:
            # seleting the models
            model_cmip6 = cmip6_models[model_i]
            model_cmip5 = cmip5_models[model_i]
            print(ssp, model_cmip6)
            ssp_option = rcp_option

            # Plotting CMIP6
            if ssp == 'ssp585':
                    plt.plot(x_data_cmip6[ssp_option, model_i], y_data_cmip6[ssp_option, model_i], marker='s', color=model_colours[model_i], markersize=20, alpha=1)
            elif ssp == 'ssp245':
                    plt.plot(x_data_cmip6[ssp_option, model_i], y_data_cmip6[ssp_option, model_i], marker='^', color=model_colours[model_i], markersize=20, alpha=1)
            elif ssp == 'ssp126':
                    plt.plot(x_data_cmip6[ssp_option, model_i], y_data_cmip6[ssp_option, model_i], marker='o', color=model_colours[model_i], markersize=20, alpha=1)

            cmip5_modelshape = model_i+7
            # Plotting CMIP5
            if rcp == 'rcp85':
                    plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker='s', color=model_colours[cmip5_modelshape], markersize=20, alpha=1)
            elif rcp == 'rcp45':
                    plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker='^', color=model_colours[cmip5_modelshape], markersize=20, alpha=1)
            elif rcp == 'rcp26':
                    plt.plot(x_data_cmip5[rcp_option, model_i], y_data_cmip5[rcp_option, model_i], marker='o', color=model_colours[cmip5_modelshape], markersize=20, alpha=1)


#%%
# calculating the constrained values

xmin_limit = 550
xmax_limit = 0

# loading saved numpy arrays
x_values = np.loadtxt("saved_data/combined_x_2_degree_warming_cmip6cmip5.csv", delimiter=",")
y_values = np.loadtxt("saved_data/combined_y_2_degree_warming_cmip6cmip5.csv", delimiter=",")
new_xobs = np.loadtxt("saved_data/x_obs_2_degree_warming_cmip6cmip5.csv", delimiter=",")
new_dxobs = np.loadtxt("saved_data/dx_obs_2_degree_warming_cmip6cmip5.csv", delimiter=",")

# creating constrained data line and shaded uncertainty
x_line = np.linspace(-xmin_limit, xmax_limit, 100)
global_array = np.zeros([100,1])
global_array = np.squeeze(global_array)
for b in range(0,100):
    global_array[b] = new_xobs
plt.plot(global_array, x_line, color='darkgreen', linewidth=5, alpha=1)
plt.axvspan(new_xobs-new_dxobs, new_xobs+new_dxobs, color='lightgreen', alpha=0.4, zorder=20)

# Plotting the y axis constrained values
mean_ec_y_value, lower_ec_limit, upper_ec_limit = EC_pdf_UU_reduced(x_values, y_values, new_xobs.item(), new_dxobs.item())
y_line = np.linspace(-xmin_limit, new_xobs-new_dxobs, 100)
ec_array = np.zeros([100,1])
ec_array = np.squeeze(ec_array)
for b in range(0,100):
    ec_array[b] = mean_ec_y_value
plt.plot(y_line, ec_array, color='b', linewidth=5, alpha=1)
xmax = (xmin_limit+(new_xobs-new_dxobs))/(xmin_limit+xmax_limit)
plt.axhspan(lower_ec_limit, upper_ec_limit, xmin=0, xmax=xmax, color='lightblue', alpha=0.4, zorder=20)

# plotting emergent relationship
plt.plot(xfit, yfit, color='k', linewidth=3)

# plotting one to one line
one_to_one_line = np.linspace(-xmin_limit, xmax_limit, 100)
plt.plot(one_to_one_line, one_to_one_line, 'darkgrey', linewidth=0.5)


#  legend
handels_2 = []
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='SSP126 / RCP2.6')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='SSP245 / RCP4.5')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='SSP585 / RCP8.5')])
handels_2.extend([Line2D([0,0],[0,0], linewidth=20, color='lightgreen', label='Observational Constraint')])
handels_2.extend([Line2D([0,0],[0,0], linewidth=20, color='lightblue', label='Emergent Constraint')])
label_2 = ['SSP126 / RCP2.6', 'SSP245 / RCP4.5', 'SSP585 / RCP8.5', 'Observational Constraint', 'Emergent Constraint']
leg3 = ax.legend(handels_2, label_2, loc=2, fontsize=34)
plt.gca().add_artist(leg3)

ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks())))

# axis limits and labels
ax.set_xlim((-xmin_limit, xmax_limit))
ax.set_ylim((-xmin_limit, xmax_limit))
ax.set_xlabel(r'Relationship-derived $\Delta C_{\mathrm{s, \tau}}$ (PgC)')
ax.set_ylabel(r'Model $\Delta C_{\mathrm{s, \tau}}$ (PgC)')

# (a)
ax.text(-0.16, 0.9999, 'a',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


# emergent constraint plus uncertainty:
print('new mean:', ec_array[0])
print('new std:', upper_ec_limit-ec_array[0])


#%% FIGURE 4b
column += 1
ax = fig_figure4.add_subplot(gs[row, column])


x_obs = new_xobs # observational constraint
dx_obs = new_dxobs # observational uncertainty

# Read x and y variables
x = np.loadtxt("saved_data/combined_x_2_degree_warming_cmip6cmip5.csv", delimiter=",")
y = np.loadtxt("saved_data/combined_y_2_degree_warming_cmip6cmip5.csv", delimiter=",")

# Calculate mean and stdev of (equal model weight) prior
mn_pr=np.mean(y)
std_pr=np.std(y)

# Observational Constraint
mn=x_obs
std=dx_obs
xbest=x_obs
xlo=x_obs-dx_obs
xhi=x_obs+dx_obs

# Define ranges for plots
yr1=min(y)-0.1*(max(y)-min(y))
yr2=max(y)+0.1*(max(y)-min(y))
xr1=min(x)-0.1*(max(x)-min(x))
xr2=max(x)+0.1*(max(x)-min(x))

# Calculate best-fit straight-line between x & y
yf,a,b,da,db,xfit,yfit,yband=lin_reg_UU(x,y)

# Calculate PDF for IAV constraints
x2=xfit
nfitx=len(xfit)
dx=x2[1]-x2[0]
Px=x2
Pi=3.142
Px=1/np.sqrt(2*Pi*std**2) * np.exp(-((x2-mn)/(np.sqrt(2)*std))**2)

miny=mn_pr-5*std_pr
maxy=mn_pr+5*std_pr
mfity=2000
dy=(maxy-miny)/mfity
y2=miny+dy*np.arange(0,mfity)

# Calculate prior
Py_pr=y2
Py_pr=1/np.sqrt(2*Pi*std_pr**2)*np.exp(-((y2-mn_pr)/(np.sqrt(2)*std_pr))**2)

# Calculate contours of probability in (x,y) space
Pxy=np.zeros((nfitx,mfity))
Pyx=np.zeros((mfity,nfitx))
Py=np.zeros(mfity)
Py_norm=0.0
for m in range(0, mfity):
    Py[m]=0.0
    for n in range(0,nfitx):
        Py_given_x=1/np.sqrt(2*Pi*yband[n]**2) \
            * np.exp(-((y2[m]-yfit[n])/(np.sqrt(2)*yband[n]))**2)
        Pxy[n,m]=Px[n]*Py_given_x
        Pyx[m,n]=Pxy[n,m]
# Integrate over x to get Py
        Py[m]=Py[m]+Pxy[n,m]*dx
    pass
    Py_norm=Py_norm+Py[m]*dy
pass

# Normalise Py
for m in range(0, mfity):
    Py[m]=Py[m]/Py_norm
pass

# Plot Probability Density Function
ax.set_xlim(min(y2),max(y2))
ax.plot(y2,Py,'b-',label='Emergent Constraint',linewidth=5)
ax.plot(y2,Py_pr,'k-',label='Gaussian Model Fit',linewidth=5)
ax.set_xlabel(r'Model $\Delta C_{\mathrm{s, \tau}}$ (PgC)')
ax.set_ylabel(r'Probablity Density Per PgC')
ax.set_xlim((-650,250))
dum=np.argmax(Py)
ybest=y2[dum]

dum_pr=np.argmax(Py_pr)
ybest_pr=y2[dum_pr]
# binny=min(y2)+(max(y2)-min(y2))*arange(11)/10.0
binny=[-500, -400, -300, -200, -100, 0, 100]
#    binny=[100,200,300,400,500,600,700,800,900]
n, bins, patches = plt.hist(y, bins=binny,\
                    normed=1, facecolor='grey',label='Model Range')
ax.legend(fontsize=34, loc='upper right')
print(y2, y)

# (b)
ax.text(-0.17, 0.9999, 'b',transform=ax.transAxes,va = 'top',fontweight = 'bold', fontsize=34)


#%% FIGURE 4c
column += 1
ax = fig_figure4.add_subplot(gs[row, column])


# global mean temperature changes being considered
temperature_change_options = [1, 2, 3]
temperature_change_options_length = len(temperature_change_options)

# creating empty arrays
mean_ec_y_array = np.zeros((len(temperature_change_options)+1))
lower_ec_limit_array = np.zeros((len(temperature_change_options)+1))
upper_ec_limit_array = np.zeros((len(temperature_change_options)+1))

# loading saved data
for temp_option in range(0, temperature_change_options_length):
    # selecting the temperature change
    min_temperature = temperature_change_options[temp_option]

    mean_ec_y_array[temp_option+1] = np.loadtxt("saved_data/mean_EC_yvalue_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    lower_ec_limit_array[temp_option+1] = np.loadtxt("saved_data/EC_lower_limit_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    upper_ec_limit_array[temp_option+1] = np.loadtxt("saved_data/EC_upper_limit_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")


#%% Observational soil carbon

# regrid cube
regrid_cube = iris.load_cube('/home/links/rmv203/obs_datasets/Tair_WFDEI_ann.nc')
regrid_cube = time_average(regrid_cube)
regrid_cube.coord('latitude').guess_bounds()
regrid_cube.coord('longitude').guess_bounds()
regrid_modelcube = regrid_cube.copy()

# observational land frac
landfraction_obs = combine_netCDF_observations('/home/links/rmv203/obs_datasets/luc4c_landmask.nc', 'mask')

# observational soil carbon
observational_Cs_data = np.load('saved_variables/observational_Cs_data.npy')
observational_Cs_mask = np.load('saved_variables/observational_Cs_mask.npy')
observational_Cs0 = np.ma.masked_array(observational_Cs_data, mask=observational_Cs_mask)
# global total of observational soil carbon
observational_Cs0_cube = numpy_to_cube(observational_Cs0, regrid_modelcube, 2)
observational_Cs0_cube = global_total(observational_Cs0_cube, landfrac=landfraction_obs, latlon_cons=None)
observational_Cs0_data = observational_Cs0_cube.data


#%% finding delta Cs q10s

x_axis_options = [1, 2, 3]
new_temperature_change_options = [0, 1, 2, 3]
new_temperature_change_options_length = len(new_temperature_change_options)

# the q10 parameters being considered
q10_parameters = [2, 2.5, 3]

# empty number arrays
temp_degree_q10s = np.zeros((len(q10_parameters), len(new_temperature_change_options)))

# loop through q10 parameters
for q10 in range(0, len(q10_parameters)):
    q10_parameter = q10_parameters[q10]

    # loop through global mean temperature changes
    for new_temp_option in range(0, new_temperature_change_options_length):
        # selecting the temperature change
        temp_change = new_temperature_change_options[new_temp_option]
        
        if temp_change == 0:
            temp_degree_q10s[q10, new_temp_option] = 0
        else:
            temp_degree_q10s[q10, new_temp_option] = obtaining_Cs_q10(q10_parameter, observational_Cs0_data, temp_change)


# plotting
q10_colours = ['lightgrey', 'dimgrey', 'k']
labels = [r'q$_{10}$=2', r'q$_{10}$=2.5', r'q$_{10}$=3']

# q10s
for q10 in range(0, len(q10_parameters)):
    q10_parameter = q10_parameters[q10]
    
    plt.scatter(new_temperature_change_options, temp_degree_q10s[q10, :], color=q10_colours[q10], s=100, label=labels[q10])
    best_trend_q10 = np.ma.polyfit(new_temperature_change_options, temp_degree_q10s[q10, :], 3)
    best_trend_equation_q10 = np.poly1d(best_trend_q10)
    plt.plot(new_temperature_change_options, best_trend_equation_q10(new_temperature_change_options), color=q10_colours[q10], linewidth=5)

# emergent constraints
plt.scatter(new_temperature_change_options, mean_ec_y_array, color='b', s=100, label='Constrained Values')
best_trend = np.ma.polyfit(new_temperature_change_options, mean_ec_y_array, 3)
best_trend_equation = np.poly1d(best_trend)
sorted_temp_options = np.sort(new_temperature_change_options)
plt.plot(sorted_temp_options, best_trend_equation(sorted_temp_options), 'b-', linewidth=5)
# uncertainty bounds
plt.fill_between(new_temperature_change_options, lower_ec_limit_array, upper_ec_limit_array, color='lightblue', label='Confidence Limits', alpha=0.4, zorder=20)


ax.legend(fontsize=34)
ax.set_xlim((0, 3))
ax.set_ylim((-425,25))
plt.xlabel(r'Global mean temperature warming ($^{\circ}$C)')
plt.ylabel(r'Magnitude of global $C_\mathrm{s}$ loss due to $\Delta \tau_\mathrm{s}$ (PgC)')

# (c)
ax.text(-0.16, 0.9999, 'c',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


#%% Save figure
fig_figure4.savefig('paper_figures/Figure4_v1.pdf', bbox_inches='tight')
plt.close()
