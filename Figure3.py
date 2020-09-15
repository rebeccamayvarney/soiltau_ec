#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Script produces Figure 3 in Varney et al. 2020 Nature Communications
(a) spatial temperature sensitivity of soil turnover (tau vs T) - observational data in black, CMIP ESMs in colour
(b) proof of principle plot (model deltaCs,tau vs estimated deltaCs,tau)
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
# Set up subplot figure

fig_figure3 = plt.figure(1, figsize=(56,18))
gs = gspec.GridSpec(1, 2, figure=fig_figure3, hspace=5, wspace=0.5)
column = 0 # for figure 1
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


#%%
# FIGURE 3a
ax = fig_figure3.add_subplot(gs[row, column])


#%%
# importing saved observational variables

poly_relationship_obs = np.poly1d(np.load('saved_variables/poly_relationship_obs.npy'))

observational_temperature_data = np.load('saved_variables/observational_temperature_data.npy')
observational_temperature_mask = np.load('saved_variables/observational_temperature_mask.npy')
observational_temperature = np.ma.masked_array(observational_temperature_data, mask=observational_temperature_mask)

observational_rh_data = np.load('saved_variables/observational_rh_data.npy')
observational_rh_mask = np.load('saved_variables/observational_rh_mask.npy')
observational_rh = np.ma.masked_array(observational_rh_data, mask=observational_rh_mask)

observational_Cs_data = np.load('saved_variables/observational_Cs_data.npy')
observational_Cs_mask = np.load('saved_variables/observational_Cs_mask.npy')
observational_Cs = np.ma.masked_array(observational_Cs_data, mask=observational_Cs_mask)

#%%
# plotting obs spatial data

tau_s = observational_Cs / observational_rh
tau_s_masked = ma.masked_where(np.logical_or(tau_s < 1, tau_s > 1e4), tau_s)
observational_temperature = ma.masked_where(np.logical_or(tau_s < 1, tau_s > 1e4), observational_temperature)

x_obs = np.ma.ravel(observational_temperature)
y_obs = np.ma.ravel(tau_s_masked)
logy_obs = ma.log(y_obs)
ax.plot(x_obs, logy_obs, 'ko', markersize=0.5, alpha=0.25)


#%%
# CMIP6 and CMIP5 Models

models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL', 'BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
model_colours = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen', 'olive', 'gold', 'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5', '#bebada']
n_models = len(models)


# for loop for each CMIP5 model
for model_i in range(0, n_models):

        model = models[model_i] # seleting the models  
        print(model)

        # load model data
        poly_relationship_model = np.poly1d(np.load('saved_variables/poly_relationship_'+model+'.npy'))
        model_historical_tas_data = np.load('saved_variables/historical_tas_data_'+model+'.npy')
        model_historical_tas_mask = np.load('saved_variables/historical_tas_mask_'+model+'.npy')
        model_historical_tas = np.ma.masked_array(model_historical_tas_data, mask=model_historical_tas_mask)

        x = np.ma.ravel(model_historical_tas)
        sorted_x = np.sort(x)
        ax.plot(sorted_x, poly_relationship_model(sorted_x), model_colours[model_i], linewidth=5)


# observational polyfit
sorted_x_obs = np.sort(x_obs)
ax.plot(sorted_x_obs, poly_relationship_obs(sorted_x_obs), 'k', linewidth=20, linestyle='dotted')


ax.set_xlabel(r'$T$ ($^{o}C$)')
ax.set_ylabel(r'$\mathrm{log}(\tau_\mathrm{s}$) (log(yr))')
ax.set_xlim((-22, 30))
ax.set_ylim((1, 9))


# legends

handels_3 = []
handels_3.extend([Line2D([0,0],[0,0], linewidth=20, color='k', label='Observations')])
label_3 = ['Observations']
leg4 = ax.legend(handels_3, label_3, loc=1, fontsize=34)
plt.gca().add_artist(leg4)

handels = []
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkblue', label='ACCESS-ESM1-5')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='dodgerblue', label='BCC-CSM2-MR')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#80b1d3', label='CanESM5')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkcyan', label='CNRM-ESM2-1')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#8dd3c7', label='IPSL-CM6A-LR')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkseagreen', label='MIROC-ES2L')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='darkgreen', label='UKESM1-0-LL')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='olive', label='BNU-ESM')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='gold', label='CanESM2')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='orange', label='CESM1-CAM5')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='peachpuff', label='GFDL-ESM2G')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#fb8072', label='GISS-E2-R')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='red', label='HadGEM2-ES')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='hotpink', label='IPSL-CM5A-LR')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#fccde5', label='MIROC-ESM')])
handels.extend([Line2D([0,0],[0,0], linewidth=20, color='#bebada', label='NorESM1-M')])
labels = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL', 'BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
leg1 = ax.legend(handels, labels, loc='center right', borderaxespad=0.2, bbox_to_anchor=(1.345, 0.5), title='Model Colours', fontsize=34)
plt.gca().add_artist(leg1)

# (a)
ax.text(-0.1, 0.9999, 'a', transform=ax.transAxes, va='top', fontweight='bold', fontsize=34)


#%%
# FIGURE 3b
column += 1
ax = fig_figure3.add_subplot(gs[row, column])

#%%

cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)

# future scenarios
ssp_options = ['ssp126', 'ssp245', 'ssp585']
rcp_options = ['rcp26', 'rcp45', 'rcp85']
ssp_options_length = len(ssp_options)


# Loop through each ssp run being considered
for ssp in range(0, ssp_options_length):
        current_ssp = ssp_options[ssp]
        current_rcp = rcp_options[ssp]

        # 
        x_array_cmip6 = np.loadtxt("saved_data/x_"+str(current_ssp)+"_cmip6.csv", delimiter=",")
        y_array_cmip6 = np.loadtxt("saved_data/y_"+str(current_ssp)+"_cmip6.csv", delimiter=",")
        x_array_cmip5 = np.loadtxt("saved_data/x_"+str(current_rcp)+"_cmip5.csv", delimiter=",")
        y_array_cmip5 = np.loadtxt("saved_data/y_"+str(current_rcp)+"_cmip5.csv", delimiter=",")

        for model_j in range(0, n_models):

                # Plotting CMIP5
                if model_j > 6:
                        x_value_cmip5 = x_array_cmip5[model_j]
                        y_value_cmip5 = y_array_cmip5[model_j]

                        cmip5_marker = model_j + 7
                        # plotting
                        if current_rcp == 'rcp26':
                                ax.plot(x_value_cmip5, y_value_cmip5, marker='o', color=model_colours[cmip5_marker], markersize=20, mew=5)
                        elif current_rcp == 'rcp45':
                                ax.plot(x_value_cmip5, y_value_cmip5, marker='^', color=model_colours[cmip5_marker], markersize=20, mew=5)
                        elif current_rcp == 'rcp85':
                                ax.plot(x_value_cmip5, y_value_cmip5, marker='s', color=model_colours[cmip5_marker], markersize=20, mew=5)

                # Plotting CMIP5 and CMIP6
                else:
                        x_value_cmip5 = x_array_cmip5[model_j]
                        y_value_cmip5 = y_array_cmip5[model_j]
                        x_value_cmip6 = x_array_cmip6[model_j]
                        y_value_cmip6 = y_array_cmip6[model_j]

                        cmip5_marker = model_j + 7
                        # plotting
                        if current_rcp == 'rcp26':
                                ax.plot(x_value_cmip5, y_value_cmip5, marker='o', color=model_colours[cmip5_marker], markersize=20, mew=5)
                        elif current_rcp == 'rcp45':
                                ax.plot(x_value_cmip5, y_value_cmip5, marker='^', color=model_colours[cmip5_marker], markersize=20, mew=5)
                        elif current_rcp == 'rcp85':
                                ax.plot(x_value_cmip5, y_value_cmip5, marker='s', color=model_colours[cmip5_marker], markersize=20, mew=5)
                        # plotting
                        if current_ssp == 'ssp126':
                                ax.plot(x_value_cmip6, y_value_cmip6, marker='o', color=model_colours[model_j], markersize=20, mew=5)
                        elif current_ssp == 'ssp245':
                                ax.plot(x_value_cmip6, y_value_cmip6, marker='^', color=model_colours[model_j], markersize=20, mew=5)
                        elif current_ssp == 'ssp585':
                                ax.plot(x_value_cmip6, y_value_cmip6, marker='s', color=model_colours[model_j], markersize=20, mew=5)


# one to one line
one_to_one_line = np.linspace(-850,50,100)
ax.plot(one_to_one_line, one_to_one_line, 'grey', linewidth=1)

# legend
handels_2 = []
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='SSP126 / RCP2.6')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='SSP245 / RCP4.5')])
handels_2.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='SSP585 / RCP8.5')])
label_2 = ['SSP126 / RCP2.6', 'SSP245 / RCP4.5', 'SSP585 / RCP8.5']
leg3 = ax.legend(handels_2, label_2, loc=2, fontsize=34)
plt.gca().add_artist(leg3)


ax.set_xlim((-799, 50))
ax.set_ylim((-799, 50))
ax.set_xlabel(r'Relationship-derived $\Delta C_{\mathrm{s, \tau}}$ (PgC)', fontsize=34)
ax.set_ylabel(r'Model $\Delta C_{\mathrm{s, \tau}}$ (PgC)', fontsize=34)


# (b)
ax.text(-0.14, 0.9999, 'b',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


#%%
# Save figure
fig_figure3.savefig('paper_figures/Figure3_v1.pdf', bbox_inches='tight')
plt.close()


#%%
# finding r coefficent of proof of principle plot (b)
x_array = np.concatenate((x_array_cmip6, x_array_cmip5), axis=0)
y_array = np.concatenate((y_array_cmip6, y_array_cmip5), axis=0)
x_array_flatten = x_array.flatten()
y_array_flatten = y_array.flatten()
r_coeffient = ma.corrcoef(x_array_flatten, y_array_flatten)
print('Combined CMIP all ssps/rcps r-coefficent:', r_coeffient)