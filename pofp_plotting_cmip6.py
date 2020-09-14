#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:26:13 2020

@author: rmv203
"""
#%%

# Analysis imports
import numpy as np
import numpy.ma as ma

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#%%
# Figure

fig = plt.figure(1, figsize=(16,12))

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
# inputs

# cmip6 models
cmip6_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'UKESM1-0-LL']
n_models = len(cmip6_models)
model_shapes = ['o', '^', '+', 's', '*', 'd', 'x']


ssp_options = ['ssp126', 'ssp245', 'ssp585']
ssp_options_length = len(ssp_options)


#%%
# looping through each ssp
for j in range(0, ssp_options_length):
    ssp = ssp_options[j]
    
    # loading data
    x_data = np.loadtxt('saved_data/x_'+str(ssp)+'_cmip6.csv')
    y_data = np.loadtxt('saved_data/y_'+str(ssp)+'_cmip6.csv')
    
    # for loop for each CMIP5 model
    for model_i in range(0, n_models):
        model = cmip6_models[model_i] # seleting the models
        
        print(ssp, model)

        #%%
        # plotting
        if ssp == 'ssp585':
                plt.plot(x_data[model_i], y_data[model_i], marker=model_shapes[model_i], color='r', markersize=10, mew=5)
        elif ssp == 'ssp245':
                plt.plot(x_data[model_i], y_data[model_i], marker=model_shapes[model_i], color='g', markersize=10, mew=5)
        elif ssp == 'ssp126':
                plt.plot(x_data[model_i], y_data[model_i], marker=model_shapes[model_i], color='b', markersize=10, mew=5)


#%%
                
min_axis_value = -950
max_axis_value = 50
       
# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100)
plt.plot(one_to_one_line, one_to_one_line, 'grey', linewidth=1)

# legend
handels_1 = []
handels_1.extend([Line2D([0,0],[0,0], linewidth=15, color='b', label='ssp126')])
handels_1.extend([Line2D([0,0],[0,0], linewidth=15, color='g', label='ssp245')])
handels_1.extend([Line2D([0,0],[0,0], linewidth=15, color='r', label='ssp585')])
label_1 = ['ssp126', 'ssp245', 'ssp585']

handels = []
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='ACCESS-ESM1-5')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='BCC-CSM2-MR')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='v', markersize=20, color='k', label='CanESM5')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='1', markersize=20, color='k', label='CESM2')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='CNRM-ESM2-1')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='x', markersize=20, color='k', label='IPSL-CM6A-LR')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='+', markersize=20, color='k', label='MIROC-ES2L')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='d', markersize=20, color='k', label='NorESM2-LM')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='*', markersize=20, color='k', label='UKESM1-0-LL')])
handels.extend([Line2D([0,0],[0,0], linewidth=1, color='grey', label='one to one line')])
label = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC-ES2L', 'NorESM2-LM', 'UKESM1-0-LL', 'one to one line']

leg1 = plt.legend(handels, label, loc=2)
leg2 = plt.legend(handels_1, label_1, loc=4)

plt.gca().add_artist(leg1)
plt.gca().add_artist(leg2)

# axis limits
plt.xlim((-850, 50))
plt.ylim((-850, 50))

# axis labels
plt.xlabel(r'Relationship-derived $\Delta C_{s, \tau}$ ($PgC$)')
plt.ylabel(r'Modelled $\Delta C_{s, \tau}$ ($PgC$)')

# save figure
#fig.savefig('reviewer_plots/proof_of_principal_cmip6_cSoilAbove1m.pdf', bbox_inches='tight')
plt.close()

