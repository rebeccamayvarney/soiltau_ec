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
cmip5_models = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M']
n_models = len(cmip5_models)
model_shapes = ['o', '^', 'v', '1', 's', '*', 'x', '+', 'd']


rcp_options = ['rcp26', 'rcp45', 'rcp85']
rcp_options_length = len(rcp_options)


#%%

# looping through each rcp
for j in range(0, rcp_options_length):
    rcp = rcp_options[j]
    
    # loading data
    x_data = np.loadtxt('saved_data/x_'+str(rcp)+'_cmip5.csv')
    y_data = np.loadtxt('saved_data/y_'+str(rcp)+'_cmip5.csv')
    
    # for loop for each CMIP5 model
    for model_i in range(0, n_models):
        model = cmip5_models[model_i] # seleting the models
        
        print(rcp, model)


        #%%
        # plotting
        if rcp == 'rcp85':
                plt.plot(x_data[model_i], y_data[model_i], marker=model_shapes[model_i], color='r', markersize=10, mew=5)
        elif rcp == 'rcp45':
                plt.plot(x_data[model_i], y_data[model_i], marker=model_shapes[model_i], color='g', markersize=10, mew=5)
        elif rcp == 'rcp26':
                plt.plot(x_data[model_i], y_data[model_i], marker=model_shapes[model_i], color='b', markersize=10, mew=5)


#%%
                
min_axis_value = -850
max_axis_value = 50
       
# one to one line         
one_to_one_line = np.linspace(min_axis_value, max_axis_value, 100) # one to one line
plt.plot(one_to_one_line, one_to_one_line, 'grey', linewidth=1)

# legend
handels_1 = []
handels_1.extend([Line2D([0,0],[0,0], linewidth=15, color='b', label='RCP2.6')])
handels_1.extend([Line2D([0,0],[0,0], linewidth=15, color='g', label='RCP4.5')])
handels_1.extend([Line2D([0,0],[0,0], linewidth=15, color='r', label='RCP8.5')])
label_1 = ['RCP2.6', 'RCP4.5', 'RCP8.5']

handels = []
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='o', markersize=20, color='k', label='BNU-ESM')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='^', markersize=20, color='k', label='CanESM2')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='v', markersize=20, color='k', label='CESM1-CAM5')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='1', markersize=20, color='k', label='GFDL-ESM2G')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='s', markersize=20, color='k', label='GISS-E2-R')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='*', markersize=20, color='k', label='HadGEM2-ES')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='x', markersize=20, color='k', label='IPSL-CM5A-LR')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='+', markersize=20, color='k', label='MIROC-ESM')])
handels.extend([Line2D([0,0],[0,0], linestyle='None', marker='d', markersize=20, color='k', label='NorESM1-M')])
handels.extend([Line2D([0,0],[0,0], linewidth=1, color='grey', label='one to one line')])
label = ['BNU-ESM', 'CanESM2', 'CESM1-CAM5', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'MIROC-ESM', 'NorESM1-M', 'one to one line']

leg1 = plt.legend(handels, label, loc=2)
leg2 = plt.legend(handels_1, label_1, loc=4)

plt.gca().add_artist(leg1)
plt.gca().add_artist(leg2)

# axis limits
plt.xlim((-750, 50))
plt.ylim((-750, 50))

# axis labels
plt.xlabel(r'Relationship-derived $\Delta C_{s, \tau}$ ($PgC$)')
plt.ylabel(r'Modelled $\Delta C_{s, \tau}$ ($PgC$)')

# save figure
#fig.savefig('paper_plots/proof_of_principal_cmip5_quadratic.pdf', bbox_inches='tight')
plt.close()

