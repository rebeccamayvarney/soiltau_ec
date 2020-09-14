#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:14:35 2020

@author: rmv203
"""
#%%

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
from rmv_cmip_analysis import combine_netCDF
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
# Peter plot scatter function
def plot_scat_basic_new(x,y):#,xlab,ylab,tit,panel):
    import numpy as np
    from pylab import plot, show, bar, legend, axes, xlabel, ylabel
    from pylab import title, savefig, axis, figure, subplot, semilogx, mean, exp, sqrt
    from pylab import log, arctan
    import scipy
    import matplotlib.pyplot as plt
         
    plot(x,y,'ro',label='Models')
    #xlabel(xlab,size=12)
    #ylabel(ylab,size=12)
    #title(tit,size=12,loc='left')
       
    return


#%%
# Peter lin reg function
def lin_reg_UU(x,y) :
    import numpy as np
    from numpy import arange
    from pylab import plot, legend, xlabel, ylabel
    from pylab import  figure, mean, exp, sqrt, sum
    from scipy import stats
    
# Based on "Least Squares fitting" equations from Math World website.
# This version checked against data on the Wikipedia "Simple Linear Regression" pages.
# It also calculates the +/- 1 sigma confidence limits in the regression [xfit,yband]
#
# IN THIS VERSION THE YBAND PREDICTION ERRORS ARE CALCULATED 
# ACCORDING TO DAVID PEARSON (AS USED IN COX ET AL., 2013)

    nx=len(x)
    ny=len(y)

    xm=mean(x)
    ym=mean(y)

    x2=x*x
    y2=y*y
    xy=x*y

    ssxx=sum(x2)-nx*xm*xm
    ssyy=sum(y2)-ny*ym*ym
    ssxy=sum(xy)-ny*xm*ym

    b=ssxy/ssxx
    a=ym-b*xm

    yf=a+b*x

    r2=ssxy**2/(ssxx*ssyy)
  
    e2=(y-yf)**2
    s2=sum(e2)/(nx-2)
  
    s=sqrt(s2)

    da=s*sqrt(1.0/nx+xm*xm/ssxx)
    db=s/sqrt(ssxx)


# Calculate confidence limits on fit (see Wikipedia page on "Simple Linear Regression")
    minx=min(x)-0.1*(max(x)-min(x))
    maxx=max(x)+0.1*(max(x)-min(x))
#    minx=min(x)
#    maxx=max(x)
    nfit=200
    dx=(maxx-minx)/nfit
    xfit=minx+dx*arange(0,nfit)
    yfit=a+b*xfit
    yband=np.zeros(nfit)

# David Pearson's formula for "Prediction Error"
    for n in range (0,nfit):
        yband[n]=sqrt(s2*(1.0+1.0/nx+(xfit[n]-xm)**2/ssxx))
    pass

    return yf,a,b,da,db,xfit,yfit,yband


#%%
# Peter EC Function
def EC_pdf_UU_reduced(x,y,x_obs,dx_obs): #xtitle, ytitle, tit, panel, key
    import numpy as np
    from numpy import arange
    from pylab import plot, legend, xlabel, ylabel, figure, savefig, title
    from pylab import mean, exp, sqrt, arange, xlim, ylim
    from pylab import subplot, subplots_adjust, fill_between
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.cm as cm
    
# Calculate mean and stdev of (equal model weight) prior
    mn_pr=mean(y)
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
    ###
    # a - best-fit intercept
    # b - gradient
    # xfit & yfit - best-fit straight-line
    
# Calculate PDF for IAV constraints
    x2=xfit
    nfitx=len(xfit)
    dx=x2[1]-x2[0]
    Px=x2
    Pi=3.142
    Px=1/sqrt(2*Pi*std**2) * exp(-((x2-mn)/(sqrt(2)*std))**2)

    miny=mn_pr-5*std_pr
    maxy=mn_pr+5*std_pr
    mfity=2000
    dy=(maxy-miny)/mfity
    y2=miny+dy*arange(0,mfity)

# Calculate "prior"
    Py_pr=y2
    Py_pr=1/sqrt(2*Pi*std_pr**2)*exp(-((y2-mn_pr)/(sqrt(2)*std_pr))**2)
    
# Calculate contours of probability in (x,y) space
    Pxy=np.zeros((nfitx,mfity))
    Pyx=np.zeros((mfity,nfitx))
    Py=np.zeros(mfity)
    Py_norm=0.0
    for m in range(0, mfity):
        Py[m]=0.0
        for n in range(0,nfitx):
            Py_given_x=1/sqrt(2*Pi*yband[n]**2) \
               * exp(-((y2[m]-yfit[n])/(sqrt(2)*yband[n]))**2)
            Pxy[n,m]=Px[n]*Py_given_x
            Pyx[m,n]=Pxy[n,m]
# Integrate over x to get Py
            Py[m]=Py[m]+Pxy[n,m]*dx
        pass
        Py_norm=Py_norm+Py[m]*dy
    pass

    print('NORMALISATION REQUIRED FOR POSTERIOR        = ',Py_norm)
# Normalise Py
    for m in range(0, mfity):
        Py[m]=Py[m]/Py_norm
    pass
    
    dum=np.argmax(Py)
    ybest=y2[dum]
   
    dum_pr=np.argmax(Py_pr)
    ybest_pr=y2[dum_pr]

# Calculate CDFs
    CDF=np.zeros(mfity)
    CDF_pr=np.zeros(mfity)
    CDF[0]=Py[0]*dy
    CDF_pr[0]=Py_pr[0]*dy
    for m in range(1,mfity):
        CDF[m]=CDF[m-1]+Py[m]*dy
        CDF_pr[m]=CDF_pr[m-1]+Py_pr[m]*dy
    pass
    
# Find 68% confidence limits    
    dum_up=CDF-0.84
    dum_840=dum_up**2
    dum_lo=CDF-0.16
    dum_160=dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_160))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_840))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py))
    y_best=y2[n_best]
    y_lo_1sd=y2[n_lo]
    y_hi_1sd=y2[n_hi]
    
# Find 95% confidence limits    
    dum_up=CDF-0.975
    dum_975=dum_up**2
    dum_lo=CDF-0.025
    dum_025=dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_025))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_975))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py))
    y_best=y2[n_best]
    y_lo=y2[n_lo]
    y_hi=y2[n_hi]
    
       
# Find 95% confidence limits for prior 
    dum_up=CDF_pr-0.975
    dum_975=dum_up**2
    dum_lo=CDF_pr-0.025
    dum_025=dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_025))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_975))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py_pr))
    y_best_pr=y2[n_best]
    y_lo_pr=y2[n_lo]
    y_hi_pr=y2[n_hi]
    
# Find 68% confidence limits for prior 
    dum_up=CDF_pr-0.84
    dum_840=dum_up**2
    dum_lo=CDF_pr-0.16
    dum_160=dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_160))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_840))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py_pr))
    y_best_pr_1sd=y2[n_best]
    y_lo_pr_1sd=y2[n_lo]
    y_hi_pr_1sd=y2[n_hi]
 
    
    print('Gaussian Prior (Mean)                       = ',y_best_pr)
    print('Gaussian Prior (68% confidence limits)      = [',y_lo_pr_1sd,'-',y_hi_pr_1sd,']')
    print('Gaussian Prior (95% confidence limits)      = [',y_lo_pr,'-',y_hi_pr,']')
    print('Emergent Constraint (Mean)                  = ',y_best)
    mean_ec_y_value = y_best
    print('Emergent Constraint (68% confidence limits) = [',y_lo_1sd,'-',y_hi_1sd,']')
    lower_ec_limit = y_lo_1sd
    upper_ec_limit = y_hi_1sd
    print('Emergent Constraint (95% confidence limits) = [',y_lo,'-',y_hi,']')
    print('Gradient of Emergent Relationship           = ',b,'+/-',db)
    
# Plot emergent relationship etc.
    # subplot(panel)
#    xd=[xr1,xbest]
#    yd1=[y_hi_1sd,y_hi_1sd]
#    yd2=[y_lo_1sd,y_lo_1sd]
#    yd_best=[y_best,ybest]
#    plot(xd,yd_best,'g--')
#    fill_between(xd,yd1,yd2, facecolor='lightgreen')
#    
#    ylim(yr1,yr2)
#    xlim(xr1,xr2)
#
#    plot([xbest,xbest],[yr1,yr2],'b--')
#    xd=[xlo,xhi]
#    yd1=[yr1,yr1]
#    yd2=[yr2,yr2]
#    fill_between(xd,yd1,yd2, facecolor='lightblue',label='Observation')
#    fill_between([xbest,xbest],yd1,yd1, facecolor='lightgreen',label='Emergent Constraint')
#    
## Plot contours of probability from linear regression
#    mult=[-1,1]
#    jconts=len(mult)
#    y1=np.zeros((jconts,nfitx))
#    for j in range (0,jconts):
#      ydum=yfit+mult[j]*yband
#      y1[j,:]=ydum
#    pass
#    fill_between(xfit,y1[0,:],y1[1,:], facecolor='lightgrey')
#    plot_scat_basic_new(x,y) #,xtitle,ytitle,tit, panel,
#    plot(xfit,yfit,'k--')
#    
#    #if (key==1):
#    legend(fontsize=9)


    return mean_ec_y_value, lower_ec_limit, upper_ec_limit, xfit, yfit


#%%
# analysis code

temperature_change_options = [1, 2, 3]
temperature_change_options_length = len(temperature_change_options)


for temp_option in range(0, temperature_change_options_length):
    min_temperature = temperature_change_options[temp_option] # selecting the temperature change
    
    print(min_temperature)


    #%%

    ### not forcing (0,0)

    # import saved data
    x_obs = np.loadtxt("saved_data/x_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    dx_obs = np.loadtxt("saved_data/dx_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    x_values = np.loadtxt("saved_data/combined_x_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    y_values = np.loadtxt("saved_data/combined_y_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")


    # ### forcing (0,0)
    #x_values = np.append(x_values, 0, axis=None)
    #y_values = np.append(y_values, 0, axis=None)

    # using ec function to obtain constrained values
    mean_ec_y_value, lower_ec_limit, upper_ec_limit, xfit, yfit = EC_pdf_UU_reduced(x_values, y_values, x_obs.item(), dx_obs.item())
    
    # setting as arrays
    mean_ec_y_value = np.array([mean_ec_y_value])
    lower_ec_limit = np.array([lower_ec_limit])
    upper_ec_limit = np.array([upper_ec_limit])

    print('new constrained mean:', mean_ec_y_value)
    print('new std', mean_ec_y_value-lower_ec_limit, upper_ec_limit-mean_ec_y_value)

    #print(xfit, xfit.shape, type(xfit))
    #print(yfit, yfit.shape, type(yfit))
    xfit = np.array([xfit])
    yfit = np.array([yfit])
    
    # saving EC values
    np.savetxt("saved_data/mean_EC_yvalue_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", mean_ec_y_value, delimiter=",")
    np.savetxt("saved_data/EC_lower_limit_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", lower_ec_limit, delimiter=",")
    np.savetxt("saved_data/EC_upper_limit_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", upper_ec_limit, delimiter=",")
    np.savetxt("saved_data/EC_xfit_"+str(min_temperature)+"degreewarming_cmip6cmip5.csv", xfit, delimiter=",")
    np.savetxt("saved_data/EC_yfit_"+str(min_temperature)+"degreewarming_cmip6cmip5.csv", yfit, delimiter=",")
    
    
    #%%
    # Plotting Probability density function of constraint
    
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
    'axes.labelsize': 22,
    'xtick.labelsize':22,
    'ytick.labelsize':22,
    'font.size':22,
    'text.usetex': False,
    "svg.fonttype": 'none'
    }
    
    plt.rcParams.update(params)
    
    
    y = y_values
    x = x_values
    
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
    
    # Calculate "prior"
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
    
    print('old standard dev: ', std_pr)
    print('old mean: ', mn_pr)
    
    # Plot PDF
    plt.xlim(min(y2),max(y2))
    plt.plot(y2,Py,'b-',label='Emergent Constraint',linewidth=3)
    plt.plot(y2,Py_pr,'k-',label='Gaussian Model Fit',linewidth=3)
    plt.xlabel(r'Global Modelled $\Delta C_{s, \tau}$ ($PgC$)')
    plt.ylabel(r'Probablity Density Per PgC')
    dum=np.argmax(Py)
    ybest=y2[dum]
    
    dum_pr=np.argmax(Py_pr)
    ybest_pr=y2[dum_pr]
    binny=min(y2)+(max(y2)-min(y2))*np.arange(16)/15.0
    #binny=[-500, -400, -300, -200, -100, 0, 100]
    #binny = [-400, -350, -300, -250, -200, -150, -100, -50, 0, 50, 100]
    #    binny=[100,200,300,400,500,600,700,800,900]
    print(y)
    n, bins, patches = plt.hist(y, bins=binny,\
                    normed=1, facecolor='grey',label='Model Range')
    plt.legend(fontsize=32, loc='upper right')
    
    # save figure
    if min_temperature == 0.5:
        fig.savefig('final_plots/cmip6_EC_PDF_05degreeswarming_oldSR.pdf', bbox_inches='tight')
        plt.close()
    else:
        fig.savefig('reviewer_plots/cmip6cmip5_EC_PDF_'+str(min_temperature)+'degreeswarming_CARDrh.pdf', bbox_inches='tight')
        plt.close()

