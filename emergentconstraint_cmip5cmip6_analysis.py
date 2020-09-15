#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  6 13:46:43 2020
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

Analysis Python Script for Varney et al. 2020 Nature Communications
- script finds the emergent constrained values and associated uncertainty bounds,
and the fit (xfit and yfit) which represents the emergent relationship amongst the models
"""

#%%
# analysis imports
import numpy as np
import numpy.ma as ma


#%%
# Peter linear Regression Function
def lin_reg_UU(x ,y) :
    import numpy as np
    from numpy import arange
    from pylab import plot, legend, xlabel, ylabel
    from pylab import  figure, mean, exp, sqrt, sum
    from scipy import stats
    
    # Based on "Least Squares fitting" equations from Math World website.
    # This version checked against data on the Wikipedia "Simple Linear Regression" pages.
    # It also calculates the +/- 1 sigma confidence limits in the regression [xfit,yband]

    # IN THIS VERSION THE YBAND PREDICTION ERRORS ARE CALCULATED 
    # ACCORDING TO DAVID PEARSON (AS USED IN COX ET AL., 2013)

    nx = len(x)
    ny = len(y)

    xm = mean(x)
    ym = mean(y)

    x2 = x*x
    y2 = y*y
    xy = x*y

    ssxx = sum(x2) - nx*xm*xm
    ssyy = sum(y2) - ny*ym*ym
    ssxy = sum(xy) - ny*xm*ym

    b = ssxy/ssxx
    a = ym - b*xm

    yf = a + b*x

    r2 = ssxy**2/(ssxx*ssyy)
  
    e2 = (y-yf)**2
    s2 = sum(e2)/(nx-2)
  
    s = sqrt(s2)

    da = s*sqrt(1.0/nx + xm*xm/ssxx)
    db = s/sqrt(ssxx)

    # Calculate confidence limits on fit (see Wikipedia page on "Simple Linear Regression")
    minx = min(x) - 0.1*(max(x) - min(x))
    maxx = max(x) + 0.1*(max(x) - min(x))
    nfit = 200
    dx = (maxx-minx)/nfit
    xfit = minx + dx*arange(0,nfit)
    yfit = a + b*xfit
    yband = np.zeros(nfit)

    # David Pearson's formula for "Prediction Error"
    for n in range (0,nfit):
        yband[n] = sqrt(s2*(1.0+1.0/nx+(xfit[n]-xm)**2/ssxx))
    pass

    return yf, a, b, da, db, xfit, yfit, yband


#%%
# Peter Emergent Constraint Function
def EC_pdf_UU_reduced(x, y, x_obs, dx_obs):
    import numpy as np
    from numpy import arange
    from pylab import plot, legend, xlabel, ylabel, figure, savefig, title
    from pylab import mean, exp, sqrt, arange, xlim, ylim
    from pylab import subplot, subplots_adjust, fill_between
    
    # Calculate mean and stdev of (equal model weight) prior
    mn_pr = mean(y)
    std_pr = np.std(y)
    
    # Observational Constraint
    mn = x_obs
    std = dx_obs
    xbest = x_obs
    xlo = x_obs - dx_obs
    xhi = x_obs + dx_obs
    
    # Define ranges for plots
    yr1 = min(y) - 0.1*(max(y) - min(y))
    yr2 = max(y) + 0.1*(max(y) - min(y))
    xr1 = min(x) - 0.1*(max(x) - min(x))
    xr2 = max(x) + 0.1*(max(x) - min(x))
    
    # Calculate best-fit straight-line between x & y
    yf, a, b, da, db, xfit, yfit, yband = lin_reg_UU(x, y)
    # a - best-fit intercept
    # b - gradient
    # xfit & yfit - best-fit straight-line
    
    # Calculate PDF for IAV constraints
    x2 = xfit
    nfitx = len(xfit)
    dx = x2[1] - x2[0]
    Px = x2
    Pi = 3.142
    Px = 1/sqrt(2*Pi*std**2) * exp(-((x2-mn)/(sqrt(2)*std))**2)

    miny = mn_pr-5*std_pr
    maxy = mn_pr+5*std_pr
    mfity = 2000
    dy = (maxy - miny)/mfity
    y2 = miny + dy*arange(0,mfity)

    # Calculate "prior"
    Py_pr = y2
    Py_pr = 1/sqrt(2*Pi*std_pr**2)*exp(-((y2-mn_pr)/(sqrt(2)*std_pr))**2)
    
    # Calculate contours of probability in (x,y) space
    Pxy = np.zeros((nfitx,mfity))
    Pyx = np.zeros((mfity,nfitx))
    Py = np.zeros(mfity)
    Py_norm = 0.0
    for m in range(0, mfity):
        Py[m] = 0.0
        for n in range(0,nfitx):
            Py_given_x = 1/sqrt(2*Pi*yband[n]**2) * exp(-((y2[m]-yfit[n])/(sqrt(2)*yband[n]))**2)
            Pxy[n,m] = Px[n]*Py_given_x
            Pyx[m,n] = Pxy[n,m]
        # Integrate over x to get Py
            Py[m] = Py[m] + Pxy[n,m]*dx
        pass
        Py_norm = Py_norm + Py[m]*dy
    pass
    print('NORMALISATION REQUIRED FOR POSTERIOR        = ',Py_norm)
    # Normalise Py
    for m in range(0, mfity):
        Py[m] = Py[m]/Py_norm
    pass
    
    dum = np.argmax(Py)
    ybest = y2[dum]
   
    dum_pr = np.argmax(Py_pr)
    ybest_pr = y2[dum_pr]

    # Calculate CDFs
    CDF = np.zeros(mfity)
    CDF_pr = np.zeros(mfity)
    CDF[0] = Py[0]*dy
    CDF_pr[0] = Py_pr[0]*dy
    for m in range(1,mfity):
        CDF[m] = CDF[m-1] + Py[m]*dy
        CDF_pr[m] = CDF_pr[m-1] + Py_pr[m]*dy
    pass
    
    # Find 68% confidence limits    
    dum_up = CDF - 0.84
    dum_840 = dum_up**2
    dum_lo = CDF - 0.16
    dum_160 = dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_160))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_840))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py))
    y_best = y2[n_best]
    y_lo_1sd = y2[n_lo]
    y_hi_1sd = y2[n_hi]
    
    # Find 95% confidence limits    
    dum_up = CDF - 0.975
    dum_975 = dum_up**2
    dum_lo = CDF - 0.025
    dum_025 = dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_025))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_975))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py))
    y_best = y2[n_best]
    y_lo = y2[n_lo]
    y_hi = y2[n_hi]
    
    # Find 95% confidence limits for prior 
    dum_up = CDF_pr - 0.975
    dum_975 = dum_up**2
    dum_lo = CDF_pr - 0.025
    dum_025 = dum_lo**2     
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_025))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_975))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py_pr))
    y_best_pr = y2[n_best]
    y_lo_pr = y2[n_lo]
    y_hi_pr = y2[n_hi]
    
    # Find 68% confidence limits for prior 
    dum_up = CDF_pr - 0.84
    dum_840 = dum_up**2
    dum_lo = CDF_pr - 0.16
    dum_160 = dum_lo**2   
    val, n_lo = min((val, idx) for (idx, val) in enumerate(dum_160))
    val, n_hi = min((val, idx) for (idx, val) in enumerate(dum_840))
    val, n_best = max((val, idx) for (idx, val) in enumerate(Py_pr))
    y_best_pr_1sd = y2[n_best]
    y_lo_pr_1sd = y2[n_lo]
    y_hi_pr_1sd = y2[n_hi]
 
    # print found values
    print('Gaussian Prior (Mean)                       = ',y_best_pr)
    print('Gaussian Prior (68% confidence limits)      = [',y_lo_pr_1sd,'-',y_hi_pr_1sd,']')
    print('Gaussian Prior (95% confidence limits)      = [',y_lo_pr,'-',y_hi_pr,']')
    print('Emergent Constraint (Mean)                  = ',y_best)
    print('Emergent Constraint (68% confidence limits) = [',y_lo_1sd,'-',y_hi_1sd,']')
    print('Emergent Constraint (95% confidence limits) = [',y_lo,'-',y_hi,']')
    print('Gradient of Emergent Relationship           = ',b,'+/-',db)

    mean_ec_y_value = y_best.copy()
    lower_ec_limit = y_lo_1sd.copy()
    upper_ec_limit = y_hi_1sd.copy()

    return mean_ec_y_value, lower_ec_limit, upper_ec_limit, xfit, yfit


#%%
# Analysis code

# global mean temperature change
temperature_change_options = [1, 2, 3]
temperature_change_options_length = len(temperature_change_options)

# loop through each global mean temperature change
for temp_option in range(0, temperature_change_options_length):
    min_temperature = temperature_change_options[temp_option] # selecting the temperature change
    
    print(min_temperature)

    #%%

    # import saved data
    x_obs = np.loadtxt("saved_data/x_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    dx_obs = np.loadtxt("saved_data/dx_obs_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    x_values = np.loadtxt("saved_data/combined_x_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")
    y_values = np.loadtxt("saved_data/combined_y_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", delimiter=",")


    # using EC function to obtain constrained values
    mean_ec_y_value, lower_ec_limit, upper_ec_limit, xfit, yfit = EC_pdf_UU_reduced(x_values, y_values, x_obs.item(), dx_obs.item())
    # converting to numpy arrays
    mean_ec_y_value = np.array([mean_ec_y_value])
    lower_ec_limit = np.array([lower_ec_limit])
    upper_ec_limit = np.array([upper_ec_limit])
    xfit = np.array([xfit])
    yfit = np.array([yfit])
    

    #%%

    print('new constrained mean:', mean_ec_y_value)
    print('new std', mean_ec_y_value-lower_ec_limit, upper_ec_limit-mean_ec_y_value)

    # saving EC values
    np.savetxt("saved_data/mean_EC_yvalue_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", mean_ec_y_value, delimiter=",")
    np.savetxt("saved_data/EC_lower_limit_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", lower_ec_limit, delimiter=",")
    np.savetxt("saved_data/EC_upper_limit_"+str(min_temperature)+"_degree_warming_cmip6cmip5.csv", upper_ec_limit, delimiter=",")
    np.savetxt("saved_data/EC_xfit_"+str(min_temperature)+"degreewarming_cmip6cmip5.csv", xfit, delimiter=",")
    np.savetxt("saved_data/EC_yfit_"+str(min_temperature)+"degreewarming_cmip6cmip5.csv", yfit, delimiter=",")
