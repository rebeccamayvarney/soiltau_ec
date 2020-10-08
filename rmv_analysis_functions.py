#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Rebecca Varney, University of Exeter (rmv203@exeter.ac.uk)

"""

#%%

# Analysis
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from scipy import stats
import scipy as sp
from numpy import arange
from pylab import plot, legend, xlabel, ylabel
from pylab import  figure, mean, exp, sqrt, sum
from pylab import subplot, subplots_adjust, fill_between
import sys
from scipy import stats

# Iris imports
import iris
import iris.coord_categorisation
import iris.analysis.cartography
import iris.quickplot as qplt
import iris.plot as iplt
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes
from iris.util import unify_time_units


#%%
def combine_netCDF_variable(directory, variable):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (variable input - OBSERVATIONAL DATASETS)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)

            # Load each file named variable as a cube
            cube = iris.load_cube(newlist[i], variable)

            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    unify_time_units(Cubelist)
    equalise_attributes(Cubelist)

    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()

    return new_cube


#%%
def combine_netCDF_model(directory, model):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (model input - STANDARD MODEL)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)

            # Load cube
            cube = iris.load_cube(newlist[i])

            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    unify_time_units(Cubelist)
    equalise_attributes(Cubelist)

    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()

    return new_cube


#%%
def combine_netCDF_cmip5(directory, variable, model):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (CMIP5 models)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)

            # Load each file named variable as a cube
            cube = iris.load_cube(newlist[i], variable)

            # CRUDE WORKAROUND TO REMOVE OVERLAPPING COORDINATE
            if model == 'HadGEM2-ES':
                if i == 3: # change to 9 if timeseries plots
                    cube = cube[0:-1]

            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    iris.util.unify_time_units(Cubelist)
    equalise_attributes(Cubelist)
    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()

    return new_cube


#%%
def combine_netCDF_rh_cmip6(directory, model):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (CMIP6 models, rh variable)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)
            
            # Load cube
            cube = iris.load_cube(newlist[i])
            
            # matching all standard names
            cube.standard_name = 'heterotrophic_respiration_carbon_flux'

            # matching cube metadata
            if i == 0:
                metadata1 = cube.metadata
            else:
                cube.metadata = metadata1
            
            # creating latitude and longitude bounds
            if model=='IPSL-CM6A-LR' or model=='CNRM-ESM2-1':
                cube.coord('latitude').guess_bounds()
                cube.coord('longitude').guess_bounds()
                
            # removing time attributes
            if model=='IPSL-CM6A-LR':
                cube.coord('time').attributes.pop('time_origin')
            
            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    unify_time_units(Cubelist)
    equalise_attributes(Cubelist)
    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()
    
    for cube in Cubelist:
        lon_bounds = Cubelist[0].coord('longitude').bounds
        cube.coord('longitude').bounds = lon_bounds

    for i, cube in enumerate(Cubelist):
        if cube.coord('time').units == Cubelist[0].coord('time').units:
            pass
        else:
            print(i)

    return new_cube


#%%
def combine_netCDF_cSoil_cmip6(directory, model):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (CMIP6 models, cSoil variable)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):
        with warnings.catch_warnings():
        
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)
            
            # Load cube
            cube = iris.load_cube(newlist[i])
            
            # matching all standard names
            cube.standard_name = 'soil_carbon_content'
            
            # creating latitude and longitude bounds
            if model=='IPSL-CM6A-LR' or model=='CNRM-ESM2-1':
                cube.coord('latitude').guess_bounds()
                cube.coord('longitude').guess_bounds()
            
            # removing time attributes
            if model=='IPSL-CM6A-LR':
                cube.coord('time').attributes.pop('time_origin')
            
            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    unify_time_units(Cubelist)
    equalise_attributes(Cubelist)
    
    for cube in Cubelist:
        lon_bounds = Cubelist[0].coord('longitude').bounds
        cube.coord('longitude').bounds = lon_bounds

    for i, cube in enumerate(Cubelist):
        if cube.coord('time').units == Cubelist[0].coord('time').units:
            pass
        else:
            print(i)

    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()

    return new_cube


#%%
def combine_netCDF_cmip6(directory, model):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (CMIP6 models, not rh & cSoil)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)
            
            # Load cube
            cube = iris.load_cube(newlist[i])

            # remove latitude & longitude attributes
            cube.coord('latitude').attributes = {}
            cube.coord('longitude').attributes = {}  
            
            # creating latitude and longitude bounds
            if model=='IPSL-CM6A-LR' or model=='CNRM-ESM2-1':
                cube.coord('latitude').guess_bounds()
                cube.coord('longitude').guess_bounds()
         
            # CESM2 bound issue fix
            if (model=='CESM2') & (i==0):
                lat_data = cube.coord('latitude').points
                lon_data = cube.coord('longitude').points
                lat_bounds = cube.coord('latitude').bounds
                lon_bounds = cube.coord('longitude').bounds
            elif (model=='CESM2') & (i>0):
                cube.coord('latitude').points = lat_data
                cube.coord('longitude').points = lon_data
                cube.coord('latitude').bounds = lat_bounds
                cube.coord('longitude').bounds = lon_bounds
    
            # removing time attributes
            if model=='IPSL-CM6A-LR':
                cube.coord('time').attributes.pop('time_origin')
            
            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    unify_time_units(Cubelist)
    equalise_attributes(Cubelist)

    for cube in Cubelist:
        lon_bounds = Cubelist[0].coord('longitude').bounds
        cube.coord('longitude').bounds = lon_bounds

    for i, cube in enumerate(Cubelist):
        if cube.coord('time').units == Cubelist[0].coord('time').units:
            pass
        else:
            print(i)
            
    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()

    return new_cube


#%%
def combine_netCDF_observations_temp(directory, variable):
    """ 
    Function combines netCDF files to create one file for entire time period considered
    (observational temperature dataset)

    directory - where data sits
    variable - being considered (must be the name used in the netCDF file)
    model - name of model
    """

    # Make a list of the files in the above folder to loop through
    list_files = glob.glob(directory)
    list_files = np.array(list_files)
    newlist = np.sort(list_files)

    # Make a cubelist to add each file (cube) to
    Cubelist = iris.cube.CubeList([])

    # loop for each file in newlist
    for i in range(0, len(newlist)):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            warnings.simplefilter('ignore', UserWarning)

            # Load each file named variable as a cube
            cube = iris.load_cube(newlist[i], variable)

            # key attributes error fix
            for key in list(cube.attributes.keys()):
                del cube.attributes[key]

            time_coord = iris.coords.DimCoord(i, standard_name='time', units='year')
            cube = iris.util.new_axis(cube)
            cube.add_dim_coord(time_coord,0)

            # Append this cube to the cubelist
            Cubelist.append(cube)

    # matching attributes
    unify_time_units(Cubelist)
    equalise_attributes(Cubelist)

    # Concatenate each cube in cubelist together to make one data file (cube)
    new_cube = Cubelist.concatenate_cube()

    return new_cube


#%%
def open_netCDF(new_cube):
    """ Function adds time coordinates to cube """

    iris.coord_categorisation.add_year(new_cube, 'time', name='year') # add year
    iris.coord_categorisation.add_month(new_cube, 'time', name ='month') # add month
    iris.coord_categorisation.add_month(new_cube, 'time', name ='decade') # add month

    return new_cube


#%%
def define_attributes(new_cube):
    """ Function returns the time, latitude and longitude coordinates of the cube """

    time_cmip5 = new_cube.coord('time').points # Defining the time variable
    lats_cmip5 = new_cube.coord('latitude').points # Defining the lat variable
    lons_cmip5 = new_cube.coord('longitude').points # Defining the lon variable

    return time_cmip5, lats_cmip5, lons_cmip5


#%%
def select_time(new_cube, lower, upper):
    """ Function that selects the time period in years """

    sliced_cube = new_cube.extract(iris.Constraint(year=lambda y: lower<=y<=upper))

    return sliced_cube


#%%
def time_average(new_cube):
    """ Function calculates time average for cube in current time unit """

    time_average_cube = new_cube.collapsed('time', iris.analysis.MEAN)

    return time_average_cube


#%%
def annual_average(new_cube):
    """ Function calculates annual average for cube """

    annual_average_cube = new_cube.aggregated_by('year', iris.analysis.MEAN)

    return annual_average_cube


#%%
def decadal_average(new_cube):
    """ Function calculates annual average for cube """

    decadal_average_cube = new_cube.aggregated_by('decade', iris.analysis.MEAN)

    return decadal_average_cube


#%%
def numpy_to_cube(np_array, similar_cube, dimensions):
    """
    Function that converts a 1, 2, or 3 dimensional numpy array to a cube.
    (Inverse is array = cube.data)
    """

    new_cube = iris.cube.Cube.copy(similar_cube) # copy similar cube

    # time, lat, lon
    if dimensions == 3:
        new_cube.data[:,:,:] = np.nan # convert new cube entries to nan
        new_cube.data[:,:,:] = np_array # fill with numpy array data

    # lat, lon
    elif dimensions == 2:
        new_cube.data[:,:] = np.nan # convert new cube entries to nan
        new_cube.data[:,:] = np_array # fill with numpy array data

    # either time, lat or lon only
    elif dimensions == 1:
        new_cube.data[:] = np.nan # convert new cube entries to nan
        new_cube.data[:] = np_array # fill with numpy array data

    # return the numpy array, failed to convert to a cube
    else:
        print('failed to convert')
        new_cube = np_array

    return new_cube


#%%
def regrid_model(cube, regridcube):
    """ Function regrids cube to the dimensions of regridcube """

    regridcube.coord('latitude').standard_name = 'latitude'
    regridcube.coord('longitude').standard_name = 'longitude'

    model_units = cube.coord('latitude').units
    regridcube.coord('latitude').units = model_units
    regridcube.coord('longitude').units = model_units

    new_model_cube = cube.regrid(regridcube, iris.analysis.Linear())

    return new_model_cube


#%%
def area_average(cube, region):
    """
    Function to create weighted area average, by collapse a cube to a weighted area average over a specified region,
    global: region = [0, 360, -90,  90]
    """
    
    # Specify the latitudes and longitudes starting from the smallest number to largest or in latitude and longitude from south to north and east to west
    lon1, lon2, lat1, lat2 = region[0], region[1], region[2], region[3] 
    # Then intersect the data at these points
    cube = cube.intersection(longitude=(lon1, lon2),latitude=(lat1, lat2))

    #cube.coord('latitude').guess_bounds()
    #cube.coord('longitude').guess_bounds()

    #  area weighting
    weights = iris.analysis.cartography.area_weights(cube)
    # Average that area by latitude and longitudes by the weighted mean
    cube = cube.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=weights)

    return cube


#%%
def area_average_obs(cube, region, model_units):
    """
    Observational dataset version - 
    Function to create weighted area average, by collapse a cube to a weighted area average over a specified region,
    global: region = [0, 360, -90,  90]
    """
    
    # Specify the latitudes and longitudes starting from the smallest number to largest or in latitude and longitude from south to north and east to west
    lon1, lon2, lat1, lat2 = region[0], region[1], region[2], region[3]

    print(cube.coord('latitude').var_name)
    print(cube.coord('latitude').units.modulus)
    cube.coord('latitude').units = model_units
    cube.coord('longitude').units = model_units
    print(cube.coord('latitude').units.modulus)

    # Then intersect the data at these points
    cube = cube.intersection(longitude=(lon1, lon2),latitude=(lat1, lat2))

    # cube.coord('latitude').guess_bounds()
    # cube.coord('longitude').guess_bounds()

    # area weighting
    weights = iris.analysis.cartography.area_weights(cube)
    # Average that area by latitude and longitudes by the weighted mean
    cube = cube.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=weights)

    return cube


#%%
def global_total(cubein, landfrac=None, latlon_cons=None):
    '''
    Function to calculate area weighted sum of non missing data: written by Eleanor Burke
 
    Input
    ----
    * cube: :class:`iris.cube.Cube`
        variable for which total amount needs to be calculated

    * landfrac: :class:`iris.cube.Cube`
        landfrac mask so only the land fraction of the coastal points are included in the global totals

    * latlon_cons: :class:'iris.Constraint'
          used to extract the land frac sub-region for analysis if the varaible cube and land frac are on different grids      

    Returns
    -------
    * cube_areaweight: :class:`iris.cube.Cube`
        this cube is the global total

    '''

    cube = cubein.copy()
    if landfrac is not None:
        try:
            cube.data = cube.data * landfrac.data
        except:
            landfrac = landfrac.extract(latlon_cons)
            cube.data = cube.data * landfrac.data

    if cube.coord('latitude').bounds is not None:
        pass
    else:
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()

    weights = iris.analysis.cartography.area_weights(cube)

    cube_areaweight = cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=weights)/1e12

    return cube_areaweight


#%%
def global_total_percentage(cubein, landfrac=None, latlon_cons=None):
    '''
    Function to calculate area weighted sum of non missing data: written by Eleanor Burke,
    edited by Rebecca Varney
 
    Input
    ----
    * cube: :class:`iris.cube.Cube`
        variable for which total amount needs to be calculated

    * landfrac: :class:`iris.cube.Cube`
        landfrac mask so only the land fraction of the coastal points are included in the global totals

    * latlon_cons: :class:'iris.Constraint'
          used to extract the land frac sub-region for analysis if the varaible cube and land frac are on different grids      

    Returns
    -------
    * cube_areaweight: :class:`iris.cube.Cube`
        this cube is the global total

    '''

    cube = cubein.copy()
    if landfrac is not None:
        try:
            cube.data = cube.data * (landfrac.data/100)
        except:
            landfrac = landfrac.extract(latlon_cons)
            cube.data = cube.data * (landfrac.data/100)

    if cube.coord('latitude').bounds is not None:
        pass
    else:
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()

    weights = iris.analysis.cartography.area_weights(cube)

    cube_areaweight = cube.collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=weights)/1e12

    return cube_areaweight


#%%
def obtaining_Cs_q10(q10_parameter, observational_Cs, temp_change):
    ''' delta Cs q10 calculation '''
    
    ln = np.log(q10_parameter)
    delta_Cs_q10 = observational_Cs*(np.exp(-(ln/10)*temp_change) - 1)    
    
    return delta_Cs_q10


#%%
def obtaining_fractional_deltaCs_q10(q10_parameter, temp_change):
    ''' deltaCs/Cs,0 q10 calculation '''
    
    ln = np.log(q10_parameter)
    fractional_delta_Cs_q10 = (np.exp(-(ln/10)*temp_change) - 1)    
    
    return fractional_delta_Cs_q10


#%%
def obtaining_fractional_deltaCs_q10_logged(q10_parameter, temp_change):
    ''' deltaCs/Cs,0 logged q10 calculation '''
    
    ln = np.log(q10_parameter)
    fractional_delta_Cs_q10 = -(ln/10)*temp_change
    
    return fractional_delta_Cs_q10


#%%
def lin_reg_UU(x ,y) :
    ''' Linear Regression Function '''
    
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
def EC_pdf_UU_reduced(x, y, x_obs, dx_obs):
    '''
    Emergent Constraint Function
    - finds an emergent relationship between x and y (xfit, yfit)
    - finds emergent constraint on y values (mean_ec_y_value), with corresponding uncertainty bounds (lower_ec_limit, upper_ec_limit),
    where the uncertainty is dependent on the quality of the emergent relationship and the observational uncertainty (dx_obs)

    x - range of model x values
    y - range of model y values
    x_obs - mean calculated using observations
    dx_obs - associated observational uncertainty
    '''
    
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
    
    # Calculate PDF for observational constraint
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

    return mean_ec_y_value, lower_ec_limit, upper_ec_limit
