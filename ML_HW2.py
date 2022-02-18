# Name: 
# COMP 347 - Machine Learning
# HW No. 2


# Libraries
import pandas as pd
#------------------------------------------------------------------------------


# Problem 1 - Linear Regression with Athens Temperature Data
#------------------------------------------------------------------------------

# In the following problem, implement the solution to the least squares problem.

# 1a. Complete the following functions:

def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input data.
       deg: degree of the polynomial fit."""

       #return FIXME

def LLS_Solve(x,y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit."""
    #return FIXME

def LLS_ridge(x,y,deg,lam):
    """Find the vector w that solves the ridge regresssion problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression."""
    #return FIXME

def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial."""
    #return FIXME

def LLS_func(x,y,w,deg):
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    #return FIXME

def RMSE(x,y,w):
    """Compute the root mean square error.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights."""
    #return FIXME


# 1b. Solve the least squares linear regression problem for the Athens 
#     temperature data.  Make sure to annotate the plot with the RMSE.
data = pd.read_csv('/Users/mayagonzalez/Desktop/Academic/ML_HW/athens_ww2_weather.csv')
numRows = len(data.index)  # should be 358
print(numRows)
#A_mat(data, )

# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  Additionally, create plots comparing the 
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).
    
    
# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.  
   
