# Name: 
# COMP 347 - Machine Learning
# HW No. 2


# Libraries
import csv
import math
from tkinter import Y
import pandas as pd
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------


# Problem 1 - Linear Regression with Athens Temperature Data
#------------------------------------------------------------------------------

# In the following problem, implement the solution to the least squares problem.

# 1a. Complete the followingfunctions:

def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input data.
       deg: degree of the polynomial fit."""
    N = len(x)
    ones = np.ones(N, dtype=float )
    A = np.c_[np.power(x, deg)]
    for i in range(deg-1):
        newCol = np.power(x, (deg-1) -i)
        A = np.c_[A, newCol]
    A = np.c_[A, ones]
    return A

def LLS_Solve(x,y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit."""
    A = A_mat(x, deg)
    w = (LA.inv(A.T@A))@A.T@y
    return(w)

def LLS_ridge(x,y,deg,lam):
    """Find the vector w that solves the ridge regresssion problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression."""
    A = A_mat(x, deg)
    ATA = A.T@A
    I = np.identity(len(ATA))
    lambdaI = np.dot(lam, I)
    invAdd = LA.inv(np.add(ATA, lambdaI))
    w = invAdd@(A.T@y)
    print('ridge regression:\n',w)
    return(w)

def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial."""
    y = data@coeffs
    return(y)

def LLS_func(x,y,w,deg):
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    A = A_mat(x, deg)
    weights = LLS_Solve(x, y, deg)
    y_pred = poly_func(A, weights)
    N = len(A)
    f = (1/N) * (LA.norm(y_pred - y) **2)
    return f

def RMSE(x,y,w):
    """Compute the root mean square error.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights."""
    wtd_rmse = np.sqrt(((y-x)**2).mean())
    return wtd_rmse

# 1b. Solve the least squares linear regression problem for the Athens 
#     temperature data.  Make sure to annotate the plot with the RMSE.



data = np.genfromtxt('athens_ww2_weather.csv', delimiter=',')
# numRows = len(data)  # deg of polynomial fit must be less than numRows
deg = 1
x_vals= data[1:, 8:9] # 8th col is min vals
y_vals = data[1:, 7:8] # 7th col is max vals
A = A_mat(x_vals, deg)

weights = LLS_Solve(x_vals, y_vals, deg) 
ridge_vector = LLS_ridge(x_vals, y_vals, deg, lam = .5)
y_pred = poly_func(A, weights)
f_func = LLS_func(x_vals, y_vals, weights, deg)
rmse = RMSE(x_vals,y_vals,weights)


plt.scatter( x_vals, y_vals, color = 'b', marker = (5, 1))
plt.plot(x_vals, y_pred, color = "r") 
plt.title('Temperature Variations in Athens, Greece (Nov. 1944 - Dec. 1945)')
plt.text(-2,43,'RMSE: {RMSE_val}'.format(RMSE_val = round(rmse,2)), size = 10, color = 'purple')
plt.show()


# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  A
# 
#     Additionally, create plots comparing the 
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).
    
    
# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.  
   
