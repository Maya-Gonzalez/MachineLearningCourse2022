# Name: 
# COMP 347 - Machine Learning
# HW No. 2


# Libraries
import csv
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
    # parameters m,b (weight vector)
    # outputs the weights?? or the coefficients?
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit."""
    A = A_mat(x, deg)
    w = (LA.inv(A.T@A))@A.T@y
    # print('LLS Solve:\n',w)
    return(w)

def LLS_ridge(x,y,deg,lam):
    # does this output the weights/parameters?
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
    # coefficients are the numbers in front of x 
    y = data@coeffs
    # print('y = \n', y)
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

    weights = LLS_Solve(x, y, deg)
    N = len(x)
    # mse = (1/N)* (LA.norm(y - x@w) **2)
    # rmse = np.sqrt(mse)
    wtd_rmse = np.sqrt(((y-x)**2).mean())
    # total_error += LA.norm((y_pred - y))**2
    print(wtd_rmse)
    return wtd_rmse

# 1b. Solve the least squares linear regression problem for the Athens 
#     temperature data.  Make sure to annotate the plot with the RMSE.



data = np.genfromtxt('athens_ww2_weather.csv', delimiter=',')
# numRows = len(data)  # deg of polynomial fit must be less than numRows
deg = 1
x_vals= data[1:, 8:9] # 8th col is min vals
y_vals = data[1:, 7:8] # 7th col is max vals
A = A_mat(x_vals, deg)
weights = LLS_Solve(x_vals,y_vals, deg)
y_pred = LLS_func(x_vals, y_vals, weights, deg)
y_pred = poly_func(A, weights)
f_func = LLS_func(x_vals, y_vals, weights, deg)
print(f_func)
rmse = RMSE(x_vals,y_vals,weights)
print(rmse)

# x_line = np.linspace(x_vals.min(), x_vals.max(), len(x_vals))
# regression_line = LLS_func(x_vals, y_vals, weights, deg)
# print(regression_line)

# plt.plot(x_line, weights[0]*x_vals + weights[1], color = 'r')
# plt.plot(x_line, regression_line, color = 'r')
# plt.plot(x_vals, w*x_vals, c = 'red')

plt.scatter( x_vals, y_vals, color = 'g', marker = 'o', s = 30)
plt.title('Training Data')
plt.plot(x_vals, y_pred, color = "g") 
plt.show()

