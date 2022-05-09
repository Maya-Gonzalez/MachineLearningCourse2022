# Name: 
# COMP 347 - Machine Learning
# HW No. 2


# Libraries
import csv
import math
from telnetlib import X3PAD
from tkinter import Y
import pandas as pd
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import random

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
    #print(A)
    print(np.identity((A).shape[1]))
    I = np.identity((A).shape[1])
    w = LA.inv(A.T@A + (lam*I))@(A.T@y)
    return w

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
    deg = len(w) -1
    A = A_mat(x, deg)
    Aw = poly_func(A, w)
    wtd_rmse = np.sqrt(((y-Aw)**2).mean())
    return wtd_rmse

# 1b. Solve the least squares linear regression problem for the Athens 
#     temperature data.  Make sure to annotate the plot with the RMSE.
data = np.genfromtxt('athens_ww2_weather.csv', delimiter=',')
deg = 1
x_vals= data[1:, 8:9] # 8th col is min vals
y_vals = data[1:, 7:8] # 7th col is max vals
A = A_mat(x_vals, deg)

weights = LLS_Solve(x_vals, y_vals, deg) 
y_pred = poly_func(A, weights)
rmse = RMSE(x_vals,y_vals,weights)
plt.scatter( x_vals, y_vals, color = 'b', marker = (5, 1))
plt.plot(x_vals, y_pred, color = "r") 
plt.title('Temperature Variations in Athens, Greece (Nov. 1944 - Dec. 1945)')
plt.text(-2,43,'RMSE: {RMSE_val}'.format(RMSE_val = round(rmse,2)), size = 10, color = 'purple')
plt.xlabel("Minimum Temperature C°")
plt.ylabel("Maximum Temperature C°")
plt.show()
plt.clf()



# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

yosemite_data = pd.read_csv('Yosemite_Visits.csv', thousands=',', index_col =0)
yearIndex = [0,10,20,30]

allYears = []
for i in yearIndex:
    allYears.append( yosemite_data.iloc[i][:])
allYears = np.array(allYears)

year2018 = yosemite_data.loc[2018, :]
year2008 = yosemite_data.loc[2008, :]
year1998 = yosemite_data.loc[1998, :]
year1988 = yosemite_data.loc[1988, :]

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  A
# 
#     Additionally, create plots comparing the 
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).


deg = 11
monthRow= np.array([elem for elem in [1,2,3,4,5,6,7,8,9,10,11,12]])
x_vals = np.c_[monthRow]

A = A_mat(x_vals, deg)
col2018 = np.c_[year2018]
y_vals = np.c_[col2018, year2008, year1998, year1988]
weights = LLS_Solve(x_vals, allYears.T, deg ) 
y_pred = poly_func(A, weights)
func = LLS_func(x_vals, y_pred, weights, deg)
rmse = RMSE(x_vals,y_vals,weights)

def plotYears():
    plt.scatter(x_vals, year2018, color = 'cyan', marker = '*', s = 40, facecolors = 'none')
    plt.plot(x_vals, year2018, color = "cyan", label = '2018') 
    plt.scatter(x_vals, year2008, color = 'purple', marker = '*', s = 40, facecolors = 'none')
    plt.plot(x_vals, year2008, color = "purple", label = '2008') 
    plt.scatter(x_vals, year1998, color = '#88c999', marker = '*', s = 40, facecolors = 'none')
    plt.plot(x_vals, year1998, color = "#88c999", label = '1998') 
    plt.scatter(x_vals, year1988, color = 'hotpink', marker = '*', s = 40, facecolors = 'none') 
    plt.plot(x_vals, year1988, color = "hotpink", label = '1988') 

# # for loop to create 20 plots for deg 1-20
# for deg in range(1,21):
#     plotYears()
#     A = A_mat(x_vals, deg)
#     weights = LLS_Solve(x_vals, allYears.T, deg ) 
#     y_pred = poly_func(A, weights)
#     plt.plot(x_vals, np.mean(y_pred, axis = 1), color = "r", label = 'Degree-%d Fit' % deg)
#     handles, labels = plt.gca().get_legend_handles_labels()
#     plt.legend(handles, labels, loc = 'upper left')
#     plt.title('Yosemite Visitors with Degree-%d Least Squares (Var. Years)' % deg)
#     plt.show()

# used_years = []
# for i  in range(1,4):
#     # ensure year is not training year 
#     year = random.randint(1979,2018)
#     while (year in [2018,2008,1998,1988] or year in used_years):
#         year = random.randint(1979,2018)
    
#     used_years.append(year)
#     deg_list = []
#     rmse_list = []
#     func_list= []
#     testingYear = yosemite_data.loc[year,:]
#     testingYear = np.c_[testingYear]
    
#     for deg in range(1,21):
#         x = np.linspace(1,20, 1)
#         rmse = RMSE(x_vals,testingYear,weights)
#         func = LLS_func(x_vals, testingYear, weights, deg)
#         deg_list.append(deg)
#         rmse_list.append(rmse)
#         func_list.append(func)
#     plt.plot(deg_list, func_list, color = "blue", label = 'Training Error')
#     plt.scatter(deg_list, func_list, color = 'blue', marker = '*', s = 40) 
    
#     plt.plot(deg_list, rmse_list, color = "orange", label = 'RMSE')   
#     plt.scatter(deg_list, rmse_list, color = 'orange', marker = 'o',s = 40) 
       
#     handles, labels = plt.gca().get_legend_handles_labels()
#     plt.legend(handles, labels, loc = 'upper left')
#     plt.title('Training vs. RMSE for Yosemite Visitors %d' % year)
#     plt.show()

# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.  
   
deg = 5
lam = np.linspace(0.0,1.0,20)
monthCol = monthRow.T
newx_vals = np.hstack((monthCol, monthCol, monthCol, monthCol))
y_vals = np.hstack((year2018, year2008, year1998, year1988))


for i in range(20):
    # print(x_vals); print(y_vals)
    plotYears()
    ridge = LLS_ridge(newx_vals,y_vals,deg,lam[i])
    print(ridge.shape)
    print(ridge)
    plt.plot(x_vals, ridge, color = "r", label = 'Ridge, lambda= {:1.2f}'.format(lam[i]))
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc = 'upper left')
    plt.title('Yosemite Visitors Ridge Regression (%d Degree)' % deg)
    plt.show()

    

