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
# plt.show()


# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  A
# 
#     Additionally, create plots comparing the 
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).
    

yosemite_data = pd.read_csv('Yosemite_Visits.csv', index_col = 0)
# gives all value pairs for each month
Jan = yosemite_data['JAN'].str.split(',')
Feb = yosemite_data['FEB'].str.split(',')
Mar = yosemite_data['MAR'].str.split(',')
Apr = yosemite_data['APR'].str.split(',')
May = yosemite_data['MAY'].str.split(',')
Jun = yosemite_data['JUN'].str.split(',')
Jul = yosemite_data['JUL'].str.split(',')
Aug = yosemite_data['AUG'].str.split(',')
Sep = yosemite_data['SEP'].str.split(',')
Oct = yosemite_data['OCT'].str.split(',')
Nov = yosemite_data['NOV'].str.split(',')
Dec = yosemite_data['DEC'].str.split(',')

allMonths = []
allMonths.append(Jan)
allMonths.append(Feb)
years = [2018,2008,1998,1988]
yearIndex = [0,10,20,30]
print(Jan.iloc[10][0]) # prints x val for Jan of 2008

# now we need 12 vals for each of the 5 years
allYears = []
year2008 = []
year1998 = []
year1988 = []
# for each year in Years, get all the month valies
for i in (yearIndex): # index through yearIndex
    print(i)
    allYears.append(Jan.iloc[i][0])
    allYears.append(Feb.iloc[i][0])
    allYears.append(Mar.iloc[i][0])
    allYears.append(Apr.iloc[i][0])
    allYears.append(May.iloc[i][0])
    allYears.append(Jun.iloc[i][0])
    allYears.append(Jul.iloc[i][0])
    allYears.append(Aug.iloc[i][0])
    allYears.append(Sep.iloc[i][0])
    allYears.append(Oct.iloc[i][0])
    allYears.append(Nov.iloc[i][0])
    allYears.append(Dec.iloc[i][0])
    # # continue for rest of 12 months
    # year2008.append(Jan.iloc[i][0])
    # year2008.append(Feb.iloc[i][0])
    # year2008.append(Mar.iloc[i][0])
    # year2008.append(Apr.iloc[i][0])
    # year2008.append(May.iloc[i][0])
    # year2008.append(Jun.iloc[i][0])
    # year2008.append(Jul.iloc[i][0])
    # year2008.append(Aug.iloc[i][0])
    # year2008.append(Sep.iloc[i][0])
    # year2008.append(Oct.iloc[i][0])
    # year2008.append(Nov.iloc[i][0])
    # year2008.append(Dec.iloc[i][0])

    # year1998.append(Jan.iloc[i][0])
    # year1998.append(Feb.iloc[i][0])
    # year1998.append(Mar.iloc[i][0])
    # year1998.append(Apr.iloc[i][0])
    # year1998.append(May.iloc[i][0])
    # year1998.append(Jun.iloc[i][0])
    # year1998.append(Jul.iloc[i][0])
    # year1998.append(Aug.iloc[i][0])
    # year1998.append(Sep.iloc[i][0])
    # year1998.append(Oct.iloc[i][0])
    # year1998.append(Nov.iloc[i][0])
    # year1998.append(Dec.iloc[i][0])

    # year1988.append(Jan.iloc[i][0])
    # year1988.append(Feb.iloc[i][0])
    # year1988.append(Mar.iloc[i][0])
    # year1988.append(Apr.iloc[i][0])
    # year1988.append(May.iloc[i][0])
    # year1988.append(Jun.iloc[i][0])
    # year1988.append(Jul.iloc[i][0])
    # year1998.append(Aug.iloc[i][0])
    # year1988.append(Sep.iloc[i][0])
    # year1988.append(Oct.iloc[i][0])
    # year1988.append(Nov.iloc[i][0])
    # year1988.append(Dec.iloc[i][0])
    # for j in range(12): 
    #     print(Jan.iloc[yearIndex][0])
    #     # append to year array
    #     year2018.append(Jan.iloc[i][0])
    #     year2018.append(Dec.iloc[i][0])
    #     # continue for rest of 12 months
    #     year2008.append(Jan.iloc[i][0])
    #     year2008.append(Dec.iloc[i][0])
print(allYears)
year2018 = allYears[0:12]
year2008 = allYears[12:24]
year1998 = allYears[24:36]
year1988 = allYears[36:48]
print(year2018)
print(year2008)
print(year1998)
print(year1988)
# yosemite_data['jan'] = yosemite_data['JAN'].str.split(',')
# print('here ye :\n',yosemite_data.loc[2018][:])
N = len(yosemite_data)
# for i in range(N):
#     print(Jan.iloc[i][0])
# print('here ye :\n',Jan.iloc[0][0])

# year2018 = yosemite_data.loc[2018]
# # print(yosemite_data.JAN.head(3))
# print(year2018)
# print(yosemite_data)
# deg = 1
# A = A_mat(x_vals, deg)



# x_vals_2018 = [1,2,3,4,5,6,7,8,9,10,11,12]
# y_vals_2018 = yosemite_data[0]
# plt.scatter( x_vals, y_vals, color = 'b', marker = (5, 1))
# plt.plot(x_vals_2018, y_vals_2018, color = "b") 
# plt.plot(x_vals, y_pred, color = "r") 
# plt.plot(x_vals, y_pred, color = "g") 
# plt.plot(x_vals, y_pred, color = "p") 
# plt.title('Yosemite Visitors in Various Years')
# plt.show()

# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.  
   
