# Name: 
# COMP 347 - Machine Learning
# HW No. 3

# Libraries
import csv
import math
from telnetlib import X3PAD
from tkinter import W, Y
import pandas as pd
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import random
#------------------------------------------------------------------------------

# Problem 1 - Gradient Descent Using Athens Temperature Data
#------------------------------------------------------------------------------
# For this problem you will be implementing various forms of gradient descent  
# using the Athens temperature data.  Feel free to copy over any functions you 
# wrote in HW #2 for this.  WARNING: In order to get gradient descent to work
# well, I highly recommend rewriting your cost function so that you are dividing
# by N (i.e. the number of data points there are).  Carry this over into all 
# corresponding expression where this would appear (the derivative being one of them).

data = np.genfromtxt('athens_ww2_weather.csv', delimiter=',')


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
    temp = (LA.inv(A.T@A))@A.T
    w = temp@y

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
    N = len(x)
    f = (1/N) * (LA.norm(A@w - y) **2)

    # f = LA.norm(A@x-y) ** 2
    return f
# 1a. Fill out the function for the derivative of the least-squares cost function:
def LLS_deriv(x,y,w,deg):
    """Computes the derivative of the least squares cost function with input
    data x, output data y, coefficient vector w, and deg (the degree of the
    least squares model)."""
    A = A_mat(x, deg) 
    N = len(x)
    if len(np.shape(A)) == 1: np.outer(A,A)
    return (2/N) * A.T@(A@w-y)


# 1b. Implement gradient descent as a means of optimizing the least squares cost
#     function.  Your method should include the following:
#       a. initial vector w that you are optimizing,
#       b. a tolerance K signifying the acceptable derivative norm for stopping
#           the descent method,
#       c. initial derivative vector D (initialization at least for the sake of 
#           starting the loop),
#       d. an empty list called d_hist which captures the size (i.e. norm) of your
#           derivative vector at each iteration, 
#       e. an empty list called c_hist which captures the cost (i.e. value of
#           the cost function) at each iteration,
#       f. implement backtracking line search as part of your steepest descent
#           algorithm.  You can implement this on your own if you're feeling 
#           cavalier, or if you'd like here's a snippet of what I used in mine:
#
#                eps = 1    
#                m = LA.norm(D)**2
#                t = 0.5*m
#                while LLS_func(a_min, a_max, w - eps*D, 1) > LLS_func(a_min, a_max, w, 1) - eps*t:
#                    eps *= 0.9

#       Plot curves showing the derivative size (i.e. d_hist) and cost (i.e. c_hist)
#       with respect to the number of iterations.
# 


x = data[1:, 8:9] # min
y = data[1:, 7:8] # max
deg = 1
K = 0.01  
w = np.array([[100],[-100]])
D = np.array([[-1], [1]])


d_hist = []
c_hist = []

x = data[1:, 8:9] # min
y = data[1:, 7:8] # max


count = 0
def backtrackingLineSearch(w, D):
    eps = 1    
    m = LA.norm(D)**2
    t = 0.5*m
    while LLS_func(x, y, w - eps*D, 1) > LLS_func(x, y, w, 1) - eps*t:
        eps *= 0.9
    return eps

def min_gd(K, w, D):
    global count
    # global w_new
    while(LA.norm(D) >= K):
        # if statement --> pass in w_new to bLS fn
        # if (count == 0):
        #     w_new = w
        eps = backtrackingLineSearch(w, D)
        w = w - (eps*D)
        D = LLS_deriv(x, y, w, deg)

        d_hist.append(LA.norm(D))
        c_hist.append(LLS_func(x,y,w,deg))
        count+=1

        print('\n', count)
        # print('eps = ', eps)
        # print('D = ', D)
        # print('eps*D = ', eps*D)
        # print('w = ', w)
        # print('w_new = ', w)
        # print('w = ', w)
        # print('D norm' , LA.norm(D))
        # print('C val', LLS_func(x,y,w,deg))
        # return min_gd(K, w, D)
    return w
# min_gd(K, w, D)


# plot will  be spikey, batch gradients descent and schcastic will be smooth,
# # randomize will make loinger process cs cutting complexity f each iterastion
# newtons limits num of iterations, but complexity of each iteration is bigger
# converge faster, find optimaum in more math

# UNCOMMENT ME TO PLOT 1b. !!!!!!
# iterations = np.linspace(1,count, count)
# plt.plot(iterations, d_hist, color = 'b', marker = (5, 1))
# plt.title('Derivative Size with Respect to Iterations')
# plt.ylabel("Derivative Size")
# plt.xlabel("Iterations")
# plt.show()
# plt.clf()

# plt.plot(iterations, c_hist, color = 'b', marker = (5, 1))
# plt.title('Cost Size with Respect to Iterations')
# plt.ylabel("Cost Size")
# plt.xlabel("Iterations")
# plt.show()
# plt.clf()

# 1c. Repeat part 1b, but now implement mini-batch gradient descent by randomizing
#     the data points used in each iteration.  Perform mini-batch descent for batches
#     of size 5, 10, 25, and 50 data points.  For each one, plot the curves
#     for d_hist and c_hist.  Plot these all on one graph showing the relationship
#     between batch size and convergence speed (at least for the least squares 
#     problem).  Feel free to adjust the transparency of your curves so that 
#     they are easily distinguishable.

#for loop that goes through range (5,10,25,50)

def mini_batch_gd(x_mini, y_mini, K, w, D):
    # with fixed step size
    global miniCount
    miniCount = 0
    eps = .9
    while(LA.norm(D) >= K):
        #FIXME: why does the derivative keep increasing --> goes to infinity 
        # need deriv to get closer to 0 
        w = w - (eps*D)
        D = LLS_deriv(x_mini, y_mini, w, deg)
        

        miniCount+=1
        d_hist.append(LA.norm(D))
        c_hist.append(LLS_func(x,y,w,deg))
        print(miniCount)
        print(LA.norm(D))

    return w



numDataPoints = [5,10,25,50]
for i in numDataPoints:
    randomDataListX = []
    randomDataListY = []
    for j in range(i):
        # this inner loop should execute 5 times, 10 times, etc.
        randIndex = np.random.choice(y.shape[0], replace=False)
        dataPointX = x[randIndex]
        dataPointY = y[randIndex]
        randomDataListX.append(dataPointX)
        randomDataListY.append(dataPointY)
        # pass data points into gradient descent fn
    # implement gradient descent function for each set of x and y 
    x = np.array(randomDataListX)
    y = np.array(randomDataListY)
    K = 0.01  
    w = np.array([[100],[-100]])
    D = np.array([[-1], [1]])
    mini_batch_gd(x, y, K, w, D)

    # plot 
    iterations = np.linspace(1,count, count)
    plt.plot(iterations, d_hist, color = 'b', marker = (5, 1))
    plt.plot(iterations, c_hist, color = 'b', marker = (5, 1))
    plt.title('Derivative and Cost Size with Respect to Iterations for %d Data Points' % i)
    plt.ylabel("Derivative/Cost Size")
    plt.xlabel("Iterations")
    plt.show()
    plt.clf()

# 1d. Repeat 1b, but now implement stochastic gradient descent.  Plot the curves 
#     for d_hist and c_hist.  WARNING: There is a strong possibility that your
#     cost and derivative definitions may not compute the values correctly for
#     for the 1-dimensional case.  If needed, make sure that you adjust these functions
#     to accommodate a single data point.  
def stoc_gd():
    return 0
# 1e. Aggregate your curves for batch, mini-batch, and stochastic descent methods
#     into one final graph so that a full comparison between all methods can be
#     observed.  Make sure your legend clearly indicates the results of each 
#     method.  Adjust the transparency of the curves as needed.


# Problem 2 - LASSO Regularization
#------------------------------------------------------------------------------
# For this problem you will be implementing LASSO regression on the Yosemite data.

# 2a. Fill out the function for the soft-thresholding operator S_lambda as discussed
#     in lecture:

def soft_thresh(v, lam):
    """Perform the soft-thresholding operation of the vector v using parameter lam."""
    #return FIXME
    
# 2b. Using 5 years of the Yosemite data, perform LASSO regression with the values 
#     of lam ranging from 0.25 up to 5, spacing them in increments of 0.25.
#     Specifically do this for a cubic model of the Yosemite data.  In doing this
#     save each of your optimal parameter vectors w to a list as well as solving
#     for the exact solution for the least squares problem.  Make the following
#     graphs:
#
#       a. Make a graph of the l^2 norms (i.e. Euclidean) and l^1 norms of the 
#          optimal parameter vectors w as a function of the coefficient lam.  
#          Interpret lam = 0 as the exact solution.  One can find the 1-norm of
#          a vector using LA.norm(w, ord = 1)
#       b. For each coefficient in the cubic model (i.e. there are 4 of these),
#           make a separate plot of the absolute value of the coefficient as a 
#           function of the parameter lam (again, lam = 0 should be the exact
#           solution to the original least squares problem).  Is there a 
#           discernible trend of the sizes of our entries for increasing values
#           of lam?

# Friendly Reminder: for LASSO regression you don't need backtracking line search.
# In essence the parameter lam serves as our stepsize.  
    
    
