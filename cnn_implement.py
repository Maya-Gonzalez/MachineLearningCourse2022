import numpy as np
from random import shuffle
import sys
import splitfolders
import os
from os.path import exists
from PIL import Image
from matplotlib import pyplot as plt
from keras.datasets import mnist
# ________________________________________________________________________________________________________________________
def load_MINIST():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)

    # Flattening the images from the 28x28 pixels to 1D 787 pixels
    # X_train = X_train.reshape(60000, 784)
    # X_test = X_test.reshape(10000, 784)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) 
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255


    return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = load_MINIST()
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
    return X_pad
def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W 
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    return Z
def forward(A_prev, W, b, hparameters):
    # Retrieve dimensions from A_prev's shape (≈1 line) 
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = np.shape(W)
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']
    # Compute the dimensions of the CONV output volume using the formula given above. 
    n_H = int((n_H_prev - f + 2*pad)/stride +1)
    n_W = int((n_W_prev - f + 2*pad)/stride +1)
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros([m, n_H, n_W, n_C]) 
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]      # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" 
            vert_start = h*stride
            vert_end = h*stride +f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" 
                horiz_start = w*stride
                horiz_end = w*stride +f
                
                for c in range(n_C):   # loop over channels (= #filters) of the output volume          
                    # Use the corners to define the (3D) slice of a_prev_pad 
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start: horiz_end, :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. 
                    weights = W[:,:,:, c]
                    biases = b[:,:,:,c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache
def backward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]

        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start  + f
                    horiz_start = w * stride
                    horiz_end = horiz_start  +f 

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[ vert_start:vert_end, horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i,h,w,c]

        # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db

hparameters = {"pad" : 2,
               "stride": 2}
A_prev = A_prev = np.random.randn(2, 5, 7, 4)
W = np.random.randn(3, 3, 3, 8) # should be (3,3,4,8)
b = np.random.randn(1, 1, 1, 8)
Z, cache = forward(X_train, W, b, hparameters)
class CONV1():
    forward(A_prev, W, b, hparameters)
    Z, cache = forward(X_train, W, b, hparameters)
    backward(Z, cache)

network = [
    CONV1()
]

epochs = 40
learning_rate = 0.1

# training


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

print('traiing')
for epoch in range(epochs):
    error = 0
    for x, y_true in zip(X_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # error (display purpose only)
        error += mse(y_true, output)

        # backward
        output_error = mse_prime(y_true, output)
        for layer in reversed(network):
            output_error = layer.backward(output_error, learning_rate)
    print('done 1')
    
    error /= len(X_train)
    print('%d/%d, error=%f' % (epoch + 1, epochs, error))