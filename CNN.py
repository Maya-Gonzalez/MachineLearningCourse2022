from random import shuffle
import sys
import splitfolders
import os
from os.path import exists
from PIL import Image
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import h5py
import matplotlib.pyplot as plt
# from public_tests import *

# split dataset into train, test, eval
#splitfolders.ratio('/Users/mayagonzalez/Desktop/Dataset', output="/Users/mayagonzalez/Desktop/test/train", seed=1337, ratio=(.8, 0.2)) 

# data from https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset
path_test = '/Users/mayagonzalez/Desktop/test_train/train'
CATEGORIES = ["MILD", "MODERATE", "NON", "VERYMILD"]
W= 128
H = 128
label_to_class = {
    'Mild_Demented': 0,
    'Moderate_Demented': 1,
    'Non_Demented': 2,
    'Very_Mild_Demented':3
}

def get_images(dir_name = '/Users/mayagonzalez/Desktop/test_train/train', label_to_class = label_to_class):
    Images = []
    Classes = []

    for label_name in os.listdir(dir_name):
            cls = label_to_class[label_name]
            for image_name in os.listdir('/'.join([dir_name, label_name])):
                # do something
                img = load_img('/'.join([dir_name, label_name, image_name]), target_size=(160, 160))
                img = img_to_array(img)
                # plt.imshow(img, interpolation='nearest')
                # plt.show()

                

                Images.append(img)
                Classes.append(cls)
    X_train = np.array(Images, dtype=np.float32)
    Y_train = np.array(Classes, dtype=np.float32)

    # np.save('/Users/mayagonzalez/Desktop/test_train/trainArr.npy', trainImages)
    # np.save('/Users/mayagonzalez/Desktop/test_train/trainClass.npy', trainClasses)
    print('X_trainA shape ', X_train.shape)
    print('Y_trainA shape ', Y_train.shape)
    return X_train, Y_train

# X_train, Y_train = get_images()
# print(X_train[0])
# print(Y_train[0])

def load_dataset(path_to_train, path_to_test):
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # y reshaped
    train_y = train_y.reshape((1, train_x.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y
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
# X_train, y_train, X_test, y_test = load_MINIST()
# print(X_train[0,:,:,:].shape)
# print(y_train[0])
def exampleImg():
    Images = []
    path = '/Users/mayagonzalez/Desktop/ex'
    label_name = 1
    for image_name in os.listdir('/'.join([path])):
        # do something
        print('/'.join([path, image_name]))
        img = load_img('/'.join([path, image_name]), target_size=(160, 160))
        img = img_to_array(img)
        # plt.imshow(img, interpolation='nearest')
        # plt.show()
        Images.append(img)
            
    X_train = np.array(Images, dtype=np.float32)

    print('X_trainA shape ', X_train.shape)
    return X_train
X_train = exampleImg()
print(X_train[0])
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
    return X_pad

X_pad = zero_pad(X_train, pad = 1)
print(X_pad)
# randomzie weights of 3 dimensions 
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

def conv_single_step(a_slice_prev, W, b):

    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W 
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)

    return Z

# GRADED FUNCTION: conv_forward
hparameters = {"pad" : 1,
               "stride": 2}
# randomize weights of 4 dimensions
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
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

# pass previosu slice into conv_forward()) 
# A_prev will be X_train[i,:,:,:] i think 
Z, cache = conv_forward(X_pad, W, b, hparameters)
print('Z ', Z)
print('cache', cache)

# # GRADED FUNCTION: pool_forward

# def pool_forward(A_prev, hparameters, mode = "max"):
#     """
#     Implements the forward pass of the pooling layer
    
#     Arguments:
#     A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     hparameters -- python dictionary containing "f" and "stride"
#     mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
#     Returns:
#     A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
#     cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
#     """
    
#     # Retrieve dimensions from the input shape
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
#     # Retrieve hyperparameters from "hparameters"
#     f = hparameters["f"]
#     stride = hparameters["stride"]
    
#     # Define the dimensions of the output
#     n_H = int(1 + (n_H_prev - f) / stride)
#     n_W = int(1 + (n_W_prev - f) / stride)
#     n_C = n_C_prev
    
#     # Initialize output matrix A
#     A = np.zeros((m, n_H, n_W, n_C))              
    
#     for i in range(m):                         # loop over the training examples
#         for h in range(n_H):                     # loop on the vertical axis of the output volume
#             # Find the vertical start and end of the current "slice" (≈2 lines)
#             vert_start = h * stride 
#             vert_end = h * stride + f

#             for w in range(n_W):                 # loop on the horizontal axis of the output volume
#                 # Find the horizontal start and end of the current "slice" (≈2 lines)
#                 horiz_start = w * stride 
#                 horiz_end = w * stride + f

#                 for c in range (n_C):            # loop over the channels of the output volume

#                     # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
#                     a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]

#                     # Compute the pooling operation on the slice. 
#                     # Use an if statement to differentiate the modes. 
#                     # Use np.max and np.mean.
#                     if mode == "max":
#                         A[i, h, w, c] = np.max(a_prev_slice)
#                     elif mode == "average":
#                         A[i, h, w, c] = np.mean(a_prev_slice)
    
#     # Store the input and hparameters in "cache" for pool_backward()
#     cache = (A_prev, hparameters)
    
#     # Making sure your output shape is correct
# #     assert(A.shape == (m, n_H, n_W, n_C))
    
#     return A, cache

# def conv_backward(dZ, cache):
#     """
#     Implement the backward propagation for a convolution function
    
#     Arguments:
#     dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
#     cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
#     Returns:
#     dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
#                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     dW -- gradient of the cost with respect to the weights of the conv layer (W)
#           numpy array of shape (f, f, n_C_prev, n_C)
#     db -- gradient of the cost with respect to the biases of the conv layer (b)
#           numpy array of shape (1, 1, 1, n_C)
#     """    
    

#     # Retrieve information from "cache"
#     (A_prev, W, b, hparameters) = cache
#     # Retrieve dimensions from A_prev's shape
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
#     # Retrieve dimensions from W's shape
#     (f, f, n_C_prev, n_C) = W.shape
    
#     # Retrieve information from "hparameters"
#     stride = hparameters["stride"]
#     pad = hparameters["pad"]
    
#     # Retrieve dimensions from dZ's shape
#     (m, n_H, n_W, n_C) = dZ.shape
    
#     # Initialize dA_prev, dW, db with the correct shapes
#     dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                
#     dW = np.zeros((f, f, n_C_prev, n_C))
#     db = np.zeros((1, 1, 1, n_C))
    
#     # Pad A_prev and dA_prev
#     A_prev_pad = zero_pad(A_prev, pad)
#     dA_prev_pad = zero_pad(dA_prev, pad)
    
#     for i in range(m):                       # loop over the training examples
#         # select ith training example from A_prev_pad and dA_prev_pad
#         a_prev_pad = A_prev_pad[i,:,:,:]
#         da_prev_pad = dA_prev_pad[i,:,:,:]

#         for h in range(n_H):                   # loop over vertical axis of the output volume
#             for w in range(n_W):               # loop over horizontal axis of the output volume
#                 for c in range(n_C):           # loop over the channels of the output volume

#                     # Find the corners of the current "slice"
#                     vert_start = h * stride
#                     vert_end = vert_start  + f
#                     horiz_start = w * stride
#                     horiz_end = horiz_start  +f 

#                     # Use the corners to define the slice from a_prev_pad
#                     a_slice = a_prev_pad[ vert_start:vert_end, horiz_start:horiz_end,:]

#                     # Update gradients for the window and the filter's parameters using the code formulas given above
                    
#                     da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
#                     dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
#                     db[:,:,:,c] += dZ[i,h,w,c]

#         # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
#         dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
#     # YOUR CODE ENDS HERE
    
#     # Making sure your output shape is correct
#     assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
#     return dA_prev, dW, db

# def create_mask_from_window(x):
#     """
#     Creates a mask from an input matrix x, to identify the max entry of x.
    
#     Arguments:
#     x -- Array of shape (f, f)
    
#     Returns:
#     mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
#     """    
#     # (≈1 line)
#     # mask = None
#     # YOUR CODE STARTS HERE
#     mask = (x == np.max(x))
    
#     # YOUR CODE ENDS HERE
#     return mask

# def create_mask_from_window(x):
#     """
#     Creates a mask from an input matrix x, to identify the max entry of x.
    
#     Arguments:
#     x -- Array of shape (f, f)
    
#     Returns:
#     mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
#     """    
#     # (≈1 line)
#     # mask = None
#     # YOUR CODE STARTS HERE
#     mask = (x == np.max(x))
    
#     # YOUR CODE ENDS HERE
#     return mask


# def pool_backward(dA, cache, mode = "max"):
#     """
#     Implements the backward pass of the pooling layer
    
#     Arguments:
#     dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
#     cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
#     mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
#     Returns:
#     dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
#     """
#     # Retrieve information from cache (≈1 line)
#     (A_prev, hparameters) = cache
    
#     # Retrieve hyperparameters from "hparameters" (≈2 lines)
#     stride = hparameters['stride']
#     f = hparameters['f']
    
#     # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
#     m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
#     m, n_H, n_W, n_C = dA.shape
    
#     # Initialize dA_prev with zeros (≈1 line)
#     dA_prev = np.zeros( A_prev.shape)
    
#     for i in range(m): # loop over the training examples
        
#         # select training example from A_prev (≈1 line)
#         a_prev = A_prev[i,:]

#         for h in range(n_H):                   # loop on the vertical axis
#             for w in range(n_W):               # loop on the horizontal axis
#                 for c in range(n_C):           # loop over the channels (depth)
        
#                     # Find the corners of the current "slice" (≈4 lines)
#                     vert_start = h*stride
#                     vert_end = vert_start + f
#                     horiz_start = w*stride
#                     horiz_end = horiz_start + f
                    
#                     # Compute the backward propagation in both modes.
#                     if mode == "max":
                        
#                         # Use the corners and "c" to define the current slice from a_prev (≈1 line)
#                         a_prev_slice = a_prev[vert_start: vert_end, horiz_start:horiz_end, c]
                        
#                         # Create the mask from a_prev_slice (≈1 line)
#                         mask = create_mask_from_window(a_prev_slice)

#                         # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
#                         dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i,h,w,c]
                        
#                     elif mode == "average":
                        
#                         # Get the value da from dA (≈1 line)
#                         da = dA[i,h,w,c]
                        
#                         # Define the shape of the filter as fxf (≈1 line)
#                         shape = (f , f)

#                         # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
#                         dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    
#     # Making sure your output shape is correct
#     assert(dA_prev.shape == A_prev.shape)
    
#     return dA_prev