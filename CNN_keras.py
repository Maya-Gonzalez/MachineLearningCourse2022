import numpy as np
import pandas as pd
import splitfolders

splitfolders.ratio('/Users/mayagonzalez/Desktop/Dataset', output="/Users/mayagonzalez/Desktop/subAD", seed=1337, ratio=(.8, 0.1,0.1)) 

# def load_dataset(path_to_train, path_to_test):
#     train_dataset = h5py.File(path_to_train)
#     train_x = np.array(train_dataset['train_set_x'][:])
#     train_y = np.array(train_dataset['train_set_y'][:])

#     test_dataset = h5py.File(path_to_test)
#     test_x = np.array(test_dataset['test_set_x'][:])
#     test_y = np.array(test_dataset['test_set_y'][:])

#     # y reshaped
#     train_y = train_y.reshape((1, train_x.shape[0]))
#     test_y = test_y.reshape((1, test_y.shape[0]))

#     return train_x, train_y, test_x, test_y

# # Loading the data (signs)
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# # Example of an image from the dataset
# index = 9
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# X_train = X_train_orig/255.
# X_test = X_test_orig/255.
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))


# # GRADED FUNCTION: convolutional_model

# def convolutional_model(input_shape):
#     """
#     Implements the forward propagation for the model:
#     CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
#     Note that for simplicity and grading purposes, you'll hard-code some values
#     such as the stride and kernel (filter) sizes. 
#     Normally, functions should take these values as function parameters.
    
#     Arguments:
#     input_img -- input dataset, of shape (input_shape)

#     Returns:
#     model -- TF Keras model (object containing the information for the entire training process) 
#     """

#     input_img = tf.keras.Input(shape=input_shape)
#     ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
#     # Z1 = None
#     ## RELU
#     # A1 = tf.keras.layers.ReLU()
#     ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
#     # P1 = None
#     ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
#     # Z2 = None
#     ## RELU
#     # A2 = None
#     ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
#     # P2 = None
#     ## FLATTEN
#     # F = None
#     ## Dense layer
#     ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
#     # outputs = None
#     # YOUR CODE STARTS HERE
    
#     ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
#     Z1 = tf.keras.layers.Conv2D(filters= 8 , kernel_size= (4,4)  , strides = 1, padding='same')(input_img)
#     ## RELU
#     A1 = tf.keras.layers.ReLU()(Z1)
#     ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
#     P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)
#     ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
#     Z2 = tf.keras.layers.Conv2D(filters= 16, kernel_size= (2,2) , strides = 1, padding='same')(P1)
#     ## RELU
#     A2 = tf.keras.layers.ReLU()(Z2)
#     ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
#     P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)
#     ## FLATTEN
#     F = tf.keras.layers.Flatten()(P2)
#     ## Dense layer
#     ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
#     outputs = tf.keras.layers.Dense(units= 6, activation='softmax')(F)
    
    
#     # YOUR CODE ENDS HERE
#     model = tf.keras.Model(inputs=input_img, outputs=outputs)
#     return model


# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
# history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)


# history.history

# # The history.history["loss"] entry is a dictionary with as many values as epochs that the
# # model was trained on. 
# df_loss_acc = pd.DataFrame(history.history)
# df_loss= df_loss_acc[['loss','val_loss']]
# df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
# df_acc= df_loss_acc[['accuracy','val_accuracy']]
# df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
# df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
# df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')