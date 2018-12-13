from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Maral Kasiri, Sepehr Jalali
CNN-> Model 1
withou PCA, raw data in form of 2*2500*1 images as input
2D convolution, 3 hidden layers
"""
##########################################################################
##########################################################################
### Import
##########################################################################

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras import backend as K
from scipy.io import loadmat
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.patches as mpatches
import seaborn as sn
import pandas as pd
##########################################################################
##########################################################################
# Load Data, Split and preparation
##########################################################################

xData=loadmat('RawDataCnn.mat')['Data1']
yData=loadmat('RawDataCnn.mat')['Label']

x_test=xData[2100:,:,:]
x_train=xData[0:2100,:,:]
y_test=yData[2100:,:]
y_train=yData[0:2100,:]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


##########################################################################
##########################################################################
### Params
##########################################################################

input_shape= ( x_test.shape[1],x_test.shape[2], 1)
epochs=20
batch_size=100
n_class=6
learning_rate = .001
y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)


##########################################################################
# Model: deep neural network , 3 hidden layer
##########################################################################

    

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 10),padding= 'same', activation='relu',
                 input_shape=input_shape))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), data_format= 'channels_first'))
model.add(Conv2D(64, kernel_size=(2, 5),padding= 'same', activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(2, 5),padding= 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format= 'channels_last'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(n_class, activation='softmax'))
model.summary()
##########################################################################
# Optimizer and compiling  : SGD and RMSprop
##########################################################################

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])
history= model.fit(x_train, y_train, nb_epoch=epochs, validation_data=(x_test, y_test), batch_size=batch_size)



##########################################################################
# Model evaluation and Visualization
##########################################################################
def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    

plot_loss_accuracy(history)



predictions = model.predict(x_test)
y_pred= np.argmax(predictions, axis=1)
max_y_test=np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(max_y_test, y_pred))
print()
print("Classification Report")
print(classification_report(max_y_test, y_pred))

Lab= ["cyl", "hook", "lat", "palm", "spher", "tip"]

array= confusion_matrix(max_y_test, y_pred)
df= pd.DataFrame(array, index=[i for i in Lab], columns=[i for i in Lab])
plt.figure(figsize=(8,5))
sn.heatmap(df, annot=True)





