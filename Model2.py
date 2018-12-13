from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Maral Kasiri, Sepehr Jalali
CNN-> Model 2
PCA, Wavelet ctransform coefficients as input
"""
##########################################################################
##########################################################################
### Import
##########################################################################

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling2D
from keras import backend as K
from scipy.io import loadmat
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sn
import pandas as pd


# the data, split between train and test sets
#Data=loadmat('Data_Input.mat')

Data=loadmat('CNN_Feature2.mat')['featureCNN']


y_train= Data[0:2100,0]
y_test= Data[2100:,0]



scaler = StandardScaler()

# Fit on training set only.
scaler.fit(Data)

# Apply transform to both the training set and the test set.
Data = scaler.transform(Data)

pca = PCA(.9)
pca.fit(Data)

pca.n_components_
Data = pca.transform(Data)


x_train=Data[0:2100, :]
x_test= Data[2100:,:]

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

input_shape= ( x_train.shape[1], 1)

##########################################################################
##########################################################################
### Params
##########################################################################

num_classes=6
learning_rate= 0.001
batch_size=200
epochs = 10
momentum=0.4



##########################################################################
# Model: Shallow neural network , one hidden layer
##########################################################################
model = Sequential()
model.add(Convolution1D(nb_filter=50, filter_length=10, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()



##########################################################################
# Optimizer and compiling  : SGD and RMSprop
##########################################################################


sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=momentum)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
history= model.fit(x_train, y_train, nb_epoch=epochs, validation_data=(x_test, y_test), batch_size=batch_size)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




##########################################################################
# Model evaluation and Visualization
##########################################################################
 
score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

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


Labels= ["cyl", "hook", "lat", "palm", "spher", "tip"]

array= confusion_matrix(max_y_test, y_pred)
df= pd.DataFrame(array, index=[i for i in Labels], columns=[i for i in Labels])
plt.figure(figsize=(8,5))
sn.heatmap(df, annot=True)
