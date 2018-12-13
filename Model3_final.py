from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Maral Kasiri, Sepehr Jalali
CNN-> Model 3
No PCA, Wavelet ctransform coefficients as input
"""
##########################################################################
##########################################################################
### Import
##########################################################################
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from scipy.io import loadmat
from keras.optimizers import RMSprop
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd
##########################################################################
##########################################################################
# Load Data, Split and preparation
##########################################################################
Data=loadmat('CNN_Feature2.mat')['featureCNN']

x_train=Data[0:2100, 1:]
x_test= Data[2100:,1:]
y_train= Data[0:2100,0]
y_test= Data[2100:,0]


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
input_shape= ( x_test.shape[1], 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')



##########################################################################
##########################################################################
### Params
##########################################################################
batch_size = 500
n_class = 6
epochs = 60

##########################################################################
# convert class vectors to binary class matrices
##########################################################################
y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)


##########################################################################
# Model: Shallow neural network , one hidden layer
##########################################################################
model = Sequential()
model.add(Convolution1D(nb_filter=50, filter_length=10, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(batch_size, activation='relu'))
model.add(Dense(n_class))
model.add(Activation('softmax'))

##########################################################################
# Optimizer and compiling  : SGD and RMSprop
##########################################################################
sgd = SGD(lr=0.001, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.summary()


##########################################################################
# Model fit
##########################################################################
history= model.fit(x_train, y_train, nb_epoch=epochs, validation_data=(x_test, y_test), batch_size=batch_size)



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
    

predictions = model.predict(x_test)
y_pred= np.argmax(predictions, axis=1)
max_y_test=np.argmax(y_test, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(max_y_test, y_pred))
print()
print("Classification Report")
print(classification_report(max_y_test, y_pred))

##########################################################################
# result visualization
##########################################################################


plot_loss_accuracy(history)


##########################################################################
Labels= ["cyl", "hook", "lat", "palm", "spher", "tip"]

array= confusion_matrix(max_y_test, y_pred)
df= pd.DataFrame(array, index=[i for i in Labels], columns=[i for i in Labels])
plt.figure(figsize=(8,5))
sn.heatmap(df, annot=True)



##########################################################################
'''
CNN_Report= classification_report(max_y_test, y_pred)
print(CNN_Report)
import matplotlib.patches as mpatches
lab= [0,1,2,3,4,5]
cnn=[0.56, 0.59, 0.43, 0.30, 0.96, 0.51]
gb=[0.9 ,0.87 ,0.83 ,0.79 ,0.94 ,0.79]
w=0.2
plt.subplot(111)
plt.bar([float(x)+w for x in lab], cnn,width=0.2,color='r',align='center')
plt.bar([float(x) for x in lab], gb,width=0.2,color='g',align='center')
red_patch = mpatches.Patch(color='red', label='CNN')
green_patch = mpatches.Patch(color='green', label='Gradient boosting')
plt.legend(handles=[red_patch, green_patch], loc= 'lower right')
plt.xticks(lab, Labels)
plt.show()
'''