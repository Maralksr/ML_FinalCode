'''
Gradient Boosting Model for EMG Classification
Maral Kasiri
Sepehr Jalali
'''

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
#import matplotlib.patches as mpatches
## Import data and features and labels
########################################## 

data= loadmat('Feature_GB.mat')['Feature_gb']

x_train= data[:,1:]
y_train= data[:,0]





######################################nData preparation

num_classes=6



scaler = MinMaxScaler()
x_train_scale = scaler.fit_transform(x_train)

######## splitting training and the test data
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(x_train_scale, y_train,train_size= 0.75, random_state=0)


#Gradient boosting Model

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.6,0.75,0.8 , 0.9, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate = learning_rate, max_features=46, max_depth = 3, random_state = 0)
    gb.fit(X_train_sub, y_train_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))
    print()
predictions = gb.predict(X_validation_sub)

print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions))

###############################################################

n_estimators=[20, 50, 100, 200]

learning_rate = 0.25
for n in n_estimators:
    gb1 = GradientBoostingClassifier(n_estimators=n, learning_rate = learning_rate, max_features=46, max_depth = 3, random_state = 0)
    gb1.fit(X_train_sub, y_train_sub)
    print("Estimators: ", n)
    print("Accuracy score (training): {0:.3f}".format(gb1.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb1.score(X_validation_sub, y_validation_sub)))
    print()

predictions1 = gb1.predict(X_validation_sub)

print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions1))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions1))
################################################################
max_depth=[2,3,10,20]
n=100

for md in max_depth:
    gb2 = GradientBoostingClassifier(n_estimators=n, learning_rate = learning_rate, max_features=46, max_depth =md, random_state = 0)
    gb2.fit(X_train_sub, y_train_sub)
    print("Max Depth: ", md)
    print("Accuracy score (training): {0:.3f}".format(gb2.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb2.score(X_validation_sub, y_validation_sub)))
    print()


predictions2 = gb2.predict(X_validation_sub)

print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions2))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions2))
###############################################################
###############################################################
###############################################################
######## Final Model###########################################
###############################################################
###############################################################
###############################################################
learning_rate=0.6
max_depth=3
n_estimators=100
max_features=46

gb_final = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate = learning_rate, max_features=max_features, max_depth =max_depth, random_state = 0)
gb_final.fit(X_train_sub, y_train_sub)
print("Accuracy score (training): {0:.3f}".format(gb_final.score(X_train_sub, y_train_sub)))
print("Accuracy score (validation): {0:.3f}".format(gb_final.score(X_validation_sub, y_validation_sub)))
print()


predictions_final = gb_final.predict(X_validation_sub)

print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions_final))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions_final))



###############################################################
###############################################################
###############################################################
#######Result Visualization ###################################
###############################################################
###############################################################
###############################################################

Lab= ["cyl", "hook", "lat", "palm", "spher", "tip"]
array= confusion_matrix(y_validation_sub, predictions_final)
df= pd.DataFrame(array, index=[i for i in Lab], columns=[i for i in Lab])
plt.figure(figsize=(8,5))
sn.heatmap(df, annot=True)

'''
plt.plot(learning_rates, val_acc, 'bs', learning_rates, train_acc, 'r^')
plt.xlabel('Learning Rate')
plt.ylabel('% Accuracy ')
plt.axis([0, 1.01, 50, 110])
plt.grid(True)
red_patch = mpatches.Patch(color='red', label='Training Accuracy')
blue_patch = mpatches.Patch(color='blue', label='Validation Accuracy')
plt.legend(handles=[red_patch, blue_patch])
'''