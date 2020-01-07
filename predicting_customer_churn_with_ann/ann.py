# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:38:26 2020

@author: dhruv
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values          ## as 1st 3 columns are not significant in predicting if customer will exit , so drop them
y = dataset.iloc[:,13].values   ## exited column

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()     ## encoding country variable to integer
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender= LabelEncoder()     ## encoding gender variable to integer
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])

## for creating dummy variables for our country class as there are 3 countries
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

## now we can see that our country variable has been converted into dummy variables 
## now we could remove one column to avoid dummy variable trap
X = X[:,1:]   # all except 1'st column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



## import libraries for our DL model
import keras
from keras.models import Sequential
from keras.layers import Dense
 
### our ANN build

## classifier is our ANN model
classifier = Sequential()
## our dependent layer has only 2 values 0 &1 so output layer dim = 11+1/2 = 6
 ## uniform is initializing random uniform number close to 0 as weight for our nodes
 ## selecting activation as relu
classifier.add(Dense(input_dim=11,output_dim=6,init='uniform',activation='relu',)) ## first hidden layer

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',)) ## second hidden layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid',)) ## output layer
 

## generally we in  logistic we get log-loss as loss function
## binary_crossentropy for binary classification
## for more than 3 classification we would use categorical_crossentropy
## metrics set to reach max accuracy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics =['accuracy'])


classifier.fit(X_train,y_train,batch_size=10,epochs=100)
 
# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)  ### this pred method will return probabilities
y_pred =(y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



     