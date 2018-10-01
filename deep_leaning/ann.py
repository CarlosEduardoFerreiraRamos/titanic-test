# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import os 
# dir_path = os.path.dirname(os.path.realpath('deep_leaning'))

# Importing the dataset
dataset = pd.read_csv('data_sets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

"""
output_dim: number of nodes, Donsn't exist a optimal number of nodes to be used. But there is a tip,
the number of nodes in the hidden layaer is average of nodes in the input layer nad the outputlayer.
Or, can be use parameter tunning,k folds, with cross validation
init: 
activation:
input_dim:
"""

# the number of the input layers is the number of independet variables
# the number of the output layer is the dependent variable
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# retifier function = 'relu', in this library

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# the sigmoid function is recomended fo the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

"""
optimizer: adam is a  Stochastic Gradient Descent algorithm
loss: is the loss function
metrics: criteria to avluete the model
"""
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""
When you take a simple neuro network, with only one neuron, witch make it a perseptron model nad use a sigmoid activation function
for this perceptron, you obtain a logisticRegression Model. If you go deep in the Stochastic Gradient Descent algorithm for the
logisticRegression Model, you are gonna find taht the loss function is not the sum of the sqare error, but is a loguerical mic function
called loguerical mic loss
"""

"""
bbatch size: the number of observation processed required to change the weith
nb_epoch: number of epoch
"""
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)