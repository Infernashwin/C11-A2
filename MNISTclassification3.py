#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:54:33 2024

@author: mehdy
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.asarray(X)

Y = np.asarray(y)
print('X shape', X.shape)
print('Y shape', y.shape)

def show_digit(x_, y_):
    X_reshape = x_.reshape(28, 28) # reshape it to have 28*28
    plt.imshow(X_reshape, 'gray')
    plt.title('Label is: ' + y_)
    plt.show()

#show_digit(X[0],y[0])

test_percentage = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size= test_percentage, random_state=12) # split the data to have 20% test set and 80% training set
X_test = X_test.astype('float64')
X_test += np.random.uniform(0,255,size = X_test.shape)
min_train = np.min(X_train)
max_train = np.max(X_train)
X_train = (X_train - min_train) / (max_train - min_train)  # Normalize X_train based on the maximum value of X_train
X_test = (X_test - min_train) / (max_train - min_train) #  Normalize X_test based on the maximum value of X_train
Noise = [str(index) for index in np.random.choice(10,20000, p =[0.25,0.05,0.05,0.05,0.08,0.17,0.13,0.1,0.12,0.00])]
y_train[:20000] = Noise

from sklearn.neighbors import KNeighborsClassifier
R_train_KNN = []
R_test_KNN = []
Neighbours = np.arange(1,100,5)
for k in Neighbours:
  model = KNeighborsClassifier(n_neighbors=k, weights="distance")  # define KNN model
  model.fit(X_train, y_train)  # fit the data
  y_res_train = model.predict(X_train) # Output for training set
  y_res_test = model.predict(X_test) # Output for test set
  R_train_KNN.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_KNN.append(sklearn.metrics.accuracy_score(y_test, y_res_test))


from sklearn.ensemble import RandomForestClassifier
R_train_RF = []
R_test_RF = []
Min_sample = np.arange(5,51,5)
for s in Min_sample:
  model_RF = RandomForestClassifier(min_samples_leaf=s, ) # define Random forest model
  model_RF.fit(X_train, y_train) # fit the data
  y_res_train = model_RF.predict(X_train) # Output for train
  y_res_test = model_RF.predict(X_test)
  R_train_RF.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_RF.append(sklearn.metrics.accuracy_score(y_test, y_res_test))

# Plot the Results
plt.figure(figsize=(12, 5))
plt.plot(Neighbours, R_train_KNN, label="Training Accuracy")
plt.plot(Neighbours, R_test_KNN, label="Testing Accuracy")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs Number of Neighbors MNIST 3")
plt.legend()
plt.grid()
plt.show()
plt.savefig("MNIST3_KNN")

plt.figure(figsize=(12, 5))
plt.plot(Min_sample, R_train_RF, label="Training Accuracy")
plt.plot(Min_sample, R_test_RF, label="Testing Accuracy")
plt.xlabel("Min Sample Leaf")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy vs Min Sample Leaf MNIST 3")
plt.legend()
plt.grid()
plt.show()
plt.savefig("MNIST3_RF")

# Print the best K for KNN
best_k = Neighbours[np.argmax(R_test_KNN)]
print(f"Best K for KNN: {best_k} with Test Accuracy: {max(R_test_KNN):.4f}")

# Print the best Min Sample Leaf for Random Forest
best_min_samples_leaf = Min_sample[np.argmax(R_test_RF)]
print(f"Best Min Sample Leaf for Random Forest: {best_min_samples_leaf} with Test Accuracy: {max(R_test_RF):.4f}")

