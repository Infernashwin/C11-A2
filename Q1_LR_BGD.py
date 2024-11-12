#!/usr/bin/env python3
# A2-Q1 Starter Code: Logistic Regression Model with Batch Gradient Descent

import numpy as np

# Data: [Annual Income, Credit Score, Loan Approved (0 = No, 1 = Yes)]
data = np.array([
    [40000, 600, 0],
    [70000, 720, 1],
    [50000, 650, 0],
    [90000, 750, 1],
    [100000, 800, 1],
    [45000, 630, 0]
], dtype=np.float32)

# Split your data to Features (X) and labels (y)
X = data[:, :-1]  # Features
y = data[:, -1]  # Labels

# Feature Scaling (Standardization)
# Standardize the features to have mean 0 and standard deviation 1
# numpy ndarray has mean() and std() functions
income_mean = np.mean(X[:, 0])
income_std = np.std(X[:, 0])
credit_mean = np.mean(X[:, 1])
credit_std = np.std(X[:, 1])

X[:, 0] = (X[:, 0] - income_mean) / income_std
X[:, 1] = (X[:, 1] - credit_mean) / credit_std
# Augment X with bias term
X = np.hstack((np.ones((X.shape[0], 1)), X))


# Initialize parameters to zeros or small (random) numbers
theta = np.full(X.shape[1], 0.1)  # model parameters

# Prediction function
def predict(X, theta):
    """Return 1 (loan approved) or 0 (loan rejected) given model parameter theta and
       augmented input X, where each row is a data point and the data point has d features.
    """
    # Run the sigmoid function on the dot product of X and theta
    probabilities = sigmoid(np.dot(X, theta))
    return (probabilities >= 0.5).astype(int)


# Compute the Accuarcy
def accuracy(y, predicted_y):
    """Return the fraction of data where y == predicted_y
    input:
        - y (ndarray) vecotr of training output data
        - predicted_y (ndarray) vecotr of predicted output given input of training X
    output:
        - accuracy (float) fracton of data points wehre y == predicted_y
    """
    # Return the mean of the comparison between y and predicted_y
    return np.mean(y == predicted_y)

# Sigmoid function
def sigmoid(z):
    """Return the sigmoid function output given input z.
    args:
        - z (ndarray (shape: (N, 1))): A column vector consisting N scalars .
    output:
        - sigmoid of z (ndarray (shape: (N, 1))): A column vector consisting N scalars .
    """
    # Return the sigmoid function of z
    return 1 / (1 + np.exp(-z))

#  Gradient descent algorithm
def gradient_descent(X, y, theta, learning_rate, iterations):
    """Return updated parameters theta after one step of gradient descent.
    args:
        - X (ndarray): the augmented design matrix
        - y (ndarray): the output scalar vector
        - theta: (ndarray) vector of model parameters
        - learning_rate: a real number
        - iterations: an integer, maximum iterations
    output:
        - updated model parameter theta
    """
    new_theta = theta.copy()
    print('Iteration:', 0, '(Initial Values)')
    print('Theta:', new_theta)
    print('Predictions:', predict(X, new_theta))
    print('Accuracy:', accuracy(y, predict(X, new_theta)))
    print("-" * 50)
    for i in range(iterations):
        gradient = np.dot(X.T, (sigmoid(np.dot(X, new_theta)) - y))/y.size
        new_theta -= learning_rate * gradient
        print('Iteration:', i+1)
        print('Theta:', new_theta)
        print('Predictions:', predict(X, new_theta))
        print('Accuracy:', accuracy(y, predict(X, new_theta)))
        print("-" * 50)
    return new_theta

# Some simple test
# Hyperparameters
learning_rate = 0.1
iterations = 3

# Run gradient descent for 3 iterations
theta = gradient_descent(X, y, theta, learning_rate, iterations)
print(theta)

