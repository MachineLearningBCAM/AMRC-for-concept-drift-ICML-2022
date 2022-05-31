#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""
import sys

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools
import numpy as np
import scipy.io
import math
import os
import tracking_uncertainty_sets as tus
from feature_mapping_function import feature_vector 
from efficient_learning import learning 
from prediction import predict_label 


"""
  Input
  ------
      The name of dataset file

      k: Order

      W: Number of past labels
    
      N: Number of subgradients in the learning stage

      K: Number of iterations in the learning stage
      
      Deterministic: "True" for Deterministic AMRC and "False" for AMRC

      Unidimensional: "True" for Unidimensional AMRC and "False" for AMRC

      feature mapping

  Output
  ------

      Mistakes rate

      Mistakes indices

      R(U_t)

"""

# Import data
filename = 'usenet2' # without .mat extension
suffix = '.mat'
dataset_name = os.path.join('../' + filename + suffix)
data = scipy.io.loadmat(dataset_name)
X = data['X']
Y = data['Y']
# Number of classes
n_classes = len(np.unique(Y))
# Length of the instance vectors
d = len(X[0])

# Choose a feature mapping
feature_mapping = 'linear'

order = 1
W = 200
N = 100
K = 2000

# Deterministic AMRC or AMRC
deterministic = "True"

# Unidimensional AMRC or AMRC
unidimensional = "False"
if unidimensional  ==  'False':
    order = order
elif unidimensional == 'True':
    order = 0
else:
    print('Error')

# Parameters of the RFF feature mapping
D = 200
gamma = 2^6

# Calculate the length m of the feature vector
if feature_mapping == 'linear':
    feature_parameters = [];
    m = n_classes*(d);
elif feature_mapping == 'RFF':
    feature_parameters = np.zeros((2, 1), dtype=object)
    feature_parameters[0, 0] = D
    feature_parameters[1, 0] = gamma*np.random.randn(d, feature_parameters[0, 0])
    m = n_classes*(2*feature_parameters[1, 0]);

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# number of instances
T = len(Y)
p = np.zeros((n_classes, T))
s = np.zeros((n_classes, T))

# Initialize classifier parameter, upper bounds vector, and matrix and vector that are used to obtain local approximations of varphi function
M = np.zeros((1, m))
h = np.zeros((1, 1))
mu = np.zeros((m, 1))
w = np.zeros((m, 1))
w0 = np.zeros((m, 1))
R_Ut = np.zeros((T, 1))


# Initialize mistakes counter
mistakes_idx = np.zeros((T, 1))
mistakes = 0

# Initialize mean vector estimate
Ht, e1, eta, Sigma, eta0, Sigma0, epsilon, Q, R = tus.initialize_tracking(m, order)

for t in range(0, T-1):
    # New instance-label pair
    x = X[t]
    y = Y[t]

    # Estimating probabilities
    for i in range(0, n_classes):
        if t < W:
            p[i, t] = np.mean(Y[0:t+1] == i)
            s[i, t] = np.std(p[i, 0:t+1])
        else:
            p[i, t] = np.mean(Y[t-W:t+1] == i)
            s[i, t] = np.std(p[i, t-W:t+1])
    # Feature vector
    feature = feature_vector(x, y[0], n_classes, feature_mapping, feature_parameters)

    # Update mean vector estimate and confidence vector
    tau, lmb, eta, Sigma, eta0, Sigma0, epsilon, Q, R = tus.tracking(feature, y, order, Ht, eta, Sigma, eta0, Sigma0, epsilon, Q, R, e1, p[:, t], s[:, t], unidimensional)
   
    # Update classifier parameter and obtain upper bound
    mu, M, h, R_Ut, varphi,  w, w0 = learning(x, N, n_classes, mu, t, tau, lmb, M, h, w, w0, K, feature_mapping, feature_parameters)

    # New  instance, test instance
    x_test = X[t+1]

    # Predict label for the new instance
    hat_y = predict_label(x_test, mu, n_classes, varphi, deterministic, feature_mapping, feature_parameters)

    # Receive the true label, test label
    y_test = Y[t+1]

    # Mistakes count
    if hat_y[0] != y_test[0]:
        mistakes = mistakes + 1
        mistakes_idx[i] = 1

# Print mistakes rate
print("AMRC for order ", order, " has a mistake rate ", mistakes/(T-1), " in ", filename , " dataset using ", feature_mapping, " feature mapping") 

# Save results as .mat file
mdic = {"mistakes_idx": mistakes_idx, "mistakes": mistakes, "R_Ut": R_Ut}
results_file_name = os.path.join(filename + "_results"  + suffix)
scipy.io.savemat(results_file_name, mdic)
