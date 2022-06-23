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
from feature_mapping_function import feature_vector 

def tracking(feature, y, k, Ht, eta, Sigma, eta0, Sigma0, epsilon, Q, R, e1, p, s, unidimensional):
    """
     Tracking uncertainty sets

     This function obtains mean vector estimates and confidence vectors

     Input
     -----

     feature: feature vector

     y: new label

     n_classes: number of classes

     eta: state vector estimate composed by mean vector estimate and its k derivatives

     Sigma: mean quadratic error matrix

     Ht: transition matrix

     D: diagonal matrix

     eta0, Sigma0, epsilon: parameters required to update variances of noise processes

     e1: vector with 1 in the first component and 0 in the remainning components

     Q, R: variances of random noises

     p, s: probability

     unidimensional: "True" for unidimensional AMRC and "False" for AMRC

     Output
     ------

     eta: updated mean vector estimate

     Sigma: updated mean quadratic error matrix

     tau: mean vector estimate

     lmb: confidence vector

     eta0, Sigma0, epsilon: parameters required to update variances of noise processes
     
     Q, R: variances of noise processes

    """
    m = len(feature[0])
    n_classes = len(p)
    d = m/n_classes
    alpha = 0.3
    tau = np.zeros((m, 1))
    lmb = np.zeros((m, 1))
    if unidimensional == 'True':
        KK = np.zeros((m, 1))
        for i in range(0, m):
            innovation = feature[0, i] - eta[0, i]
            aa = alpha*R[i, 0] + (1-alpha)*(np.dot(epsilon[i], epsilon[i]) + np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1)))
            R[i] = aa[0]
            a = (np.dot(Sigma[i, :, :], np.transpose(e1)))
            b = np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1))+ R[i, :]
            KK[i] = a/b
        K = np.mean(KK)
        for i in range(0, m):
            eta0[:, i] = eta[:, i] + K*innovation
            Sigma0[i, :, :] = np.dot((np.identity(k+1) - np.dot(K,e1)), Sigma[i, :, :])
            Q[i, :, :] = alpha*Q[i, :, :] + (1-alpha)*np.dot(innovation*innovation*K, np.transpose(K))
            epsilon[i] = feature[0, i] - eta0[0, i]
            eta[:, i] = np.dot(Ht, eta0[:, i])
            Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]), np.transpose(Ht)) + Q[i, :, :]
            tau[i, 0] = (1/n_classes)*eta[0, i]
            lmb_eta = np.sqrt(Sigma[i, 0, 0])
            lmb[i, 0] = np.mean(lmb_eta)
    elif unidimensional == 'False':
        for i in range(0, m):
            if i > y*d-1 and i < (y+1)*d+1:
                innovation = feature[0, i] - eta[0, i]
                aa = alpha*R[i, 0] + (1-alpha)*(np.dot(epsilon[i], epsilon[i]) + np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1)))
                R[i] = aa[0]
                a = (np.dot(Sigma[i, :, :], np.transpose(e1)))
                b = np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1))+ R[i, :]
                K = a/b
                eta0[:, i] = eta[:, i] + np.transpose(K[:]*innovation)
                Sigma0[i, :, :] = np.dot((np.identity(k+1) - np.dot(K,e1)), Sigma[i, :, :])
                Q[i, :, :] = alpha*Q[i, :, :] + (1-alpha)*np.dot(innovation*innovation*K, np.transpose(K))
                epsilon[i] = feature[0, i] - eta0[0, i]
                eta[:, i] = np.dot(Ht, eta0[:, i])
                Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]), np.transpose(Ht)) + Q[i, :, :]
                tau[i, 0] = p[y[0]]*eta[0, i]
                lmb_eta = np.sqrt(Sigma[i, 0, 0])
                lmb[i, 0] = np.sqrt((lmb_eta**2 + eta[0, i]**2)*(s[y[0]]**2 + p[y[0]]**2) - ((eta[0, i])**2)*(p[y[0]]**2))
            else:
                eta[:, i] = np.dot(Ht, eta0[:, i])
                Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]), np.transpose(Ht)) + Q[i, :, :]
                tau[i, 0] = (p[int((i)/d)])*eta[0, i]
                lmb_eta = np.sqrt(Sigma[i, 0, 0])
                lmb[i, 0] = np.sqrt((lmb_eta**2 + eta[0, i]**2)*(s[int((i)/d)]**2 + p[int((i)/d)]**2) - ((eta[0, i])**2)*(p[int((i)/d)]**2))
    else:
            print("Error1")
    return tau, lmb, eta, Sigma, eta0, Sigma0, epsilon, Q, R
  
def initialize_tracking(m, k):
    """
    Initialize tracking stage

    This function initializes mean vector estimates, confidence vectors,
    and defines matrices and vectors that are used to update mean vector estimates and confidence vectors.

    Attributes
    ----------

    m: length of mean vector estimate

    k: order

    Output
    ------

    Ht: transition matrix

    e1: vector with 1 in the first component and 0 in the remainning components

    eta: state vectors

    Sigma: mean squared error matrices

    eta0, Sigma0, epsilon: parameters required to obtain variances of noise processes

    Q, R: variances of noise processes

    """

    e1 = np.ones((1, k+1))
    for i in range(1, k+1):
        e1[0, i] = 0
    deltat = 1
    variance_init = 0.001
    Ht = np.identity(k+1)
    for i in range(0, k):
        for j in range(i+1, k+1):
            Ht[i, j] = pow(deltat, j-i)/math.factorial(j-i)
    eta0 = np.zeros((k+1, m))
    eta = np.zeros((k+1, m))
    Sigma0 = np.zeros((m, k+1, k+1))
    Sigma = np.zeros((m, k+1, k+1))
    Q = np.zeros((m, k+1, k+1))
    R = np.zeros((m, 1))
    epsilon = np.zeros((m, 1))

    for i in range(0, m):
        for j in range(0, k+1):
            Sigma0[i, j, j] = 1
            Q[i, j, j] = variance_init
        R[i] = variance_init
        epsilon[i] = -eta0[0, i]
        eta[:, i] = np.dot(Ht, eta0[:, i])
        Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]), np.transpose(Ht)) + Q[i, :, :]
    return Ht, e1, eta, Sigma, eta0, Sigma0, epsilon, Q, R