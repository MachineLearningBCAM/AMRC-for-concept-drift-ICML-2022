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

def predict_label(x, mu, n_classes, varphi, deterministic, feature_mapping, feature_parameters):
    """
    Predict

    This function assigns labels to instances

    Attributes
    ----------

    x: instance

    mu: classifier parameter

    n_classes: number of classes

    varphi: varphi function obtained at learning

    deterministic: "True" for deterministic AMRC and "False" for AMRC

    feature_mapping: 'linear' or 'RFF'

    feature_parameters:
        if feature_mapping == 'linear': feature_parameters = []
        if feature_mapping == 'RFF': feature_parameters = [D, u] where
            D = number of random Fourier components
            u = random Fourier components

    Output
    ------

    y_pred: predicted label

    """
    M = np.zeros((n_classes, len(mu)))
    c = np.zeros((n_classes, 1))
    for j in range(0, n_classes):
        M[j, :] = feature_vector(x, j, n_classes, feature_mapping, feature_parameters)
    for j in range(0, n_classes):
        c[j, 0] = max([np.dot(M[j, :], mu)[0] - varphi, 0])
    cx = sum(c)
    h = np.zeros((n_classes, 1))
    for j in range(0, n_classes):
        if cx == 0:
            h[j, 0] = 1/n_classes
        else:
            h[j, 0] = c[j, 0]/cx
    if deterministic == "True":
        y_pred = np.where(np.max(h));
    elif deterministic == "False":
        y_pred = np.where(np.random.multinomial(1, np.transpose(h)[0]) == 1)
    else:
        print('Error2')
    return y_pred
