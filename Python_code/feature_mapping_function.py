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

def feature_vector(x, y, n_classes, feature_mapping, feature_parameters):
    """
    Feature mappings

    This function obtains feature vectors

    Attributes
    ----------

    x: new instance

    y: new label

    n_classes: number of classes

    feature_mapping: 'linear' or 'RFF'

    feature_parameters:
        if feature_mapping == 'linear': feature_parameters = []
        if feature_mapping == 'RFF': feature_parameters = [D, u] where
            D = number of random Fourier components
            u = random Fourier components


    Output
    ------

    phi: feature vector

    """
    if feature_mapping == 'linear':
        x_phi = x
    elif feature_mapping == 'RFF':
        D_RFF = feature_parameters[0, 0];
        u = feature_parameters[1, 0];
        u1 = (1/np.sqrt(D_RFF))*np.cos(np.dot(np.transpose(u), x))
        u2 = (1/np.sqrt(D_RFF))*np.sin(np.dot(np.transpose(u), x))
        x_phi = (np.concatenate((u1,  u2)))

    e = np.zeros((1, n_classes))
    e[0, y] = 1
    phi = np.kron(e, x_phi)
    return phi
