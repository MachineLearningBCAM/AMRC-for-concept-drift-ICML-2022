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

def learning(x, N, n_classes, mu, t, tau, lmb, F, h, w, w0, K, feature_mapping, feature_parameters):
    """
Learning

    This function efficientle learns classifier parameters

    Input
    -----

    x: instance

    N: number of subgradients

    n_classes: number of classes

    mu: classifier parameter

    tau: mean vector estimate

    lmb: confidence vector

    F: matrix that is used to obtain local approximations of function varphi

    h: vector that is used to obtain local approximations of function varphi

    w, w0: Nesterov's-SG parameters

    K: number of iterations Nesterov's-SG

    feature_mapping: 'linear' or 'RFF'

    feature_parameters:
        if feature_mapping == 'linear': feature_parameters = []
        if feature_mapping == 'RFF': feature_parameters = [D, u] where
            D = number of random Fourier components
            u = random Fourier components

    Output
    ------

    mu: updated classifier parameter

    F: updated matrix that is used to obtain local approximations of function varphi

    h: updated vector that is used to obtain local approximations of function varphi

    R_Ut: upper bounds

    varphi: function that is then used at prediction

    w, w0: Nesterov's-SG parameters

    """
    theta = 1
    theta0 = 1
    d = len(x)
    muaux = mu
    R_Ut = 0
    M = np.zeros((n_classes, len(mu)))
    for j in range(0, n_classes):
        M[j, :] = feature_vector(x, j, n_classes, feature_mapping, feature_parameters)
    for j in range(0, n_classes):
        aux = list(itertools.combinations([*range(0, n_classes, 1)], j+1))
        for k in range(0, np.size(aux, 0)):
            idx = np.zeros((1, n_classes))
            a = aux[k]
            for mm in range(0, len(a)):
                idx[0, a[mm]] = 1
            a = (np.dot(idx, M))/(j+1)
            b = np.size(F, 0)
            F2 = np.zeros((b+1, len(mu)))
            h2 = np.zeros((b+1, 1))
            for mm in range(0, b):
                for jj in range(0, len(mu)):
                    F2[mm, jj] = F[mm, jj]
                h2[mm, :] = h[mm, :]
            F2[-1, :] = a
            b = -1/(j+1)
            h2[-1, 0] = b
            F = F2
            h = h2
    if t == 0:
        F = np.delete(F, 0, 0)
        h = np.delete(h, 0, 0)
    v = np.dot(F, muaux) + h
    varphi = max(v)[0]
    regularization = sum(lmb*abs(muaux))
    R_Ut_best_value = 1  - np.dot(np.transpose(tau), muaux)[0] + varphi + regularization
    F_count = np.zeros((len(F[:, 0]), 1))
    for i in range(0, K):
        muaux = w + theta*((1/theta0) - 1)*(w-w0)
        v = np.dot(F, muaux) + h
        varphi = max(v)[0]
        idx_mv = np.where(v == varphi)
        if len(idx_mv[0])>1:
            fi = F[[idx_mv[0][0]], :]
            F_count[[idx_mv[0][0]]] = F_count[[idx_mv[0][0]]] + 1
        else:
            fi = F[idx_mv[0], :]
            F_count[idx_mv[0]] = F_count[idx_mv[0]] + 1
        subgradient_regularization = lmb*np.sign(muaux)
        regularization = sum(lmb*abs(muaux))
        g = - tau + np.transpose(fi) + subgradient_regularization
        theta0 = theta
        theta = 2/(i+2)
        alpha = 1/((i+2)**(3/2))
        w0 = w
        w = muaux - alpha*g
        R_Ut = 1 - np.dot(np.transpose(tau), muaux)[0] + varphi + regularization
        if R_Ut < R_Ut_best_value:
            R_Ut_best_value = R_Ut
            mu = muaux
    v = np.dot(F, muaux) + h
    varphi = max(v)[0]
    regularization = sum(lmb*abs(w))
    R_Ut = 1 - np.dot(np.transpose(tau), w)[0] + varphi + regularization
    if R_Ut < R_Ut_best_value:
        R_Ut_best_value = R_Ut
        mu = w;
    if len(F[:, 0]) > N:
        idx_F_count = np.where(F_count==0)
        if len(idx_F_count) > len(F[:, 0]) - N:
            for j in range(0, len(idx_F_count[0])-N):
                t = len(idx_F_count[0])-1-j
                F = np.delete(F, idx_F_count[0][t], idx_F_count[1][t])
                h = np.delete(h, idx_F_count[0][t], 0)
        else:
            for j in range(0, len(idx_F_count[0])- len(F[:, 0]) + N):
                t = len(idx_F_count[0])-1-j
                F = np.delete(F, idx_F_count[0][t], idx_F_count[1][t])
                h = np.delete(h, idx_F_count[0][t], 0)
    return mu, F, h, R_Ut, varphi, w, w0
