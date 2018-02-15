# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, train_test_split
import random

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i, :].reshape(d, 1), x_train, y_train, tau) for i in range(N_test)])
        losses[j] = ((predictions.flatten() - y_test.flatten()) ** 2).mean()
    return losses


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    x_T = np.mat(x_train)
    y_T = np.mat(y_train).T
    m = np.shape(x_T)[0]
    test = test_datum.T * np.ones([m, 1])
    A = np.mat(l2(test, x_train)/(-2.0*tau**2))
    A_max = A.max()
    A = np.multiply(np.exp(A-A_max), np.mat(np.eye(m)))
    A = A/np.sum(A)
    xTAx = np.dot(np.dot(x_T.T, A), x_T) + lam*np.mat(np.eye(d))
    xTAy = np.dot(np.dot(x_T.T, A), y_T)
    weights = np.linalg.solve(xTAx, xTAy)
    return np.dot(test_datum.T, weights)
    ## TODO


def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO

    my_list = [i for i in range(x.shape[0])]
    np.random.shuffle(my_list)
    for n in range(k):
        fold_test_index = []
        fold_train_index = []
        for j in range(x.shape[0]):
            if j % k == n:
                fold_test_index.append(my_list[j])
            else:
                fold_train_index.append(my_list[j])
        train_x, valid_x = x[fold_train_index], x[fold_test_index]
        train_y, valid_y = y[fold_train_index], y[fold_test_index]
        if n == 0:
            loss = np.array(run_on_fold(valid_x, valid_y, train_x, train_y, taus))
        else:
            loss += np.array(run_on_fold(valid_x, valid_y, train_x, train_y, taus))
    return loss/k

    '''
    i = 0
    Kf = KFold(k, shuffle= True)
    loss = np.array([])
    for index_train, index_valid in Kf.split(x, y):
        train_x, valid_x = x[index_train], x[index_valid]
        train_y, valid_y = y[index_train], y[index_valid]
        if i == 0:
            loss = np.array(run_on_fold(valid_x, valid_y, train_x, train_y, taus))
        else:
            loss += np.array(run_on_fold(valid_x, valid_y, train_x, train_y, taus))
        i = i + 1
    loss = loss/i
    return loss
    '''
    ## TODO




if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(losses)
    plt.show()
    print("min loss = {}".format(losses.min()))
