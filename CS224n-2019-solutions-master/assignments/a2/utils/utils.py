#!/usr/bin/env python

import numpy as np


def normalizeRows(X):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = X.shape[0]
    X /= np.linalg.norm(X, axis=1).reshape((N, 1)) + 1e-30
    return X


def softmax(x):
    """Compute the softmax function for each row of the input_tensor x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        tmp = np.max(x, axis=1).reshape((x.shape[0], 1))
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp
    else:
        # Vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x
