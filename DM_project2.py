# -*- coding: utf-8 -*-
"""
@author: Casablanca
"""

import numpy as np

C_ALPHA = 1.0
C_BETA1 = 0.34
C_BETA2 = 0.999999999
C_EPSILON = 1e-8


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # normalize
    X = X/float(X.max())

    return X


def project_L2(w, a):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (np.sqrt(a) * np.linalg.norm(w, 2)))


def mapper(key, value):
    # key: None
    # value: one line of input file

    features = np.genfromtxt(value, delimiter=' ').T
    y = features[:1].T
    X = features[1:].T
    X = transform(X)

    assert X.shape[0] == y.shape[0]
    w = np.zeros(X.shape[1])

    # Adam
    m = np.ones(X.shape[1])
    v = np.ones(X.shape[1])
    for t in range(X.shape[0]):
        if y[t] * np.dot(w, X[t, :]) < 1:
            eta = 1. / np.sqrt((t + 1.))
            m = C_BETA1 * m + (1. - C_BETA1) * -y[t] * X[t, :]
            m_ = m / (1. - C_BETA1 ** (t + 1.))
            v = C_BETA2 * v + (1. - C_BETA2) * (-y[t] * X[t, :]) ** 2
            v_ = v / (1. - C_BETA2 ** (t + 1.))

            w -= eta * m_ / np.sqrt(v_ + C_EPSILON)

            w = project_L2(w, C_ALPHA)

    yield "key", w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    w = np.array(values).mean(axis=0)
    w_output = w.T

    yield w_output
