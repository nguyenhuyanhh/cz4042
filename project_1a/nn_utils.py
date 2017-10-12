"""Utils for Project 1a: Classification."""

from __future__ import print_function

import os

import numpy as np
import theano
import theano.tensor as T

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

# init path
CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def init_bias(n_bias=1):
    """Initialize bias."""
    return theano.shared(np.zeros(n_bias), theano.config.floatX)


def init_weights(n_in=1, n_out=1, logistic=True):
    """Initialize weights."""
    weight_values = np.asarray(
        np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),
        dtype=theano.config.floatX
    )
    if logistic:
        weight_values *= 4
    return theano.shared(value=weight_values, name='W', borrow=True)


def scale(x_raw, x_min, x_max):
    """Scale the data."""
    return (x_raw - x_min) / (x_max - np.min(x_raw, axis=0))


def sgd(cost, params, learning_rate=0.01):
    """Update parameters."""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append([param, param - grad * learning_rate])
    return updates


def shuffle_data(samples, labels):
    """Shuffle the data."""
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def load_train_test():
    """Load training and testing data."""
    # train data
    train_input = np.loadtxt(os.path.join(
        CUR_DIR, 'sat_train.txt'), delimiter=' ')
    train_x, train_y_tmp = train_input[:, :36], train_input[:, -1].astype(int)
    train_x_min, train_x_max = np.min(train_x, axis=0), np.max(train_x, axis=0)
    train_x = scale(train_x, train_x_min, train_x_max)
    train_y_tmp[train_y_tmp == 7] = 6  # convert class label 7 to 6
    train_y = np.zeros((train_y_tmp.shape[0], 6))
    train_y[np.arange(train_y_tmp.shape[0]), train_y_tmp - 1] = 1

    # test data
    test_input = np.loadtxt(os.path.join(
        CUR_DIR, 'sat_test.txt'), delimiter=' ')
    test_x, test_y_tmp = test_input[:, :36], test_input[:, -1].astype(int)
    test_x_min, test_x_max = np.min(test_x, axis=0), np.max(test_x, axis=0)
    test_x = scale(test_x, test_x_min, test_x_max)
    test_y_tmp[test_y_tmp == 7] = 6  # convert class label 7 to 6
    test_y = np.zeros((test_y_tmp.shape[0], 6))
    test_y[np.arange(test_y_tmp.shape[0]), test_y_tmp - 1] = 1

    assert train_x.shape == (4435, 36)
    assert train_y.shape == (4435, 6)
    assert test_x.shape == (2000, 36)
    assert test_y.shape == (2000, 6)

    return train_x, train_y, test_x, test_y
