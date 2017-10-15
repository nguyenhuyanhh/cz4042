"""Utils for Project 1a: Classification."""

import numpy as np
import theano
import theano.tensor as T

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass


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
    return (x_raw - x_min) / (x_max - x_min)


def normalize(x_raw, x_mean, x_std):
    """Normalize the data."""
    return (x_raw - x_mean) / x_std


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
