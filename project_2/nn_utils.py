"""Utils for Project 2."""

import os

import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data/')


def one_hot(x_arr, n_d):
    """Generate one-hot vectors of data."""
    if isinstance(x_arr, list):
        x_arr = np.array(x_arr)
    x_arr = x_arr.flatten()
    o_h = np.zeros((len(x_arr), n_d))
    o_h[np.arange(len(x_arr)), x_arr] = 1
    return o_h


def load_mnist(ntrain=60000, ntest=10000, onehot=True):
    """Load MNIST data."""
    file_ = open(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=file_, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    file_ = open(os.path.join(DATA_DIR, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=file_, dtype=np.uint8)
    train_y = loaded[8:].reshape((60000))

    file_ = open(os.path.join(DATA_DIR, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=file_, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    file_ = open(os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=file_, dtype=np.uint8)
    test_y = loaded[8:].reshape((10000))

    train_x = train_x / 255.
    test_x = test_x / 255.

    train_x = train_x[:ntrain]
    train_y = train_y[:ntrain]

    test_x = test_x[:ntest]
    test_y = test_y[:ntest]

    if onehot:
        train_y = one_hot(train_y, 10)
        test_y = one_hot(test_y, 10)
    else:
        train_y = np.asarray(train_y)
        test_y = np.asarray(test_y)

    return train_x, test_x, train_y, test_y


def shuffle_data(samples, labels):
    """Shuffle the data."""
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels
