"""Project 1b: Approximation."""

from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np

from nn_utils import normalize, scale, shuffle_data
from nn_approx import nn_3_layer, nn_4_layer, nn_5_layer

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data_b')

np.random.seed(10)
EPOCHS = 1000


def load_train_test():
    """Load training and testing data."""
    # read and divide data into test and train sets
    cal_housing = np.loadtxt(os.path.join(
        DATA_DIR, 'cal_housing.data'), delimiter=',')
    x_data, y_data = cal_housing[:, :8], cal_housing[:, -1]
    y_data = (np.asmatrix(y_data)).transpose()
    x_data, y_data = shuffle_data(x_data, y_data)

    # separate train and test data
    test_len = 3 * x_data.shape[0] // 10  # 3:7 test:train split
    test_x, test_y = x_data[:test_len], y_data[:test_len]
    train_x, train_y = x_data[test_len:], y_data[test_len:]

    # scale and normalize data
    train_x_max, train_x_min = np.max(train_x, axis=0), np.min(train_x, axis=0)
    test_x_max, test_x_min = np.max(test_x, axis=0), np.min(test_x, axis=0)
    train_x = scale(train_x, train_x_min, train_x_max)
    test_x = scale(test_x, test_x_min, test_x_max)
    train_x_mean, train_x_std = np.mean(
        train_x, axis=0), np.std(train_x, axis=0)
    test_x_mean, test_x_std = np.mean(test_x, axis=0), np.std(test_x, axis=0)
    train_x = normalize(train_x, train_x_mean, train_x_std)
    test_x = normalize(test_x, test_x_mean, test_x_std)

    return train_x, train_y, test_x, test_y


def cross_validation(train_x, train_y, fold=5, no_hidden=30, learning_rate=1e-4):
    """Wrapper function for cross validation."""
    num_features = train_x.shape[0]
    train_errors = []
    validation_errors = []
    for i in range(fold):
        print('Fold {}:'.format(i + 1))
        # get train and test data for each fold
        start = (num_features // fold) * i
        end = (num_features // fold) * (i + 1)
        tmp_x = np.vstack((train_x[:start], train_x[end:]))
        tmp_y = np.vstack((train_y[:start], train_y[end:]))
        test_x, test_y = train_x[start:end], train_y[start:end]
        nn_args = {'no_hidden': no_hidden, 'learning_rate': learning_rate}
        # train and test
        train_cost, test_cost, _ = nn_3_layer(
            tmp_x, tmp_y, test_x, test_y, **nn_args)
        train_errors += [train_cost]
        validation_errors += [test_cost]

    return np.average(train_errors, axis=0), np.average(validation_errors, axis=0)


def search(param, search_space, plot_train=True, plot_vald=True, plot_test=False):
    """Search for the optimal parameters, and graph the results."""
    train_x, train_y, test_x, test_y = load_train_test()
    train_args = []
    validation_args = []

    for value in search_space:
        nn_args = {param: value}
        if plot_train or plot_vald:  # train and validate only
            train_cost, test_cost = cross_validation(
                train_x, train_y, **nn_args)
        elif plot_test:  # test only
            train_cost, test_cost, _ = nn_3_layer(
                train_x, train_y, test_x, test_y, **nn_args)
        train_args += [train_cost]
        validation_args += [test_cost]

    # Plot for training errors
    if plot_train:
        plt.figure()
        for item, value in zip(train_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Training Error')
        plt.legend()
        plt.savefig(os.path.join(CUR_DIR, 'p1b_sample_train.png'))

    # Plot for validation errors
    if plot_vald:
        plt.figure()
        for item, value in zip(validation_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Validation Error')
        plt.legend()
        plt.savefig(os.path.join(CUR_DIR, 'p1b_sample_validation.png'))

    # Plot for test errors
    if plot_test:
        plt.figure()
        for item, value in zip(validation_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Test Error')
        plt.legend()
        plt.savefig(os.path.join(CUR_DIR, 'p1b_sample_test.png'))

    plt.show()


def compare(plot_3=True, plot_4=True, plot_5=True):
    """Compare the 3-, 4-, and 5-layer network."""
    train_x, train_y, test_x, test_y = load_train_test()
    test_args = {}

    # train and test
    if plot_3:
        _, test_cost, _ = nn_3_layer(
            train_x, train_y, test_x, test_y, no_hidden=60)
        test_args['3-layer'] = test_cost
    if plot_4:
        _, test_cost, _ = nn_4_layer(
            train_x, train_y, test_x, test_y, no_hidden=60)
        test_args['4-layer'] = test_cost
    if plot_5:
        _, test_cost, _ = nn_5_layer(
            train_x, train_y, test_x, test_y, no_hidden=60)
        test_args['5-layer'] = test_cost

    # plot the comparison
    plt.figure()
    for label in sorted(test_args):
        plt.plot(range(EPOCHS), test_args[label], label=label)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Error')
    plt.legend()
    plt.savefig(os.path.join(CUR_DIR, 'p1b4_compare.png'))
    plt.show()


if __name__ == '__main__':
    # Q2a
    # search('learning_rate', [1e-5, 0.5e-4, 1e-4, 0.5e-3, 1e-3])

    # Q2b
    # search('learning_rate', [1e-4], plot_train=False,
    #        plot_vald=False, plot_test=True)

    # Q3a
    # search('no_hidden', [20, 30, 40, 50, 60], plot_vald=False)

    # Q3b
    # search('no_hidden', [60], plot_train=False,
    #        plot_vald=False, plot_test=True)

    # Q4
    compare()
    exit()
