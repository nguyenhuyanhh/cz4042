"""Project 1b: Approximation."""

from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from nn_utils import normalize, scale, shuffle_data

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

# init paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data_b')

np.random.seed(10)
EPOCHS = 1000
BATCH_SIZE = 32
FL_X = theano.config.floatX


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


def main(train_x, train_y, test_x, test_y, no_hidden=30, learning_rate=1e-4):
    """Entry point for module."""
    no_features = train_x.shape[1]
    x_mat = T.matrix('x')  # data sample
    d_mat = T.matrix('d')  # desired output

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden) * .01, FL_X)
    b_o = theano.shared(np.random.randn() * .01, FL_X)
    w_h1 = theano.shared(np.random.randn(
        no_features, no_hidden) * .01, FL_X)
    b_h1 = theano.shared(np.random.randn(no_hidden) * 0.01, FL_X)

    # learning rate
    alpha = theano.shared(learning_rate, FL_X)

    # define mathematical expressions
    h1_out = T.nnet.sigmoid(T.dot(x_mat, w_h1) + b_h1)
    y_vec = T.dot(h1_out, w_o) + b_o
    cost = T.abs_(T.mean(T.sqr(d_mat - y_vec)))
    accuracy = T.mean(d_mat - y_vec)

    # define gradients
    dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

    # compile train and test functions
    train = theano.function(
        inputs=[x_mat, d_mat],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h],
                 [b_h1, b_h1 - alpha * db_h]],
        allow_input_downcast=True
    )
    test = theano.function(
        inputs=[x_mat, d_mat],
        outputs=[y_vec, cost, accuracy],
        allow_input_downcast=True
    )

    # train and test
    train_cost = np.zeros(EPOCHS)
    test_cost = np.zeros(EPOCHS)
    test_accuracy = np.zeros(EPOCHS)

    min_error = 1e+15
    best_iter = 0
    best_w_o = np.zeros(no_hidden)
    best_w_h1 = np.zeros([no_features, no_hidden])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden)

    alpha.set_value(learning_rate)

    for i in tqdm(range(EPOCHS)):
        train_x, train_y = shuffle_data(train_x, train_y)
        train_cost[i] = train(train_x, np.transpose(train_y))
        _, test_cost[i], test_accuracy[i] = test(
            test_x, np.transpose(test_y))
        if test_cost[i] < min_error:
            best_iter = i
            min_error = test_cost[i]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)

    _, best_cost, best_accuracy = test(test_x, np.transpose(test_y))

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' %
          (best_cost, best_accuracy, best_iter))

    return train_cost, test_cost, test_accuracy


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
        train_cost, test_cost, _ = main(
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
            train_cost, test_cost, _ = main(
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
        plt.savefig('p1b_sample_train.png')

    # Plot for validation errors
    if plot_vald:
        plt.figure()
        for item, value in zip(validation_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Validation Error')
        plt.legend()
        plt.savefig('p1b_sample_validation.png')

    # Plot for test errors
    if plot_test:
        plt.figure()
        for item, value in zip(validation_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Test Error')
        plt.legend()
        plt.savefig('p1b_sample_test.png')

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
    search('no_hidden', [60], plot_train=False,
           plot_vald=False, plot_test=True)
    exit()
