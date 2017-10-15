"""Project 1b: Approximation."""

import time

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from nn_utils import normalize, scale, shuffle_data

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

np.random.seed(10)
epochs = 1000
batch_size = 32
floatX = theano.config.floatX


def load_train_test():
    """Load training and testing data."""
    # read and divide data into test and train sets
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    x_data, y_data = cal_housing[:, :8], cal_housing[:, -1]
    y_data = (np.asmatrix(y_data)).transpose()

    x_data, y_data = shuffle_data(x_data, y_data)

    # separate train and test data
    m = 3 * x_data.shape[0] // 10
    test_x, test_y = x_data[:m], y_data[:m]
    train_x, train_y = x_data[m:], y_data[m:]

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
    no_samples = T.scalar('no_samples')

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(no_hidden) * .01, floatX)
    b_o = theano.shared(np.random.randn() * .01, floatX)
    w_h1 = theano.shared(np.random.randn(
        no_features, no_hidden) * .01, floatX)
    b_h1 = theano.shared(np.random.randn(no_hidden) * 0.01, floatX)

    # learning rate
    alpha = theano.shared(learning_rate, floatX)

    # Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x_mat, w_h1) + b_h1)
    y_vec = T.dot(h1_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d_mat - y_vec)))
    accuracy = T.mean(d_mat - y_vec)

    # define gradients
    dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

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

    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    min_error = 1e+15
    best_iter = 0
    best_w_o = np.zeros(no_hidden)
    best_w_h1 = np.zeros([no_features, no_hidden])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden)

    alpha.set_value(learning_rate)
    print(alpha.get_value(), no_hidden)

    t = time.time()
    for iter in tqdm(range(epochs)):

        train_x, train_y = shuffle_data(train_x, train_y)
        train_cost[iter] = train(train_x, np.transpose(train_y))
        pred, test_cost[iter], test_accuracy[iter] = test(
            test_x, np.transpose(test_y))

        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)

    best_pred, best_cost, best_accuracy = test(test_x, np.transpose(test_y))

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' %
          (best_cost, best_accuracy, best_iter))

    return train_cost, test_cost, test_accuracy


def cross_validation(train_x, train_y, fold=5, no_hidden=30, learning_rate=1e-4):
    """ wrapper function for cross validation """
    num_features = train_x.shape[0]
    train_errors = []
    validation_errors = []
    for i in range(fold):
        start = (num_features // fold) * i
        end = (num_features // fold) * (i + 1)
        tmp_x = np.vstack((train_x[:start], train_x[end:]))
        tmp_y = np.vstack((train_y[:start], train_y[end:]))
        test_x, test_y = train_x[start:end], train_y[start:end]
        train_cost, test_cost, test_accuracy = main(tmp_x, tmp_y, test_x, test_y,
                                                    no_hidden=no_hidden, learning_rate=learning_rate)
        train_errors += [train_cost]
        validation_errors += [test_cost]

    return np.average(train_errors, axis=0), np.average(validation_errors, axis=0)


if __name__ == '__main__':

    search_space = [60]
    train_args = []
    validation_args = []

    train_x, train_y, test_x, test_y = load_train_test()
    for no_hidden in search_space:
        train_cost, test_cost, _ = main(
            train_x, train_y, test_x, test_y, no_hidden=no_hidden)
        #train_cost, test_cost = cross_validation(train_x, train_y, no_hidden=no_hidden)
        train_args += [train_cost]
        validation_args += [test_cost]

    # Plots
    plt.figure()
    for item, value in zip(train_args, search_space):
        plt.plot(range(epochs), item, label="neurons={}".format(value))
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error')
    plt.legend()
    # plt.savefig('p1b3_neuron_train.png')

    plt.figure()
    for item, value in zip(validation_args, search_space):
        plt.plot(range(epochs), item, label="neurons={}".format(value))
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Error')
    plt.legend()
    plt.savefig('p1b3_neuron_test.png')
    plt.show()
