"""Project 1a: Classification."""

from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from nn_utils import (init_bias, init_weights,
                      load_train_test, sgd, shuffle_data)

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass


# init path
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# number of epochs
EPOCHS = 1000


def nn_3_layer(batch_size=4, hl_neuron=10, decay=1e-6):
    """Neural network with 3 layers.

    Arguments:
        batch_size: int - batch size for mini-batch gradient descent
        hl_neuron: int - number of neurons for hidden layer
        decay: float - decay parameter
    """
    learning_rate = 0.01

    # theano expressions
    x_mat = T.matrix()  # features
    y_mat = T.matrix()  # output

    # weights and biases from input to hidden layer
    weight_1, bias_1 = init_weights(36, hl_neuron), init_bias(hl_neuron)
    # weights and biases from hidden to output layer
    weight_2, bias_2 = init_weights(hl_neuron, 6, logistic=False), init_bias(6)

    hidden_1 = T.nnet.sigmoid(T.dot(x_mat, weight_1) + bias_1)
    output_1 = T.nnet.softmax(T.dot(hidden_1, weight_2) + bias_2)

    y_x = T.argmax(output_1, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(output_1, y_mat)) + \
        decay * (T.sum(T.sqr(weight_1) + T.sum(T.sqr(weight_2))))
    params = [weight_1, bias_1, weight_2, bias_2]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(
        inputs=[x_mat, y_mat], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(
        inputs=[x_mat], outputs=y_x, allow_input_downcast=True)

    # train and test
    train_x, train_y, test_x, test_y = load_train_test()
    n_tr = len(train_x)
    test_accuracy = []
    train_cost = []
    timings = []
    start_time = 0
    for _ in tqdm(range(EPOCHS)):
        train_x, train_y = shuffle_data(train_x, train_y)
        cost = 0.0

        for start, end in zip(range(0, n_tr, batch_size), range(batch_size, n_tr, batch_size)):
            start_time = time.time()
            cost += train(train_x[start:end], train_y[start:end])
            timings.append((time.time() - start_time) * 1e6)
        train_cost = np.append(train_cost, cost / (n_tr // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(
            np.argmax(test_y, axis=1) == predict(test_x)))

    print('%.1f accuracy at %d iterations' %
          (np.max(test_accuracy) * 100, np.argmax(test_accuracy) + 1))
    average_time = np.average(timings)
    print('average time per update: {} microseconds'.format(average_time))

    return (train_cost, test_accuracy, average_time)


def search(param, search_space, plot_cost=True, plot_acc=True, plot_time=True, plot_max_acc=False):
    """Search for the optimal parameters, and graph the results."""
    cost_args = []
    accuracy_args = []
    average_times = []

    for value in search_space:
        nn_args = {param: value}
        train_cost, test_accuracy, timing = nn_3_layer(**nn_args)
        cost_args += [train_cost]
        accuracy_args += [test_accuracy]
        average_times += [timing]

    # plot for cost/iterations
    if plot_cost:
        plt.figure()
        for item, value in zip(cost_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('iterations')
        plt.ylabel('cross-entropy')
        plt.title('training cost')
        plt.legend()
        plt.savefig(os.path.join(CUR_DIR, 'p1a_sample_cost.png'))

    # plot for accuracy/iterations
    if plot_acc:
        plt.figure()
        for item, value in zip(accuracy_args, search_space):
            plt.plot(range(EPOCHS), item, label="{}={}".format(param, value))
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.title('test accuracy')
        plt.legend()
        plt.savefig(os.path.join(CUR_DIR, 'p1a_sample_accuracy.png'))

    # plot for time/param
    if plot_time:
        plt.figure()
        plt.plot(search_space, average_times, 'bx-')
        plt.xlabel(param)
        plt.ylabel('time to update in microseconds')
        plt.title('update time vs {}'.format(param))
        plt.savefig(os.path.join(CUR_DIR, 'p1a_sample_times.png'))

    # plot for accuracy/param
    if plot_max_acc:
        plt.figure()
        plt.plot(range(len(accuracy_args)), [
            np.max(i) for i in accuracy_args], 'bx-')
        plt.gca().xaxis.set_ticks(range(len(accuracy_args)))
        plt.gca().xaxis.set_ticklabels(search_space)
        plt.xlabel(param)
        plt.ylabel('accuracy in %')
        plt.title('accuracy vs {}'.format(param))
        plt.savefig(os.path.join(CUR_DIR, 'p1a_{}_accuracy.png'.format(param)))

    plt.show()


if __name__ == '__main__':
    # Q2
    # search('batch_size', [4, 8, 16, 32, 64])

    # Q3
    search('hl_neuron', [5, 10, 15, 20, 25])

    # Q4
    # search('decay', [0, 1e-12, 1e-9, 1e-6, 1e-3],
    #        plot_acc=False, plot_time=False, plot_max_acc=True)

    exit()
