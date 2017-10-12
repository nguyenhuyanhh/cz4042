"""Project 1a: Classification."""

from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from nn_utils import (init_bias, init_weights, load_train_test, scale, sgd,
                      shuffle_data)

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass


# init path
CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def nn_3_layer(batch_size=4, hl_neuron=10, decay=1e-6, learning_rate=0.01, epochs=1000):
    """Neural network with 3 layers."""
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
    for i in range(epochs):
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
    print('average time per update: {}'.format(average_time))

    return (train_cost, test_accuracy, average_time)


if __name__ == '__main__':
    train_cost = []
    test_accuracy = []
    cost_args = []
    accuracy_args = []
    average_times = []

    search_space = [5, 10, 15, 20, 25]

    for neuron_num in search_space:
        train_cost, test_accuracy, timing = nn_3_layer(hl_neuron=neuron_num)
        cost_args += [train_cost]
        accuracy_args += [test_accuracy]
        average_times += [timing]

    # Plots
    plt.figure()
    for item, value in zip(cost_args, search_space):
        plt.plot(range(1000), item, label="neurons={}".format(value))
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.legend()
    plt.savefig(os.path.join(CUR_DIR, 'p1a_sample_cost.png'))

    plt.figure()
    for item, value in zip(accuracy_args, search_space):
        plt.plot(range(1000), item, label="neurons={}".format(value))
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.legend()
    plt.savefig(os.path.join(CUR_DIR, 'p1a_sample_accuracy.png'))

    plt.figure()
    plt.plot(search_space, average_times, 'bx-')
    plt.xlabel('batch size')
    plt.ylabel('time to update in microseconds')
    plt.title('update time vs batch size')
    plt.savefig(os.path.join(CUR_DIR, 'p1a_sample_times.png'))

    # forced garbage collection test
    train_cost = []
    test_accuracy = []
    cost_args = []
    accuracy_args = []

    plt.show()
