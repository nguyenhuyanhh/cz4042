"""Project 1a: Classification."""

from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from nn_utils import init_bias, init_weights, scale, sgd, shuffle_data

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass


# init path
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data_a')

# number of epochs
EPOCHS = 1000


def load_train_test():
    """Load training and testing data."""
    # train data
    train_input = np.loadtxt(os.path.join(
        DATA_DIR, 'sat_train.txt'), delimiter=' ')
    train_x, train_y_tmp = train_input[:, :36], train_input[:, -1].astype(int)
    train_x_min, train_x_max = np.min(train_x, axis=0), np.max(train_x, axis=0)
    train_x = scale(train_x, train_x_min, train_x_max)
    train_y_tmp[train_y_tmp == 7] = 6  # convert class label 7 to 6
    train_y = np.zeros((train_y_tmp.shape[0], 6))
    train_y[np.arange(train_y_tmp.shape[0]), train_y_tmp - 1] = 1

    # test data
    test_input = np.loadtxt(os.path.join(
        DATA_DIR, 'sat_test.txt'), delimiter=' ')
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


def nn_3_layer(hl_neuron=10, decay=1e-6):
    """Neural network with 3 layers.

    Arguments:
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

    return train, predict


def nn_4_layer(hl_neuron=10, decay=1e-6):
    """Neural network with 4 layers.

    Arguments:
        hl_neuron: int - number of neurons for hidden layer
        decay: float - decay parameter
    """
    learning_rate = 0.01

    # theano expressions
    x_mat = T.matrix()  # features
    y_mat = T.matrix()  # output

    # weights and biases from input to hidden layer 1
    weight_1, bias_1 = init_weights(36, hl_neuron), init_bias(hl_neuron)
    # weights and biases from hidden layer 1 to hidden layer 2
    weight_2, bias_2 = init_weights(hl_neuron, hl_neuron), init_bias(hl_neuron)
    # weights and biases from hidden layer 2 to output layer
    weight_3, bias_3 = init_weights(hl_neuron, 6, logistic=False), init_bias(6)

    hidden_1 = T.nnet.sigmoid(T.dot(x_mat, weight_1) + bias_1)
    hidden_2 = T.nnet.sigmoid(T.dot(hidden_1, weight_2) + bias_2)
    output_1 = T.nnet.softmax(T.dot(hidden_2, weight_3) + bias_3)

    y_x = T.argmax(output_1, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(output_1, y_mat)) + \
        decay * (T.sum(T.sqr(weight_1) + T.sum(T.sqr(weight_2) +
                                               T.sum(T.sqr(weight_3)))))
    params = [weight_1, bias_1, weight_2, bias_2, weight_3, bias_3]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(
        inputs=[x_mat, y_mat], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(
        inputs=[x_mat], outputs=y_x, allow_input_downcast=True)

    return train, predict


def train_test(batch_size=4, hl_neuron=10, decay=1e-6, layer_4=False):
    """Train and test the neural network with data.

    Arguments:
        batch_size: int - batch size for mini-batch gradient descent
        hl_neuron: int - number of neurons for hidden layer
        decay: float - decay parameter
    """
    # init functions and variables
    if layer_4:
        train, predict = nn_4_layer(hl_neuron, decay)
    else:
        train, predict = nn_3_layer(hl_neuron, decay)
    train_x, train_y, test_x, test_y = load_train_test()
    n_tr = len(train_x)
    test_accuracy = []
    train_cost = []
    timings = []
    start_time = 0

    # train and test
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

    # print results
    print('%.1f accuracy at %d iterations' %
          (np.max(test_accuracy) * 100, np.argmax(test_accuracy) + 1))
    average_time = np.average(timings)
    print('average time per update: {}'.format(average_time))

    return (train_cost, test_accuracy, average_time)


def search(param, search_space, layer_4=False, plot_cost=True,
           plot_acc=True, plot_time=True, plot_max_acc=False):
    """Search for the optimal parameters, and graph the results."""
    cost_args = []
    accuracy_args = []
    average_times = []

    for value in search_space:
        nn_args = {param: value, 'layer_4': layer_4}
        train_cost, test_accuracy, timing = train_test(**nn_args)
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

    # Q5
    # search('batch_size', [32], layer_4=True, plot_time=False)
    exit()
