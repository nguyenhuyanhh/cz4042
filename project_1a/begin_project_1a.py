"""Project 1a: Classification."""

import time
from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T


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
    train_input = np.loadtxt('sat_train.txt', delimiter=' ')
    train_x, train_y_tmp = train_input[:, :36], train_input[:, -1].astype(int)
    train_x_min, train_x_max = np.min(train_x, axis=0), np.max(train_x, axis=0)
    train_x = scale(train_x, train_x_min, train_x_max)
    train_y_tmp[train_y_tmp == 7] = 6  # convert class label 7 to 6
    train_y = np.zeros((train_y_tmp.shape[0], 6))
    train_y[np.arange(train_y_tmp.shape[0]), train_y_tmp - 1] = 1

    # test data
    test_input = np.loadtxt('sat_test.txt', delimiter=' ')
    test_x, test_y_tmp = test_input[:, :36], test_input[:, -1].astype(int)
    test_x_min, test_x_max = np.min(test_x, axis=0), np.max(test_x, axis=0)
    test_x = scale(test_x, test_x_min, test_x_max)
    test_y_tmp[test_y_tmp == 7] = 6
    test_y = np.zeros((test_y_tmp.shape[0], 6))
    test_y[np.arange(test_y_tmp.shape[0]), test_y_tmp - 1] = 1

    assert train_x.shape == (4435, 36)
    assert train_y.shape == (4435, 6)
    assert test_x.shape == (2000, 36)
    assert test_y.shape == (2000, 6)

    return train_x, train_y, test_x, test_y


def main(batch_size=32, hl_neuron=10, decay=1e-6):
    """Entry point for script."""
    learning_rate = 0.01
    epochs = 1000

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

        for start, end in izip(range(0, n_tr, batch_size), range(batch_size, n_tr, batch_size)):
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

    search_space = [4, 8, 16, 32, 64]

    for batch_size in search_space:
        train_cost, test_accuracy, timing = main(batch_size=batch_size)
        cost_args += [train_cost]
        accuracy_args += [test_accuracy]
        average_times += [timing]

    # Plots
    plt.figure()
    for item, value in izip(cost_args, search_space):
        plt.plot(range(1000), item, label="batch={}".format(value))
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.legend()
    plt.savefig('p1a_sample_cost.png')

    plt.figure()
    for item, value in izip(accuracy_args, search_space):
        plt.plot(range(1000), item, label="batch={}".format(value))
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.legend()
    plt.savefig('p1a_sample_accuracy.png')

    plt.figure()
    plt.plot(search_space, average_times, 'bx-')
    plt.xlabel('batch size')
    plt.ylabel('time to update in microseconds')
    plt.title('update time vs batch size')
    plt.savefig('p1a_sample_times.png')

    # forced garbage collection test
    train_cost = []
    test_accuracy = []
    cost_args = []
    accuracy_args = []

    plt.show()
