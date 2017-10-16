"""Neural networks for approximation."""

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
from tqdm import tqdm

from nn_utils import shuffle_data

np.random.seed(10)
EPOCHS = 1000
BATCH_SIZE = 32
FL_X = theano.config.floatX


def nn_3_layer(train_x, train_y, test_x, test_y, no_hidden=30, learning_rate=1e-4):
    """Train and test the 3-layer neural network."""
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


def nn_4_layer(train_x, train_y, test_x, test_y, no_hidden=30, learning_rate=1e-4):
    """Train and test the 4-layer neural network."""
    no_features = train_x.shape[1]
    x_mat = T.matrix('x')  # data sample
    d_mat = T.matrix('d')  # desired output

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(20) * .01, FL_X)
    b_o = theano.shared(np.random.randn() * .01, FL_X)
    w_h1 = theano.shared(np.random.randn(
        no_features, no_hidden) * .01, FL_X)
    b_h1 = theano.shared(np.random.randn(no_hidden) * 0.01, FL_X)
    w_h2 = theano.shared(np.random.randn(
        no_hidden, 20) * .01, FL_X)
    b_h2 = theano.shared(np.random.randn(20) * 0.01, FL_X)

    # learning rate
    alpha = theano.shared(learning_rate, FL_X)

    # define mathematical expressions
    h1_out = T.nnet.sigmoid(T.dot(x_mat, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    y_vec = T.dot(h2_out, w_o) + b_o
    cost = T.abs_(T.mean(T.sqr(d_mat - y_vec)))
    accuracy = T.mean(d_mat - y_vec)

    # define gradients
    dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2 = T.grad(
        cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2])

    # compile train and test functions
    train = theano.function(
        inputs=[x_mat, d_mat],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h1],
                 [b_h1, b_h1 - alpha * db_h1],
                 [w_h2, w_h2 - alpha * dw_h2],
                 [b_h2, b_h2 - alpha * db_h2]],
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
    best_w_o = np.zeros(20)
    best_w_h1 = np.zeros([no_features, no_hidden])
    best_w_h2 = np.zeros([no_hidden, 20])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden)
    best_b_h2 = np.zeros(20)

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
            best_w_h2 = w_h2.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()
            best_b_h2 = b_h2.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)
    w_h2.set_value(best_w_h2)
    b_h2.set_value(best_b_h2)

    _, best_cost, best_accuracy = test(test_x, np.transpose(test_y))

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' %
          (best_cost, best_accuracy, best_iter))

    return train_cost, test_cost, test_accuracy


def nn_5_layer(train_x, train_y, test_x, test_y, no_hidden=30, learning_rate=1e-4):
    """Train and test the 5-layer neural network."""
    no_features = train_x.shape[1]
    x_mat = T.matrix('x')  # data sample
    d_mat = T.matrix('d')  # desired output

    # initialize weights and biases for hidden layer(s) and output layer
    w_o = theano.shared(np.random.randn(20) * .01, FL_X)
    b_o = theano.shared(np.random.randn() * .01, FL_X)
    w_h1 = theano.shared(np.random.randn(
        no_features, no_hidden) * .01, FL_X)
    b_h1 = theano.shared(np.random.randn(no_hidden) * 0.01, FL_X)
    w_h2 = theano.shared(np.random.randn(
        no_hidden, 20) * .01, FL_X)
    b_h2 = theano.shared(np.random.randn(20) * 0.01, FL_X)
    w_h3 = theano.shared(np.random.randn(
        20, 20) * .01, FL_X)
    b_h3 = theano.shared(np.random.randn(20) * 0.01, FL_X)

    # learning rate
    alpha = theano.shared(learning_rate, FL_X)

    # define mathematical expressions
    h1_out = T.nnet.sigmoid(T.dot(x_mat, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    h3_out = T.nnet.sigmoid(T.dot(h2_out, w_h3) + b_h3)
    y_vec = T.dot(h3_out, w_o) + b_o
    cost = T.abs_(T.mean(T.sqr(d_mat - y_vec)))
    accuracy = T.mean(d_mat - y_vec)

    # define gradients
    dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2, dw_h3, db_h3 = \
        T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2, w_h3, b_h3])

    # compile train and test functions
    train = theano.function(
        inputs=[x_mat, d_mat],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h1],
                 [b_h1, b_h1 - alpha * db_h1],
                 [w_h2, w_h2 - alpha * dw_h2],
                 [b_h2, b_h2 - alpha * db_h2],
                 [w_h3, w_h3 - alpha * dw_h3],
                 [b_h3, b_h3 - alpha * db_h3]],
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
    best_w_o = np.zeros(20)
    best_w_h1 = np.zeros([no_features, no_hidden])
    best_w_h2 = np.zeros([no_hidden, 20])
    best_w_h3 = np.zeros([20, 20])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden)
    best_b_h2 = np.zeros(20)
    best_b_h3 = np.zeros(20)

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
            best_w_h2 = w_h2.get_value()
            best_w_h3 = w_h3.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()
            best_b_h2 = b_h2.get_value()
            best_b_h3 = b_h3.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)
    w_h2.set_value(best_w_h2)
    b_h2.set_value(best_b_h2)
    w_h3.set_value(best_w_h3)
    b_h3.set_value(best_b_h3)

    _, best_cost, best_accuracy = test(test_x, np.transpose(test_y))

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' %
          (best_cost, best_accuracy, best_iter))

    return train_cost, test_cost, test_accuracy
