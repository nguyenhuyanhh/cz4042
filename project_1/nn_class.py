"""Neural networks for classification."""

from __future__ import print_function

import theano
import theano.tensor as T

from nn_utils import init_bias, init_weights, sgd


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
