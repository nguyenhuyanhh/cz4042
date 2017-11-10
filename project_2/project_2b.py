"""Project 2b: Autoencoders."""

from __future__ import division, print_function

import os

import numpy as np
import pylab
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from tqdm import tqdm

from nn_utils import load_mnist

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def init_weights(n_visible, n_hidden):
    """Initialize weights."""
    initial_weight = np.asarray(np.random.uniform(
        low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
        high=4 * np.sqrt(6. / (n_hidden + n_visible)),
        size=(n_visible, n_hidden)), dtype=theano.config.floatX)
    return theano.shared(value=initial_weight, name='W', borrow=True)


def init_bias(n_bias):
    """Initialize bias."""
    return theano.shared(value=np.zeros(n_bias, dtype=theano.config.floatX), borrow=True)


def main():
    """Entry point for script."""
    train_x, test_x, train_y, test_y = load_mnist(onehot=True)
    train_x, train_y = train_x[:12000], train_y[:12000]
    test_x, test_y = test_x[:2000], test_y[:2000]

    x_mat = T.fmatrix('x')
    y_mat = T.fmatrix('d')

    np.random.seed(10)

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    corruption_level = 0.1
    training_epochs = 25
    learning_rate = 0.1
    batch_size = 128

    weight_1 = init_weights(28 * 28, 900)
    bias_1 = init_bias(900)
    weight_2 = init_weights(900, 625)
    bias_2 = init_bias(625)
    weight_3 = init_weights(625, 400)
    bias_3 = init_bias(400)

    # output reconstructed input from 3rd hidden layer
    weight_4 = init_weights(400, 28 * 28)
    bias_4 = init_bias(28 * 28)

    # For softmax layer
    weight_5 = init_weights(400, 10)
    bias_5 = init_bias(10)

    # CORRUPT THE PURE
    tilde_x = theano_rng.binomial(size=x_mat.shape, n=1, p=1 - corruption_level,
                                  dtype=theano.config.floatX) * x_mat

    y_1 = T.nnet.sigmoid(T.dot(tilde_x, weight_1) + bias_1)
    y_2 = T.nnet.sigmoid(T.dot(y_1, weight_2) + bias_2)
    y_3 = T.nnet.sigmoid(T.dot(y_2, weight_3) + bias_3)

    # reconstruction layer
    z_1 = T.nnet.sigmoid(T.dot(y_3, weight_4) + bias_4)
    cost_1 = - T.mean(T.sum(x_mat * T.log(z_1) +
                            (1 - x_mat) * T.log(1 - z_1), axis=1))
    params_1 = [weight_1, bias_1, weight_2,
                bias_2, weight_3, bias_3, weight_4, bias_4]
    grads_1 = T.grad(cost_1, params_1)
    updates_1 = [(param1, param1 - learning_rate * grad1)
                 for param1, grad1 in zip(params_1, grads_1)]
    train_da1 = theano.function(
        inputs=[x_mat], outputs=cost_1, updates=updates_1, allow_input_downcast=True)
    test_da1 = theano.function(
        inputs=[x_mat], outputs=[y_1, y_2, y_3, z_1], allow_input_downcast=True)

    # softmax layer
    p_y_4 = T.nnet.softmax(T.dot(y_3, weight_5) + bias_5)
    y_4 = T.argmax(p_y_4, axis=1)
    cost_2 = T.mean(T.nnet.categorical_crossentropy(p_y_4, y_mat))
    params_2 = [weight_1, bias_1, weight_2,
                bias_2, weight_3, bias_3, weight_5, bias_5]
    grads_2 = T.grad(cost_2, params_2)
    updates_2 = [(param2, param2 - learning_rate * grad2)
                 for param2, grad2 in zip(params_2, grads_2)]
    train_ffn = theano.function(
        inputs=[x_mat, y_mat], outputs=cost_2, updates=updates_2, allow_input_downcast=True)
    test_ffn = theano.function(
        inputs=[x_mat], outputs=y_4, allow_input_downcast=True)

    # stacked denoising autoencoder
##    print('training dae1 ...')
##    train_cost = []
# for epoch in range(training_epochs):
# go through trainng set
##        cost = []
# for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
# cost.append(train_da1(trX[start:end]))
##        train_cost.append(np.mean(cost, dtype='float64'))
# print(train_cost[epoch])
##
# pylab.figure()
##    pylab.plot(range(training_epochs), train_cost)
# pylab.xlabel('iterations')
# pylab.ylabel('cross-entropy')
# pylab.savefig('figure_2b_train.png')
##
##    w1 = W1.get_value()
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(w1[:, i].reshape(28, 28))
##    pylab.suptitle('layer 1 weight samples')
# pylab.savefig('figure_2b_weight1.png')
##
##    w2 = W2.get_value()
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(w2[:, i].reshape(30, 30))
##    pylab.suptitle('layer 2 weight samples')
# pylab.savefig('figure_2b_weight2.png')
##
##    w3 = W3.get_value()
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(w3[:, i].reshape(25, 25))
##    pylab.suptitle('layer 3 weight samples')
# pylab.savefig('figure_2b_weight3.png')
##
##    ind = np.random.randint(low=0, high=1900)
##    layer_1, layer_2, layer_3, output = test_da1(trX[ind:ind + 100, :])
##
# show input image
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(trX[ind + i:ind + i + 1, :].reshape(28, 28))
##    pylab.suptitle('input images')
# pylab.savefig('figure_2b_input.png')
##
# hidden layer activations
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(layer_1[i, :].reshape(30, 30))
##    pylab.suptitle('layer 1 activations')
# pylab.savefig('figure_2b_layer1.png')
##
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(layer_2[i, :].reshape(25, 25))
##    pylab.suptitle('layer 2 activations')
# pylab.savefig('figure_2b_layer2.png')
##
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(layer_3[i, :].reshape(20, 20))
##    pylab.suptitle('layer 3 activations')
# pylab.savefig('figure_2b_layer3.png')
##
# reconstructed outputs
# pylab.figure()
# pylab.gray()
# for i in range(100):
##        pylab.subplot(10, 10, i + 1)
# pylab.axis('off')
##        pylab.imshow(output[i, :].reshape(28, 28))
##    pylab.suptitle('reconstructed outputs')
# pylab.savefig('figure_2b_output.png')

    # softmax
    print('training ffn ...')
    train_cost = []
    test_accr = []
    trx = len(train_x)
    for _ in tqdm(range(training_epochs)):
        # go through trainng set
        cost = []
        for start, end in zip(range(0, trx, batch_size), range(batch_size, trx, batch_size)):
            cost.append(train_ffn(train_x[start:end], train_y[start:end]))
        train_cost.append(np.mean(cost, dtype='float64'))
        test_accr.append(
            np.mean(np.argmax(test_y, axis=1) == test_ffn(test_x)))
    # output max accuracy at # iterations
    print('%.1f accuracy at %d iterations' %
          (np.max(test_accr) * 100, np.argmax(test_accr) + 1))

    pylab.figure()
    pylab.plot(range(training_epochs), train_cost)
    pylab.xlabel('iterations')
    pylab.ylabel('cross-entropy')
    pylab.savefig(os.path.join(CUR_DIR, 'project_2b_train_ffn.png'))

    pylab.figure()
    pylab.plot(range(training_epochs), test_accr)
    pylab.xlabel('iterations')
    pylab.ylabel('test accuracy')
    pylab.savefig(os.path.join(CUR_DIR, 'project_2b_test_ffn.png'))

    pylab.show()


if __name__ == '__main__':
    main()
