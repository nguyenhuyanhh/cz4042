"""Project 2a: Deep CNN."""

from __future__ import division, print_function

import os

import numpy as np
import pylab
import theano
from theano import tensor as T
from tqdm import tqdm

from nn_utils import (init_weights_bias_2d, init_weights_bias_4d, load_mnist,
                      model, rms_prop, sgd, sgd_momentum, shuffle_data)

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
np.random.seed(10)
BATCH_SIZE = 128
NO_ITERS = 100


def cnn_sgd():
    """CNN architecture with SGD."""
    x_tensor = T.tensor4('X')
    y_mat = T.matrix('Y')

    # conv layer C1, 15 9x9 window filters
    weight_1, bias_1 = init_weights_bias_4d(
        (15, 1, 9, 9), x_tensor.dtype)

    # conv layer C2, 20 5x5 window filters
    weight_2, bias_2 = init_weights_bias_4d(
        (20, 15, 5, 5), x_tensor.dtype)

    # fully connected layer F3, 100 neurons
    weight_3, bias_3 = init_weights_bias_2d(
        (20 * 3 * 3, 100), x_tensor.dtype)

    # softmax output layer, 10 neurons
    weight_4, bias_4 = init_weights_bias_2d(
        (100, 10), x_tensor.dtype)

    y_1, o_1, y_2, o_2, py_x = model(x_tensor,
                                     weight_1, bias_1,
                                     weight_2, bias_2,
                                     weight_3, bias_3,
                                     weight_4, bias_4)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, y_mat))
    params = [weight_1, bias_1, weight_2, bias_2,
              weight_3, bias_3, weight_4, bias_4]
    updates = sgd(cost, params, learning_rate=0.05)
    train = theano.function(
        inputs=[x_tensor, y_mat], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(
        inputs=[x_tensor], outputs=y_x, allow_input_downcast=True)
    test = theano.function(inputs=[x_tensor], outputs=[
        y_1, o_1, y_2, o_2], allow_input_downcast=True)

    return train, predict, test


def main():
    """Entry point for script."""
    train_x, test_x, train_y, test_y = load_mnist(onehot=True)
    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)
    train_x, train_y = train_x[:12000], train_y[:12000]
    test_x, test_y = test_x[:2000], test_y[:2000]
    print('finished loading data')

    train, predict, test = cnn_sgd()
    test_accr = []
    train_cost = []
    for i in tqdm(range(NO_ITERS)):
        train_x, train_y = shuffle_data(train_x, train_y)
        test_x, test_y = shuffle_data(test_x, test_y)
        cost = 0.0
        train_length = len(train_x)

        starts = range(0, train_length, BATCH_SIZE)
        ends = range(BATCH_SIZE, train_length, BATCH_SIZE)
        for start, end in zip(starts, ends):
            cost += train(train_x[start:end], train_y[start:end])

        # average out the cost for one epoch
        cost = cost / (train_length // BATCH_SIZE)
        train_cost += [cost]
        test_accr.append(
            np.mean(np.argmax(test_y, axis=1) == predict(test_x)))

    print('%.1f accuracy at %d iterations' %
          (np.max(test_accr) * 100, np.argmax(test_accr) + 1))

    pylab.figure()
    pylab.plot(range(NO_ITERS), test_accr)
    pylab.xlabel('epochs')
    pylab.ylabel('test accuracy')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_test.png'))

    pylab.figure()
    pylab.plot(range(NO_ITERS), train_cost)
    pylab.xlabel('epochs')
    pylab.ylabel('training cost')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_train.png'))

    # w_1 = weight_1.get_value()
    # pylab.figure()
    # pylab.gray()
    # for i in range(15):
    #     pylab.subplot(5, 5, i + 1)
    #     pylab.axis('off')
    #     pylab.imshow(w_1[i, :, :, :].reshape(9, 9))
    # pylab.suptitle('filters learned')
    # pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_filters.png'))

    ind = np.random.randint(low=0, high=2000)
    conv_1, pool_1, conv_2, pool_2 = test(test_x[ind:ind + 1, :])

    pylab.figure()
    pylab.gray()
    pylab.axis('off')
    pylab.imshow(test_x[ind, :].reshape(28, 28))
    pylab.title('input image')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_input_img.png'))

    pylab.figure()
    pylab.gray()
    for i in range(15):
        pylab.subplot(3, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(conv_1[0, i, :].reshape(20, 20))
    pylab.suptitle('layer 1 convolved feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_conv_1.png'))

    pylab.figure()
    pylab.gray()
    for i in range(15):
        pylab.subplot(3, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(pool_1[0, i, :].reshape(10, 10))
    pylab.suptitle('layer 1 pooled feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_pooled_1.png'))

    pylab.figure()
    pylab.gray()
    for i in range(20):
        pylab.subplot(4, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(conv_2[0, i, :].reshape(6, 6))
    pylab.suptitle('layer 2 convolved feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_conv_2.png'))

    pylab.figure()
    pylab.gray()
    for i in range(20):
        pylab.subplot(4, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(pool_2[0, i, :].reshape(3, 3))
    pylab.suptitle('layer 2 pooled feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'figure_2a_pooled_2.png'))
    pylab.show()


if __name__ == '__main__':
    main()
