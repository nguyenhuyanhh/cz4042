"""Project 2a: Deep CNN."""

from __future__ import division, print_function

import os

import numpy as np
import pylab
from tqdm import tqdm

from nn_utils import load_mnist, shuffle_data
from nn_cnn import cnn, sgd, sgd_momentum, rms_prop

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
np.random.seed(10)
BATCH_SIZE = 128
NO_ITERS = 100


def tt_plot_func(train_x, train_y, test_x, test_y, func=sgd):
    """Train, test and plot using a particular update function.

    Arguments:
        train_x, train_y, test_x, test_y: train and test data
        func: update function to use, default to nn_cnn.sgd
    """
    # train and test
    train, predict, test = cnn(update_func=func)
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

    # output max accuracy at # iterations
    print('%.1f accuracy at %d iterations' %
          (np.max(test_accr) * 100, np.argmax(test_accr) + 1))

    # plot test accuracy
    pylab.figure()
    pylab.plot(range(NO_ITERS), test_accr)
    pylab.xlabel('epochs')
    pylab.ylabel('test accuracy')
    pylab.savefig(os.path.join(CUR_DIR, 'project_2a_test.png'))

    # plot training cost
    pylab.figure()
    pylab.plot(range(NO_ITERS), train_cost)
    pylab.xlabel('epochs')
    pylab.ylabel('training cost')
    pylab.savefig(os.path.join(CUR_DIR, 'project_2a_train.png'))

    # pick a random image
    ind = np.random.randint(low=0, high=2000)
    conv_1, pool_1, conv_2, pool_2 = test(test_x[ind:ind + 1, :])

    # show input image
    pylab.figure()
    pylab.gray()
    pylab.axis('off')
    pylab.imshow(test_x[ind, :].reshape(28, 28))
    pylab.title('input image')
    pylab.savefig(os.path.join(CUR_DIR, 'img_input.png'))

    # show convolved and pooled feature maps
    pylab.figure()
    pylab.gray()
    for i in range(15):
        pylab.subplot(3, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(conv_1[0, i, :].reshape(20, 20))
    pylab.suptitle('layer 1 convolved feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'img_conv_1.png'))

    pylab.figure()
    pylab.gray()
    for i in range(15):
        pylab.subplot(3, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(pool_1[0, i, :].reshape(10, 10))
    pylab.suptitle('layer 1 pooled feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'img_pooled_1.png'))

    pylab.figure()
    pylab.gray()
    for i in range(20):
        pylab.subplot(4, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(conv_2[0, i, :].reshape(6, 6))
    pylab.suptitle('layer 2 convolved feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'img_conv_2.png'))

    pylab.figure()
    pylab.gray()
    for i in range(20):
        pylab.subplot(4, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(pool_2[0, i, :].reshape(3, 3))
    pylab.suptitle('layer 2 pooled feature maps')
    pylab.savefig(os.path.join(CUR_DIR, 'img_pooled_2.png'))
    pylab.show()


def tt_plot_comp(train_x, train_y, test_x, test_y):
    """Train, test and plot a comparison of update functions.

    Arguments:
        train_x, train_y, test_x, test_y: train and test data
    """
    funcs = [sgd, sgd_momentum, rms_prop]
    cost = []
    accr = []
    for func in funcs:
        # train and test
        train, predict, _ = cnn(update_func=func)
        test_accr = []
        train_cost = []
        for _ in tqdm(range(NO_ITERS)):
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

        # output max accuracy at # iterations
        print('%.1f accuracy at %d iterations' %
              (np.max(test_accr) * 100, np.argmax(test_accr) + 1))

        cost += [train_cost]
        accr += [test_accr]

    # plot test accuracy
    pylab.figure()
    for label, series in zip([x.__name__ for x in funcs], accr):
        pylab.plot(range(NO_ITERS), series, label=label)
    pylab.xlabel('epochs')
    pylab.ylabel('test accuracy')
    pylab.legend()
    pylab.savefig(os.path.join(CUR_DIR, 'project_2a_test.png'))

    # plot training cost
    pylab.figure()
    for label, series in zip([x.__name__ for x in funcs], cost):
        pylab.plot(range(NO_ITERS), series, label=label)
    pylab.xlabel('epochs')
    pylab.ylabel('training cost')
    pylab.legend()
    pylab.savefig(os.path.join(CUR_DIR, 'project_2a_train.png'))
    pylab.show()


def train_test_plot(func=sgd, compare=False):
    """Entry point for script.

    Arguments:
        func: update function to use, default to nn_cnn.sgd
        compare: whether to compare accr and cost for update functions, default False
    """
    # load data
    train_x, test_x, train_y, test_y = load_mnist(onehot=True)
    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)
    train_x, train_y = train_x[:12000], train_y[:12000]
    test_x, test_y = test_x[:2000], test_y[:2000]
    print('finished loading data')

    # compare update functions
    if compare:
        tt_plot_comp(train_x, train_y, test_x, test_y)
    # just plot for one update function
    else:
        tt_plot_func(train_x, train_y, test_x, test_y, func=func)


if __name__ == '__main__':
    # Q1
    # train_test_plot()
    # Q2
    # train_test_plot(func=sgd_momentum)
    # Q3
    # train_test_plot(func=rms_prop)
    # Q4
    train_test_plot(compare=True)
