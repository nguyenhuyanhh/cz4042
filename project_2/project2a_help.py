from __future__ import print_function, division

import numpy as np
import pylab
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from tqdm import tqdm

from load import mnist

try:
    from itertools import izip as zip
except ImportError:  # py3 without itertools.izip
    pass

# 1 convolution layer, 1 max pooling layer and a softmax layer

np.random.seed(10)
BATCH_SIZE = 128
NO_ITERS = 25


def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values = np.asarray(
        np.random.uniform(low=-bound, high=bound, size=filter_shape),
        dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values, borrow=True), theano.shared(b_values, borrow=True)


def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values = np.asarray(
        np.random.uniform(low=-bound, high=bound, size=filter_shape),
        dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values, borrow=True), theano.shared(b_values, borrow=True)


def model(X, w1, b1, w2, b2):
    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    pool_dim = (4, 4)
    o1 = pool.pool_2d(y1, pool_dim)
    o2 = T.flatten(o1, outdim=2)

    pyx = T.nnet.softmax(T.dot(o2, w2) + b2)
    return y1, o1, pyx


def sgd(cost, params, learning_rate=0.05, decay=0.0001):
    """Stochastic Gradient Descent function."""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append([param, param - (grad + decay * param) * learning_rate])
    return updates


def sgd_momentum(cost, params, learning_rate=0.05, decay=0.0001, momentum=0.5):
    """Stochastic Gradient Descent with momentum."""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        vel = theano.shared(param.get_value() * 0.)
        vel_new = momentum * vel - (grad + decay * param) * learning_rate
        updates.append([param, param + vel_new])
        updates.append([vel, vel_new])
    return updates


def rms_prop(cost, params, learning_rate=0.001, decay=0.0001, rho=0.9, epsilon=1e-6):
    """RMSProp algorithm."""
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        grad_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / grad_scaling
        updates.append((acc, acc_new))
        updates.append((param, param - learning_rate * (grad + decay * param)))
    return updates


def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def main():
    train_x, test_x, train_y, test_y = mnist(onehot=True)
    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)
    train_x, train_y = train_x[:12000], train_y[:12000]
    test_x, test_y = test_x[:2000], test_y[:2000]
    print('finished loading data')

    x_tensor = T.tensor4('X')
    y_mat = T.matrix('Y')

    num_filters = 25
    weight_1, bias_1 = init_weights_bias4(
        (num_filters, 1, 9, 9), x_tensor.dtype)
    weight_2, bias_2 = init_weights_bias2(
        (num_filters * 5 * 5, 10), x_tensor.dtype)
    y_1, o_1, py_x = model(x_tensor, weight_1, bias_1, weight_2, bias_2)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, y_mat))
    params = [weight_1, bias_1, weight_2, bias_2]
    updates = sgd(cost, params, learning_rate=0.05)
    train = theano.function(
        inputs=[x_tensor, y_mat], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(
        inputs=[x_tensor], outputs=y_x, allow_input_downcast=True)
    test = theano.function(inputs=[x_tensor], outputs=[
        y_1, o_1], allow_input_downcast=True)

    test_accr = []
    train_cost = []

    for i in tqdm(range(NO_ITERS)):
        train_x, train_y = shuffle_data(train_x, train_y)
        test_x, test_y = shuffle_data(test_x, test_y)
        cost = 0.0
        train_length = len(train_x)

        for start, end in zip(range(0, train_length, BATCH_SIZE), range(BATCH_SIZE, train_length, BATCH_SIZE)):
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
    pylab.savefig('figure_2a_test.png')

    pylab.figure()
    pylab.plot(range(NO_ITERS), train_cost)
    pylab.xlabel('epochs')
    pylab.ylabel('training cost')
    pylab.savefig('figure_2a_train.png')

    w = weight_1.get_value()
    pylab.figure()
    pylab.gray()
    for i in range(25):
        pylab.subplot(5, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(w[i, :, :, :].reshape(9, 9))
    pylab.suptitle('filters learned')
    pylab.savefig('figure_2a_filters.png')

    ind = np.random.randint(low=0, high=2000)
    convolved, pooled = test(test_x[ind:ind + 1, :])

    pylab.figure()
    pylab.gray()
    pylab.axis('off')
    pylab.imshow(test_x[ind, :].reshape(28, 28))
    pylab.title('input image')
    pylab.savefig('figure_2a_input_img.png')

    pylab.figure()
    pylab.gray()
    for i in range(25):
        pylab.subplot(5, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(convolved[0, i, :].reshape(20, 20))
    pylab.suptitle('convolved feature maps')
    pylab.savefig('figure_2a_conv_features.png')

    pylab.figure()
    pylab.gray()
    for i in range(5):
        pylab.subplot(5, 5, i + 1)
        pylab.axis('off')
        pylab.imshow(pooled[0, i, :].reshape(5, 5))
    pylab.suptitle('pooled feature maps')
    pylab.savefig('figure_2a_pooled_features.png')
    pylab.show()


if __name__ == '__main__':
    main()
