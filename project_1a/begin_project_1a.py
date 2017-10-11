"""Project 1a: Classification."""

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
    X = T.matrix()  # features
    Y = T.matrix()  # output

    # weights and biases from input to hidden layer
    w1, b1 = init_weights(36, hl_neuron), init_bias(hl_neuron)
    w2, b2 = init_weights(hl_neuron, 6, logistic=False), init_bias(
        6)  # weights and biases from hidden to output layer

    h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
    py = T.nnet.softmax(T.dot(h1, w2) + b2)

    y_x = T.argmax(py, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + \
        decay * (T.sum(T.sqr(w1) + T.sum(T.sqr(w2))))
    params = [w1, b1, w2, b2]
    updates = sgd(cost, params, learning_rate)

    # compile
    train = theano.function(
        inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(
        inputs=[X], outputs=y_x, allow_input_downcast=True)

    # train and test
    trainX, trainY, testX, testY = load_train_test()
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)

        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        train_cost = np.append(train_cost, cost / (n // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(
            np.argmax(testY, axis=1) == predict(testX)))

    print('%.1f accuracy at %d iterations' %
          (np.max(test_accuracy) * 100, np.argmax(test_accuracy) + 1))

    return (train_cost, test_accuracy)


if __name__ == '__main__':
    train_cost = []
    test_accuracy = []
    cost_args = []
    accuracy_args = []

    for batch_size in [4, 8, 16, 32, 64]:
        train_cost, test_accuracy = main(batch_size=batch_size)
        cost_args += [range(1000), train_cost]
        accuracy_args += [range(1000), test_accuracy]

    # Plots
    plt.figure()
    plt.plot(*cost_args)
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.savefig('p1a_sample_cost.png')

    plt.figure()
    plt.plot(*accuracy_args)
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.savefig('p1a_sample_accuracy.png')

    plt.show()
