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

    # read train data
    train_input = np.loadtxt('sat_train.txt', delimiter=' ')
    trainX, train_Y = train_input[:, :36], train_input[:, -1].astype(int)
    trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
    trainX = scale(trainX, trainX_min, trainX_max)

    train_Y[train_Y == 7] = 6
    trainY = np.zeros((train_Y.shape[0], 6))
    trainY[np.arange(train_Y.shape[0]), train_Y - 1] = 1

    # read test data
    test_input = np.loadtxt('sat_test.txt', delimiter=' ')
    testX, test_Y = test_input[:, :36], test_input[:, -1].astype(int)

    testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
    testX = scale(testX, testX_min, testX_max)

    test_Y[test_Y == 7] = 6
    testY = np.zeros((test_Y.shape[0], 6))
    testY[np.arange(test_Y.shape[0]), test_Y - 1] = 1

    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)

    # first, experiment with a small sample of data
    ##trainX = trainX[:1000]
    ##trainY = trainY[:1000]
    ##testX = testX[-250:]
    ##testY = testY[-250:]

    # train and test
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

    search_space = [4,8,16,32,64]
    
    for batch_size in search_space:
        train_cost, test_accuracy = main(batch_size=batch_size)
        cost_args += [train_cost]
        accuracy_args += [test_accuracy]
    
    # Plots
    plt.figure()
    for item, value in zip(cost_args,search_space):
        plt.plot(range(1000),item,label="batch={}".format(value))
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')
    plt.legend()
    plt.savefig('p1a_sample_cost.png')

    plt.figure()
    for item, value in zip(accuracy_args,search_space):
        plt.plot(range(1000),item,label="batch={}".format(value))
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.legend()
    plt.savefig('p1a_sample_accuracy.png')

    # forced garbage collection test
    train_cost = []
    test_accuracy = []
    cost_args = []
    accuracy_args = []

    plt.show()
