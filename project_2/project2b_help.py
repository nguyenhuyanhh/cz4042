from __future__ import division, print_function

import numpy as np
import pylab
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from nn_utils import load_mnist

# 1 encoder, decoder and a softmax layer


def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)


def init_bias(n):
    return theano.shared(value=np.zeros(n, dtype=theano.config.floatX), borrow=True)


def main():

    trX, teX, trY, teY = load_mnist(onehot=True)

    trX, trY = trX[:12000], trY[:12000]
    teX, teY = teX[:2000], teY[:2000]

    x = T.fmatrix('x')
    y = T.fmatrix('d')

    np.random.seed(10)
    
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    corruption_level = 0.1
    training_epochs = 25
    learning_rate = 0.1
    batch_size = 128

    W1 = init_weights(28 * 28, 900)
    b1 = init_bias(900)
    W2 = init_weights(900, 625)
    b2 = init_bias(625)
    W3 = init_weights(625, 400)
    b3 = init_bias(400)

    # output reconstructed input from 3rd hidden layer
    W4 = init_weights(400, 28 * 28)
    b4 = init_bias(28 * 28)

    # For softmax layer
    W5 = init_weights(400, 10)
    b5 = init_bias(10)

    # CORRUPT THE PURE
    tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                                  dtype=theano.config.floatX) * x
    
    y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
    y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
    y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)

    # reconstruction layer
    z1 = T.nnet.sigmoid(T.dot(y3, W4) + b4)
    
    cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))

    params1 = [W1, b1, W2, b2, W3, b3, W4, b4]
    grads1 = T.grad(cost1, params1)
    updates1 = [(param1, param1 - learning_rate * grad1)
                for param1, grad1 in zip(params1, grads1)]
    train_da1 = theano.function(
        inputs=[x], outputs=cost1, updates=updates1, allow_input_downcast=True)
    test_da1 = theano.function(
        inputs=[x], outputs=[y1, y2, y3, z1], allow_input_downcast=True)

    # softmax layer
    p_y4 = T.nnet.softmax(T.dot(y3, W5) + b5)
    y4 = T.argmax(p_y4, axis=1)
    cost2 = T.mean(T.nnet.categorical_crossentropy(p_y4, y))

    params2 = [W1, b1, W2, b2, W3, b3, W5, b5]
    grads2 = T.grad(cost2, params2)
    updates2 = [(param2, param2 - learning_rate * grad2)
                for param2, grad2 in zip(params2, grads2)]
    train_ffn = theano.function(
        inputs=[x, y], outputs=cost2, updates=updates2, allow_input_downcast=True)
    test_ffn = theano.function(
        inputs=[x], outputs=y4, allow_input_downcast=True)

    # stacked denoising autoencoder
##    print('training dae1 ...')
##    train_cost = []
##    for epoch in range(training_epochs):
##        # go through trainng set
##        cost = []
##        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
##            cost.append(train_da1(trX[start:end]))
##        train_cost.append(np.mean(cost, dtype='float64'))
##        print(train_cost[epoch])
##
##    pylab.figure()
##    pylab.plot(range(training_epochs), train_cost)
##    pylab.xlabel('iterations')
##    pylab.ylabel('cross-entropy')
##    pylab.savefig('figure_2b_train.png')
##
##    w1 = W1.get_value()
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(w1[:, i].reshape(28, 28))
##    pylab.suptitle('layer 1 weight samples')
##    pylab.savefig('figure_2b_weight1.png')
##
##    w2 = W2.get_value()
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(w2[:, i].reshape(30, 30))
##    pylab.suptitle('layer 2 weight samples')
##    pylab.savefig('figure_2b_weight2.png')
##
##    w3 = W3.get_value()
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(w3[:, i].reshape(25, 25))
##    pylab.suptitle('layer 3 weight samples')
##    pylab.savefig('figure_2b_weight3.png')
##
##    ind = np.random.randint(low=0, high=1900)
##    layer_1, layer_2, layer_3, output = test_da1(trX[ind:ind + 100, :])
##
##    # show input image
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(trX[ind + i:ind + i + 1, :].reshape(28, 28))
##    pylab.suptitle('input images')
##    pylab.savefig('figure_2b_input.png')
##
##    # hidden layer activations
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(layer_1[i, :].reshape(30, 30))
##    pylab.suptitle('layer 1 activations')
##    pylab.savefig('figure_2b_layer1.png')
##
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(layer_2[i, :].reshape(25, 25))
##    pylab.suptitle('layer 2 activations')
##    pylab.savefig('figure_2b_layer2.png')
##
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(layer_3[i, :].reshape(20, 20))
##    pylab.suptitle('layer 3 activations')
##    pylab.savefig('figure_2b_layer3.png')
##
##    # reconstructed outputs
##    pylab.figure()
##    pylab.gray()
##    for i in range(100):
##        pylab.subplot(10, 10, i + 1)
##        pylab.axis('off')
##        pylab.imshow(output[i, :].reshape(28, 28))
##    pylab.suptitle('reconstructed outputs')
##    pylab.savefig('figure_2b_output.png')

    # softmax
    print('\ntraining ffn ...')
    train_cost = []
    test_accr = []
    for epoch in range(training_epochs):
        # go through trainng set
        cost = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            cost.append(train_ffn(trX[start:end], trY[start:end]))
        train_cost.append(np.mean(cost, dtype='float64'))
        test_accr.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
        print(test_accr[epoch])

    pylab.figure()
    pylab.plot(range(training_epochs), train_cost)
    pylab.xlabel('iterations')
    pylab.ylabel('cross-entropy')
    pylab.savefig('figure_2b_train_ffn.png')

    pylab.figure()
    pylab.plot(range(training_epochs), test_accr)
    pylab.xlabel('iterations')
    pylab.ylabel('test accuracy')
    pylab.savefig('figure_2b_test_ffn.png')

    pylab.show()

if __name__ == '__main__':
    main()
