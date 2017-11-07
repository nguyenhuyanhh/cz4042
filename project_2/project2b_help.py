import numpy as np
import pylab
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from load import mnist

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


trX, teX, trY, teY = mnist()

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

x = T.fmatrix('x')
d = T.fmatrix('d')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

corruption_level = 0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128


W1 = init_weights(28 * 28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28 * 28)
W1_prime = W1.transpose()
W2 = init_weights(900, 10)
b2 = init_bias(10)

tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX) * x
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))

params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = [(param1, param1 - learning_rate * grad1)
            for param1, grad1 in zip(params1, grads1)]
train_da1 = theano.function(
    inputs=[x], outputs=cost1, updates=updates1, allow_input_downcast=True)

p_y2 = T.nnet.softmax(T.dot(y1, W2) + b2)
y2 = T.argmax(p_y2, axis=1)
cost2 = T.mean(T.nnet.categorical_crossentropy(p_y2, d))

params2 = [W1, b1, W2, b2]
grads2 = T.grad(cost2, params2)
updates2 = [(param2, param2 - learning_rate * grad2)
            for param2, grad2 in zip(params2, grads2)]
train_ffn = theano.function(
    inputs=[x, d], outputs=cost2, updates=updates2, allow_input_downcast=True)
test_ffn = theano.function(inputs=[x], outputs=y2, allow_input_downcast=True)


print('training dae1 ...')
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da1(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_1.png')

w1 = W1.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(
        10, 10, i + 1); pylab.axis('off'); pylab.imshow(w1[:, i].reshape(28, 28))
pylab.savefig('figure_2b_2.png')


print('\ntraining ffn ...')
d, a = [], []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    d.append(np.mean(c, dtype='float64'))
    a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
    print(a[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_3.png')

pylab.figure()
pylab.plot(range(training_epochs), a)
pylab.xlabel('iterations')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2b_4.png')
pylab.show()

w2 = W2.get_value()
pylab.figure()
pylab.gray()
pylab.axis('off')
pylab.imshow(w2)
pylab.savefig('figure_2b_5.png')
