# coding: utf-8
import numpy as np

def sigmoid(x, beta = 1.0):
    return 1.0 / (1.0 + np.exp(beta * -1.0 * x))

def sigmoid_diff(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def identity_function(x):
    return x

class NeuralNetwork(object):
    def __init__(self, inLayer = 0, outLayer = 0, batchsize = 1):
        if inLayer != 0 and outLayer != 0:
            self.createNetwork(inLayer, outLayer, batchsize)

    def createNetwork(self, inLayer, outLayer, batchsize):
        self._inLayer = inLayer
        self._outLayer = outLayer
        self._batchsize = batchsize

        self._w2 = np.random.rand(inLayer, outLayer)

        self._inValue = np.zeros(inLayer)
        self._outValue = np.zeros(outLayer)
        self._b2 = np.random.rand(self._outLayer)[:, np.newaxis]
        self._ones = np.ones(batchsize)[:, np.newaxis]

    def feedforward(self, inValue):
        self._inValue = inValue

        self._a2 = np.dot(self._inValue, self._w2) + np.dot(self._ones, self._b2.T)
        self._outValue = sigmoid(self._a2)

        return self._outValue

    def loss(self, teachValue):
        diff = self._outValue - teachValue
        return (1.0 / len(teachValue)) * 0.5 * np.sum(np.power(diff, 2.0))

    def back_propagation(self, teachValue, learning_rate = 0.01):
        delta2 = self._outValue - teachValue
        gradw2 = (1.0 / len(teachValue)) * np.dot(self._inValue.T, delta2)
        gradb2 = (1.0 / len(teachValue)) * np.dot(self._ones.T, delta2)
        self._w2 += -1.0 * learning_rate * gradw2
        self._b2 += -1.0 * learning_rate * gradb2

if __name__ == '__main__':
    input = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output = np.array([0.0, 1.0, 1.0, 1.0])[:, np.newaxis]
    epoch = 1000000
    end_loss = 0.0

    nn = NeuralNetwork(len(input[0]), len(output[0]), len(input))

    for i in range(epoch):
        loss = 0.0
        o = nn.feedforward(input)
        loss += nn.loss(output)
        if (i + 1) % 100000 == 0:
            print 'epoch:' + str(i + 1)
            print 'loss:' + str(loss)
        if loss < end_loss:
            print 'epoch:' + str(i + 1)
            print 'loss:' + str(loss)
            break
        nn.back_propagation(output)

    print o
