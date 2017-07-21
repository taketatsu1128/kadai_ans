# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, beta = 1.0):
    return 1.0 / (1.0 + np.exp(beta * -1.0 * x))

def sigmoid_diff(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def identity_function(x):
    return x

class NeuralNetwork(object):
    def __init__(self, inLayer = 0, hiddenLayer = 0, outLayer = 0, batchsize = 1):
        if inLayer != 0 and hiddenLayer != 0 and outLayer != 0:
            self.createNetwork(inLayer, hiddenLayer, outLayer, batchsize)

    def createNetwork(self, inLayer, hiddenLayer, outLayer, batchsize):
        self._inLayer = inLayer
        self._hiddenLayer = hiddenLayer
        self._outLayer = outLayer
        self._batchsize = batchsize

        self._w2 = np.random.rand(inLayer, hiddenLayer)
        self._w3 = np.random.rand(hiddenLayer, outLayer)

        self._inValue = np.zeros(inLayer)
        self._hiddenValue = np.zeros(hiddenLayer)
        self._outValue = np.zeros(outLayer)
        self._b2 = np.random.rand(self._hiddenLayer)[:, np.newaxis]
        self._b3 = np.random.rand(self._outLayer)[:, np.newaxis]
        self._ones = np.ones(batchsize)[:, np.newaxis]

    def feedforward(self, inValue):
        self._inValue = inValue

        self._a2 = np.dot(self._inValue, self._w2) + np.dot(self._ones, self._b2.T)
        self._z2 = sigmoid(self._a2)
        self._a3 = np.dot(self._z2, self._w3) + np.dot(self._ones, self._b3.T)
        self._outValue = identity_function(self._a3)

        return self._outValue

    def loss(self, teachValue):
        diff = self._outValue - teachValue
        return (1.0 / len(teachValue)) * 0.5 * np.sum(np.power(diff, 2.0))

    def back_propagation(self, teachValue, learning_rate = 0.1):
        delta3 = self._outValue - teachValue
        gradw3 = (1.0 / len(teachValue)) * np.dot(self._z2.T, delta3)
        gradb3 = (1.0 / len(teachValue)) * np.dot(self._ones.T, delta3)
        self._w3 += -1.0 * learning_rate * gradw3
        self._b3 += -1.0 * learning_rate * gradb3

        delta2 = sigmoid_diff(self._a2) * np.dot(delta3, self._w3.T)
        gradw2 = (1.0 / len(teachValue)) * np.dot(self._inValue.T, delta2)
        gradb2 = (1.0 / len(teachValue)) * np.dot(delta2.T, self._ones)
        self._w2 += -1.0 * learning_rate * gradw2
        self._b2 += -1.0 * learning_rate * gradb2

if __name__ == '__main__':
    input = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output = np.array([0.0, 1.0, 1.0, 0.0])[:, np.newaxis]
    hidden_num = 5
    epoch = 10000
    end_loss = 0.0
    loss_list = []
    epoch_list = []

    nn = NeuralNetwork(len(input[0]), hidden_num, len(output[0]), len(input))

    for i in range(epoch):
        loss = 0.0
        o = nn.feedforward(input)
        loss += nn.loss(output)
        loss_list.append(loss)
        epoch_list.append(i + 1)
        if (i + 1) % 1000 == 0:
            print 'epoch:' + str(i + 1)
            print 'loss:' + str(loss)
        if loss < end_loss:
            print 'epoch:' + str(i + 1)
            print 'loss:' + str(loss)
            break
        nn.back_propagation(output)

    print o
    plt.plot(epoch_list, loss_list)
    plt.ylim([0, 0.2])
    plt.show()
