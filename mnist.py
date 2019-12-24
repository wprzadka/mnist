import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np


class Network:

    def __init__(self, architecture, rate, rand_seed=34):
        self.architecture = architecture
        self.learn_rate = rate

        np.random.seed(rand_seed)

        self.weights = [np.random.randn(self.architecture[n + 1], self.architecture[n]) * 0.1
                        for n in range(len(self.architecture) - 1)]
        self.absolute_term = [np.random.randn(self.architecture[n], 1) * 0.1
                              for n in range(1, len(self.architecture))]
        # for a in range(len(self.absolute_term[0])):
        #     self.absolute_term[0][a] = 0

        # pprint(self.absolute_term)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid(x):
        return 1./(1 + np.exp(-x))

    def forward_propagation(self, input):
        output = np.array(input).T
        for w, b in zip(self.weights, self.absolute_term):
            prep = np.dot(w, output.T)
            layer = prep + b.T
            output = self.sigmoid(layer)

        return output

    @staticmethod
    def cross_entropy_loss(answer, guess):
        l_sum = np.sum(np.multiply(answer, np.log(guess)))
        m = guess.shape[0]

        return - l_sum / m

    def backward_propagation(self):
        pass


def draw(input):
    plt.figure(figsize=(28, 28))
    img = list()
    for y in range(28):
        img.append(input[y * 28:y * 28 + 28])

    plt.imshow(img)
    plt.savefig('sample.png')


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv', nrows=10)

    input = train_data.values.tolist()[9][1:]

    nn_arch = [28 * 28, 14 * 14, 7 * 7, 7 * 7, 10]

    network = Network([4, 8, 8, 2], 0.1)
    print(network.forward_propagation([20, 2, 3, 4]))

    # draw(input)
