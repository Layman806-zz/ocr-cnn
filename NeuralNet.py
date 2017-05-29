import numpy as np
import cv2
import random
import json
debug = True
ip_size = 784  # 28*28


def vectorize(x):
    c = np.zeros((62, 1))
    c[x] = 1.0
    return c


class Loader:
    @staticmethod
    def dataload(folder='TrainingData'):  # takes folder of Training data as input
        trainingInput = []
        trainingResult = []

        validationInput = []
        validationResult = []

        testingInput = []
        testingResult = []
        path = folder + '/'
        for c in range(62):
            if debug is True:
                print('LOADING FOLDER: ' + str(c+1))
            for i in range(10):
                f_path = path + str(c+1) + '/' + str(i) + '.png'  # full path
                im = cv2.imread(f_path, 0)  # opening image in gray scale mode
                im = cv2.resize(im, (28, 28))
                # im2 = np.reshape(im, (ip_size, 1))  # reshaping to an array which can be fed to the neural network
                im2 = np.asarray(im)
                # im2 = im2.ravel()
                # cells.append(np.reshape((np.array(pdata), (ip_size, 1)), int(number)))
                im2 = np.reshape(im2, (ip_size, 1))
                print(im2.shape)
                # im2 = np.reshape(im, (ip_size, 1))
                result = vectorize(c)
                if debug is True:
                    print('loaded ' + f_path)
                if i < 5:  # 816:  # leaving 200 samples from each category for validation and testing
                    trainingInput.append(im2)
                    trainingResult.append(result)
                elif i < 8:  # 916:  # 100 from each category for validation and the rest for testing
                    validationInput.append(im2)
                    validationResult.append(result)
                else:
                    testingInput.append(im2)
                    testingResult.append(result)
        training_data = zip(trainingInput, trainingResult)
        validation_data = zip(validationInput, validationResult)
        testing_data = zip(testingInput, testingResult)
        return training_data, validation_data, testing_data


'''
training_data, validation_data, testing_data = Loader.load()
ip, op = zip(*training_data)  # unzipping training data
print(ip[42516])
for x in range(62):
    print(str(x)+str(op[42516][x]))
'''

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class Network:

    def __init__(self, sizes):  # sizes is the list of number of neurons in each layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x, y) in list(zip(sizes[:-1], sizes[1:]))]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            # x = np.dot(w, a) + b
            # a = np.maximum(x, 0, x)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):  # Stochaistic Gradient Descent
        # early stopping functionality:
        best_accuracy = 1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            '''for i in range(20):
                random.shuffle(training_data)'''
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(list(training_data), lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(list(training_data), convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(list(evaluation_data), lmbda, convert=False)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(list(evaluation_data), convert=True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    # print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    # print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):  # updating in mini batches
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):  # the Backpropagation...
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # delta = (self.cost).delta(zs[-1], activations[-1], y)
        delta = activations[-1]-y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, convert=True):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        s = int(0)
        '''for (x, y) in results:
            if np.int(x) == np.int(y):
                s = s + np.int(x)
        '''
        ''' for x, y in results:
            if (x == y):
                s = s + 1
            # print('x='+str(x)+',  \ny= '+str(y))
        '''
        return np.sum(x == y for (x, y) in results)
        # return s

    def cost_fn(self, a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorize(y)
            cost += self.cost_fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * np.sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": "CrossEntropyCost"}  # str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    # cost = getattr(sys.modules[__name__], data["cost"])
    '''cost is CrossEntropyCost by default, so no need to be read'''
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

t_data, validation_data, testing_data = Loader.dataload()
net = Network([ip_size, 64, 62])
# net = Network([ip_size, 64, 64, 64, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 64, 64, 64, 64, 64, 64, 62])
'''li = [ip_size]
for i in range(59):
    li.append(16)
li.append(62)
net = Network(li)'''
net.SGD(training_data=list(t_data), epochs=500, mini_batch_size=10000, eta=0.01, lmbda=2.0,
        evaluation_data=validation_data, monitor_evaluation_cost=True, monitor_training_accuracy=True,
        monitor_evaluation_accuracy=True, monitor_training_cost=True)
net.save('network_values.dat')

