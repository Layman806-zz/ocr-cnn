import numpy as np
import cv2
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
# from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)

ip_size = 784
debug = True
GPU = True
if GPU:
    print("Trying to run under a GPU.")
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.")


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
            for i in range(1016):
                f_path = path + str(c+1) + '/' + str(i) + '.png'  # full path
                im = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)  # opening image in gray scale mode
                im = cv2.resize(im, (28, 28))
                # im2 = np.reshape(im, (ip_size, 1))  # reshaping to an array which can be fed to the neural network
                im2 = np.asarray(im)
                # im2 = im2.ravel()
                # im2 = np.reshape(im2, (ip_size, 1))
                # im2 = np.concatenate(im2)
                print(im2.shape)
                # im2 = np.reshape(im, (ip_size, 1))
                result = vectorize(c)
                result = np.concatenate(result)
                print(result.shape)
                if debug is True:
                    print('loaded ' + f_path)
                if i < 816:  # 816:  # leaving 200 samples from each category for validation and testing
                    trainingInput.append(im2)
                    trainingResult.append(result)
                elif i < 916:  # 916:  # 100 from each category for validation and the rest for testing
                    validationInput.append(im2)
                    validationResult.append(result)
                else:
                    testingInput.append(im2)
                    testingResult.append(result)
        training_data = (theano.shared(np.asarray(trainingInput, dtype=theano.config.floatX), borrow=True),
                         T.cast(theano.shared(np.asarray(trainingResult, dtype=theano.config.floatX), borrow=True),
                         "int32"))
        validation_data = (theano.shared(np.asarray(validationInput, dtype=theano.config.floatX), borrow=True),
                           T.cast(theano.shared(np.asarray(validationResult, dtype=theano.config.floatX),
                                  borrow=True), "int32"))
        testing_data = (theano.shared(np.asarray(testingInput, dtype=theano.config.floatX), borrow=True),
                        T.cast(theano.shared(np.asarray(testingResult, dtype=theano.config.floatX), borrow=True),
                        "int32"))
        return training_data, validation_data, testing_data


def load_shared_data():
    training_data, validation_data, test_data = Loader.dataload()

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return shared(training_data), shared(validation_data), shared(test_data)


class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):  # xrange() was renamed to range() in Python 3.
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, validation_data, test_data, epochs=60, mini_batch_size=10, eta=0.1, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data


        # compute number of minibatches for training, validation and testing
        num_training_batches = 816*62  # int(size(training_data) / mini_batch_size)
        num_validation_batches = 100*62  # int(size(validation_data) / mini_batch_size)
        num_test_batches = 100*62  # int(size(test_data) / mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
               0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        index = T.lscalar()  # mini-batch index
        train_mb = theano.function(
            inputs=[index], outputs=cost, updates=updates,
            givens={
                self.x:
                    training_x[index * self.mini_batch_size: (index + 1) * self.mini_batch_size],
                    # training_x,
                self.y:
                    training_y[index * self.mini_batch_size: (index + 1) * self.mini_batch_size]
                    # training_y
            })
        validate_mb_accuracy = theano.function(
            [index], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    validation_x[index * self.mini_batch_size: (index + 1) * self.mini_batch_size],
                self.y:
                    validation_y[index * self.mini_batch_size: (index + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [index], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[index * self.mini_batch_size: (index + 1) * self.mini_batch_size],
                self.y:
                    test_y[index * self.mini_batch_size: (index+ 1) * self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [index], self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[index * self.mini_batch_size: (index + 1) * self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):

            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))

        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class ConvPoolLayer(object):
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        # pooled_out = pool.pool_2d(
        #    input=conv_out, ds=self.poolsize, ignore_border=True, mode='max')
        pooled_out = pool.pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True, mode='max')
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output  # no dropout in the convolutional layers


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)


mini_batch_size = int(10)
# training_x, training_y, validation_x, validation_y, test_x, test_y = Loader.dataload()
training_data, validation_data, test_data = Loader.dataload()  # load_shared_data()
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=62)], mini_batch_size)
net.SGD(training_data, validation_data, test_data, epochs=60, mini_batch_size=mini_batch_size, eta=0.1)

