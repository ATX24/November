#Neural Network November
#Based on the 3B1B video

import random
import numpy as np


class November(object):

    def __init__(self, sizes):
    
        self.num_layers = len(sizes)
        self.sizes = sizes

        #Initialize weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

    #Create stochastic gradient descent function
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
             n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data) #nice lil shuffle

            #Split the data into mini batches
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
                ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else: 
                print(f"Epoch {j} complete")
    
    def update_mini_batch(self, mini_batch, eta):
        #This is where we will update the networks weights and biases by applying gradient descent (using back prop) to each mini batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #Creating the biases and weights using the results from backpropagation
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #Officially updating the weights and biases for this mini batch
        self.weights  = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Feed forward
        activations = x
        activations_layers = [x] #Stores all activations (for each layer)
        zs = [] #Stores all the Z vectors (for each layer)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations) + b
            zs.append(z)
            new_activations = self.sigmoid(z)
            activations_layers.append(new_activations)
        
        #Backward pass
        delta =  self.cost_derivative(activations_layers[-1], y) * \
            self.sigmoid_prime(zs[-1])
        
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, np.transpose(activations_layers[-2]))

        # l = 1 means the last layer of neurons, l = 2 is the second-last layer (and so on)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations_layers[-l-1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

network = November([3, 3, 2])
training_data = list([(1, 1), (3, 0), (5, 3)])

November.SGD(network, training_data, 2, 1, 0.1, None)










