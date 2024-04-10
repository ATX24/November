#Neural Network for predicting tik tak toe moves
from xml.dom import xmlbuilder
import numpy as np
import random
import math 

class tiktak(object): 
    def __init__(self):
        self.layers = [9, 5, 5, 9]
        self.weights_layer2 = np.random.randn(9, 5)
        self.weights_layer3 = np.random.randn(5, 5)
        self.weights_layer4 = np.random.randn(5, 9)

        self.biases_layer2 = np.random.randn(5)
        self.biases_layer3 = np.random.randn(5)
        self.biases_layer4 = np.random.randn(9)

        self.weights = [self.weights_layer2, self.weights_layer3, self.weights_layer4]
        self.biases = [self.biases_layer2, self.biases_layer3, self.biases_layer4]

        self.activations_layer1 = []
        self.activations_layer2 = []
        self.activations_layer3 = []
        self.activations_layer4 = []

        self.z = 0
        self.z2 = 0
        self.z3 = 0


    
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def feed_forward(self, X):
        X_train = np.array(X)
        print(X_train)
        print("CREATING INTIAL WEIGHTS AND BIASES")

        self.z = np.dot(X, self.weights[0]) + self.biases[0]
        self.activations_layer2 = self.sigmoid(self.z)

        self.z2 = np.dot(self.activations_layer2, self.weights[1]) + self.biases[1]
        self.activations_layer3 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.activations_layer3, self.weights[2]) + self.biases[2]
        self.activations_layer4 = self.sigmoid(self.z3)

        print(self.activations_layer4)

    

    def backward_propagation(self, X_train, y_train, epochs):
        print(f'X_TRAIN: {X_train}')
        print(f'Y_TRAIN: {y_train}')
        for i in range(epochs):
            for i, x in enumerate(X_train):
                print(f'X: {x}')
                print(f'y: {y_train[i]}')
                self.feed_forward(x)
                #Step one - compute  dc/da for the last layer thing
                #1 is the change for the last layer 
                #Remember, z3 is actually the zs for the last layer
                
                #Convert the activation input into a numpy array
                self.activations_layer1 = np.array(x)
                
                print('Calculate the bias delta for the last layer')
                #Calculate the bias delta for the last layer
                delta_b_4 = self.compute_cost_derivate(self.activations_layer4, y_train[i]) * self.sigmoid_prime(self.z3)


                print('Calculate the weight delta for 3 to 4')
                #Calculate the weight delta for 3 to 4
                test = self.activations_layer3.reshape(1, 5)
                delta_w_34 = np.dot(delta_b_4.reshape(9, 1), test)
                
                print('Calculate the bias delta for the second to last layer')
                #Calculate the bias delta for the second to last layer
                #delta_b_4 = sigmoid_prime(z3)(dc/da(l+!))
                delta_b_3 = np.dot(self.weights[-1], delta_b_4) * self.sigmoid_prime(self.z2)

                print('Calculate the weight delta for 2 to 3')
                #Calculate the weight delta for 2 to 3
                test = self.activations_layer2.reshape(1, 5)
                delta_w_23 = np.dot(delta_b_3.reshape(5, 1), test)
                
                print('Calculate the bias delta for the first layer')
                #Calculate the bias delta for the first layer
                delta_b_2 = np.dot(self.weights[-2], delta_b_3) * self.sigmoid_prime(self.z)

                print('Calculate the weight delta for 1 to 2')
                #Calculate the weight delta for 1 to 2
                test = self.activations_layer1.reshape(1, 9)
                delta_w_12 = np.dot(delta_b_2.reshape(5, 1), test)

                print('Apply all changes')
                #Apply all changes to weights and biases
                self.weights[0] = np.subtract(self.weights[0], delta_w_12.transpose())
                self.weights[1] = np.subtract(self.weights[1], delta_w_23.transpose())
                self.weights[2] = np.subtract(self.weights[2], delta_w_34.transpose())

                self.biases[0] = np.subtract(self.weights[0], delta_b_2) 
                self.biases[1] = np.subtract(self.weights[1], delta_b_3) 
                self.biases[2] = np.subtract(self.weights[2], delta_b_4) 

                self.weights[0].resize(9, 5)
                self.weights[1].resize(5, 5)
                self.weights[2].resize(5, 9)

                self.biases[0].resize(5)
                self.biases[1].resize(5)
                self.biases[2].resize(9)
                
                
            

                print(f'selected move: {X_train[i]}. Next move: {self.activations_layer4}')

            return 'test'


    def compute_cost(self, layer4, y_train):
        return sum(np.subtract(layer4, y_train) ** 2)
    
    def compute_cost_derivate(self, layer4, y_train):
        return 2*(layer4 - y_train)
    
    def rundat(self, X_train, y_train, epochs):
        backward_propagation = self.backward_propagation(X_train, y_train, epochs)
        final_weights = self.weights
        final_biases = self.biases

        while True:
            print('Model testing unit')
            move = input('Test Move: ')
            move = move

            if move == '0':
                self.feed_forward([1, 0, 0, 0, 0, 0, 0, 0, 0])
            if move == '1':
                self.feed_forward([0, 1, 0, 0, 0, 0, 0, 0, 0])
            if move == '2':
                self.feed_forward([0, 0, 1, 0, 0, 0, 0, 0, 0])
            if move == '3':
                self.feed_forward([0, 0, 0, 1, 0, 0, 0, 0, 0])
            if move == '4':
                self.feed_forward([0, 0, 0, 0, 1, 0, 0, 0, 0])
            if move == '5':
                self.feed_forward([0, 0, 0, 0, 0, 1, 0, 0, 0])
            if move == '6':
                self.feed_forward(list([0, 0, 0, 0, 0, 0, 1, 0, 0]))
            if move == '7':
                self.feed_forward(list([0, 0, 0, 0, 0, 0, 0, 1, 0]))
            if move == '8':
                self.feed_forward(list([0, 0, 0, 0, 0, 0, 0, 0, 1]))
            

            # max = 0
            # selected_value = 0
            # for i, value in enumerate(self.activations_layer4):
            #     if value > max:
            #         selected_value = i
            #         max = value
            
            # print(selected_value)
            

        
X_train = [[0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]
y_train = [[0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]]

# test = tiktak()
# # tiktak.populate(test)
# test.rundat(X_train, y_train, 4)
