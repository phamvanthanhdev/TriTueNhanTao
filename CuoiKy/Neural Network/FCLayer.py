from layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        '''
        ex:
            input_shape : [1:3]
            ouput_shape : [1:4]
        => weights : [3:4]
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5
    
    def forward_propagation(self, input):  #hàm lan truyền tiến
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward_propagation(self, output_erorr, learning_rate):  #hàm lan truyền ngược
        current_layer_err = np.dot(output_erorr, self.weights.T)
        dweight = np.dot(self.input.T, output_erorr)

        self.weights -= dweight * learning_rate
        self.bias -= learning_rate * output_erorr

        return current_layer_err