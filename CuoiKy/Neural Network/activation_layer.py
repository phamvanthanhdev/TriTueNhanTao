from layer import Layer

class ActivationLayer(Layer):
    def __init__(self, input_shape, output_shape, activation, activation_layer):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.activation_layer = activation_layer
    
    def forward_propagation(self, input):  #hàm lan truyền tiến
        self.input = input
        self.output = self.activation(input)
        return self.output
    
    def backward_propagation(self, output_erorr, learning_rate):  #hàm lan truyền ngược
        return self.activation_layer(self.input) * output_erorr
    