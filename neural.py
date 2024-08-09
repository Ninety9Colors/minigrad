from value import Value
import random

# Utility function, splits data into two lists
def train_test_split(data, percent_train, shuffle=True):
    if shuffle:
        random.seed()
        random.shuffle(data)
    n = len(data)

    return data[:int(n*percent_train)], data[int(n*percent_train):]

# Holds weight vector and single bias, with activation function and randomized initial weights
class Neuron:
    def __init__(self, input_size, seed=None, activation=None):
        if seed:
            random.seed(seed)
        self.w = [Value(random.uniform(-1 ,1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    # Feeds forward input vector and returns scalar Value result
    def __call__(self, x_input: list[Value]) -> Value:
        assert len(x_input) == len(self.w)

        raw = sum((x_input[i]*self.w[i] for i in range(len(self.w))), self.b)

        if self.activation == 'linear' or self.activation == None:
            return raw
        elif self.activation == 'relu':
            return raw.relu()
        elif self.activation == 'leaky_relu':
            return raw.leaky_relu()
        elif self.activation == 'sigmoid':
            return raw.sigmoid()

    # Returns concatenated list of all weights (w+b)
    def parameters(self) -> list[Value]:
        return self.w + [self.b]
    
    def __repr__(self, verbose=False):
        if verbose:
            return f"Neuron(input_size={len(self.w)}, activation={self.activation}, w={[v.data for v in self.w]}, b={self.b.data})"
        return f"Neuron(input_size={len(self.w)}, activation={self.activation})"
    
# List of {output_size} neurons of size {input_size}
class Layer:
    def __init__(self, input_size, output_size, seed=None, activation=None):
        if seed:
            random.seed(seed)
        self.neurons = [Neuron(input_size, activation=activation) for _ in range(output_size)]

    # Returns activation vector for a vector input x_input
    def __call__(self, x_input: list[Value]) -> list[Value]:
        assert len(x_input) == len(self.neurons[0].w)

        return [n(x_input) for n in self.neurons]

    # Returns concatenated list of weights of each neuron
    def parameters(self) -> list[Value]:
        result = []
        for n in self.neurons: result.extend(n.parameters())
        return result

    def __repr__(self, verbose=False):
        if verbose:
            result = f"Layer(input_size={len(self.neurons[0].w)}, output_size={len(self.neurons)},\n"
            for n in self.neurons:
                result += n.__repr__(True) + '\n'
            return result + ')'
        return f"Layer(input_size={len(self.neurons[0].w)}, output_size={len(self.neurons)}, activation={self.activation})"

# List of layers of multiple sizes and activations, uses MSE cost by default, but supports user defined cost functions
class MLP:
    def __init__(self, input_size, layer_sizes, cost = None, seed=None, inner_activation=None, final_activation=None):
        random.seed(seed)
        layer_sizes = [input_size]+layer_sizes

        # Appends each internal and input layer
        self.layers = [Layer(layer_sizes[i-1], layer_sizes[i], activation=inner_activation) for i in range(1, len(layer_sizes)-1)]

        # Appends output layer with final activation
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], activation=final_activation))

        if cost: self.cost = cost
        else: self.cost = self.mse
    
    # Returns matrix result of forward propagation for multiple input vectors in x_input
    def __call__(self, x_input: list[list[Value]]) -> list[list[Value]]:
        assert len(x_input[0]) == len(self.layers[0].neurons[0].w)

        # Returns vector result of forward propagation for a single input vector x_input
        def single_predict(x_input: list[Value]) -> list[Value]:
            assert len(x_input) == len(self.layers[0].neurons[0].w)
            # Start by calculating input layer activation
            activation = self.layers[0](x_input)

            # Feed forward activation through layers
            for i in range(1, len(self.layers)):
                activation = self.layers[i](activation)
            return activation

        return [single_predict(example) for example in x_input]
    
    # Default cost function for MLP, returns squared difference between each predicted and target output value
    # By default uses L2 regularization, can be turned off by setting lambda_ = 0
    def mse(self, x_input: list[list[Value]], y_target: list[list[Value]], lambda_=1e-4):
        assert len(x_input[0]) == len(self.layers[0].neurons[0].w)
        assert len(y_target[0]) == len(self.layers[len(self.layers)-1].neurons)

        # MSE Loss
        y_predict = self(x_input)
        loss = sum((y_predict[i][j]-y_target[i][j])**2 for j in range(len(y_target[0])) for i in range(len(y_target)))

        # L2 Loss
        reg_loss = lambda_ * sum((p*p for p in self.parameters()))

        # Return total loss
        return loss + reg_loss
    
    # Returns concatenated list of weights of each layer and its neurons
    def parameters(self) -> list[Value]:
        result = []
        for l in self.layers:result.extend(l.parameters())
        return result
    
    def __repr__(self, verbose=False):
        if verbose:
            result = f"MLP(input_size={len(self.layers[0].neurons[0].w)}, layer_sizes={[len(self.layers[i].neurons) for i in range(0, len(self.layers))]},\n"
            for l in self.layers:
                result += l.__repr__(True) + '\n'
            return result + ')'
        return f"MLP(input_size={len(self.layers[0].neurons[0].w)}, layer_sizes={[len(self.layers[i].neurons) for i in range(0, len(self.layers))]})"