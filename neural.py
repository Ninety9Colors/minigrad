from value import Value
import random

def train_test_split(data, percent_train, shuffle=True):
    if shuffle:
        random.seed()
        random.shuffle(data)
    n = len(data)

    return data[:int(n*percent_train)], data[int(n*percent_train):]

class Neuron:
    def __init__(self, input_size, seed=None, activation=None):
        if seed:
            random.seed(seed)
        self.w = [Value(random.uniform(-1 ,1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
    
    def __repr__(self, verbose=False):
        if verbose:
            return f"Neuron(input_size={len(self.w)}, activation={self.activation}, w={[v.data for v in self.w]}, b={self.b.data})"
        return f"Neuron(input_size={len(self.w)}, activation={self.activation})"
    
    def predict(self, x_input):
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

    def parameters(self):
        return self.w + [self.b]
    

class Layer:
    def __init__(self, input_size, output_size, seed=None, activation=None):
        if seed:
            random.seed(seed)
        self.neurons = [Neuron(input_size, activation=activation) for _ in range(output_size)]
    
    def __repr__(self, verbose=False):
        if verbose:
            result = f"Layer(input_size={len(self.neurons[0].w)}, output_size={len(self.neurons)},\n"
            for n in self.neurons:
                result += n.__repr__(True) + '\n'
            return result + ')'
        return f"Layer(input_size={len(self.neurons[0].w)}, output_size={len(self.neurons)}, activation={self.activation})"

    def predict(self, x_input):
        assert len(x_input) == len(self.neurons[0].w)

        return [n.predict(x_input) for n in self.neurons]

    def parameters(self):
        result = []
        
        for n in self.neurons:
            result.extend(n.parameters())

        return result

class MLP:
    def __init__(self, input_size, layer_sizes, cost = None, seed=None, inner_activation=None, final_activation=None):
        random.seed(seed)
        sizes = [input_size]+layer_sizes
        if cost:
            self.cost = cost
        else:
            self.cost = self.mse
        self.layers = [Layer(sizes[i-1], sizes[i], activation=inner_activation) for i in range(1, len(sizes)-1)]
        self.layers.append(Layer(sizes[-2], sizes[-1], activation=final_activation))
    
    def __repr__(self, verbose=False):
        if verbose:
            result = f"MLP(input_size={len(self.layers[0].neurons[0].w)}, layer_sizes={[len(self.layers[i].neurons) for i in range(0, len(self.layers))]},\n"
            for l in self.layers:
                result += l.__repr__(True) + '\n'
            return result + ')'
        return f"MLP(input_size={len(self.layers[0].neurons[0].w)}, layer_sizes={[len(self.layers[i].neurons) for i in range(0, len(self.layers))]})"
    
    def predict(self, x_input):
        assert len(x_input[0]) == len(self.layers[0].neurons[0].w)

        return [self.single_predict(example) for example in x_input]
    
    def single_predict(self, x_input):
        assert len(x_input) == len(self.layers[0].neurons[0].w)

        activation = self.layers[0].predict(x_input)

        for i in range(1, len(self.layers)):
            activation = self.layers[i].predict(activation)
        
        return activation
    
    def mse(self, x_input, y_target, alpha=1e-4):
        assert len(x_input[0]) == len(self.layers[0].neurons[0].w)
        assert len(y_target[0]) == len(self.layers[len(self.layers)-1].neurons)

        y_predict = [self.single_predict(i) for i in x_input]
        loss = sum((y_predict[i][j]-y_target[i][j])**2 for j in range(len(y_target[0])) for i in range(len(y_target)))

        reg_loss = alpha * sum((p*p for p in self.parameters()))
        
        return loss + reg_loss
    
    def parameters(self):
        result = []
        
        for l in self.layers:
            result.extend(l.parameters())

        return result