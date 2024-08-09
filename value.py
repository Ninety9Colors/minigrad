from collections import deque

e = 2.718281828459045

# Wrapper for scalar values
class Value:
    def __init__(self, value, children=(), op=None):
        self.data = value
        self.grad = 0
        
        # Tuple containing list of 1+ operands
        self.children = children

        # Operator used to calculate the current value
        self.op = op
    
    # Propagates the current gradient backward to the value's children using the current operator and its derivative
    def calc_grad(self):
        if self.children:
            # If undefined, all gradients become undefined
            if self.grad == 'undefined':
                for c in self.children:
                    c.grad = 'undefined'

            # Check for custom operators, if not use default arithmetic derivative calculations
            elif self.op == 'ln':
                self.ln_grad(self.children[0])
            elif self.op == 'relu':
                self.relu_grad(self.children[0])
            elif self.op == 'leaky_relu':
                self.leaky_relu_grad(self.children[0])
            elif self.op == 'sigmoid':
                self.sigmoid_grad(self.children[0])
            else:
                self.default_grad()

    # Sorts a value DAG topologically and propagates the gradient through it, starting from the most recently calculated value
    # Also zeroes the gradients for each value
    def backward(self):
        q = deque([self])
        topo = []
        visited = set()
        visited.add(self)
        
        while q:
            curr = q[-1]
            visited.add(curr)
            curr.grad = 0
            children = [c for c in curr.children if isinstance(c, Value) and c not in visited]
            if not children:
                topo.append(curr)
                q.pop()
            else:
                for c in children:
                    q.append(c)

        # Set current top node gradient to 1 to prepare for backpropagation
        self.grad = 1
        for n in reversed(topo):
            n.calc_grad()

    # Basic arithmetic derivatives
    def default_grad(self):
        a,b = self.children
        if self.op == '+':
            a.grad += self.grad
            b.grad += self.grad
        elif self.op == '*':
            a.grad += b.data*self.grad
            b.grad += a.data*self.grad
        elif self.op == '**':
            a.grad += (b.data*a.data**(b.data-1))*self.grad
            ln = a.ln().data
            if ln != 'undefined' and ln != -float('inf'):
                b.grad += (a.ln().data*a.data**b.data)*self.grad
            else: b.grad = 'undefined'


    # ---------------------------------
    # CUSTOM OPERATIONS AND DERIVATIVES

    # Natural log approximation
    def ln(self):
        if self.data < 0: return Value('undefined', (self,), op='ln')
        elif self.data == 0: return Value(-float('inf'), (self,), op='ln')
        else: return Value(99999999*(self.data**(1/99999999)-1), (self,), op='ln')
    def ln_grad(self, operand):
        assert isinstance(operand, Value)

        if operand.data == 'undefined': operand.grad = 'undefined'
        elif operand.data == 0: operand.grad += float('inf')
        else: operand.grad += (1/operand.data)*self.grad
    
    # Returns a value between 0 and 1
    def sigmoid(self):
        return Value(1/(1+e**(-self.data)), (self,), 'sigmoid')
    def sigmoid_grad(self, operand):
        assert isinstance(operand, Value)
        sig = operand.sigmoid().data
        operand.grad += sig * (1 - sig) * self.grad
    
    # Returns a value clamped to >= 0
    def relu(self):
        if self.data <= 0: return Value(0, (self,), 'relu')
        return Value(self.data, (self,), 'relu')
    def relu_grad(self, operand):
        assert isinstance(operand, Value)
        operand.grad += (self.data != 0) * self.grad
    
    # Returns a value that is decreased significantly if it is smaller than 0
    def leaky_relu(self):
        if self.data <= 0: return Value(self.data*0.01, (self,), 'leaky_relu')
        return Value(self.data, (self,), 'leaky_relu')
    def leaky_relu_grad(self, operand):
        assert isinstance(operand, Value)
        if self.data <= 0: operand.grad += (0.01)*self.grad
        else: operand.grad += self.grad
    
    # ---------------------------------



    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'
    
    def __add__(self, o):
        other = o if isinstance(o, Value) else Value(o)
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, o):
        other = o if isinstance(o, Value) else Value(o)
        return Value(self.data * other.data, (self, other), '*')

    def __pow__(self, o):
        other = o if isinstance(o, Value) else Value(o)
        return Value(self.data ** other.data, (self, other), '**')

    def __rpow__(self, o):
        other = o if isinstance(o, Value) else Value(o)
        return Value(other.data ** self.data, (other, self), '**')

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1