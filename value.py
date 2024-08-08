from collections import deque
import copy

e = 2.718281828459045

# Wrapper for algebraic types
class Value:
    def __init__(self, value, children=(), op=None):
        self.data = value
        self.grad = 0
        
        self.children = children
        self.op = op
    
    def calc_grad(self):
        if self.children:
            a,b = self.children
            if self.grad == 'undefined':
                a.grad = 'undefined'
                if b:
                    b.grad = 'undefined'
            elif self.op == '+':
                a.grad += self.grad
                b.grad += self.grad
            elif self.op == '*':
                a.grad += b.data*self.grad
                b.grad += a.data*self.grad
            elif self.op == '**':
                a.grad += (b.data*a.data**(b.data-1))*self.grad
                b.grad += (a.ln().data*a.data**b.data)*self.grad
            elif self.op == 'ln':
                if a.data == 0:
                    a.grad += float('inf')
                else:
                    a.grad += (1/a.data)*self.grad
            elif self.op == 'relu':
                a.grad += (self.data != 0) * self.grad
            elif self.op == 'leaky_relu':
                if self.data <= 0:
                    a.grad += (0.01)*self.grad
                else:
                    a.grad += self.grad
            elif self.op == 'sigmoid':
                sig = a.sigmoid().data
                a.grad += sig * (1 - sig) * self.grad
    def backward(self):
        topo = []
        visited = set()
        def topo_sort(nodes):
            for n in nodes:
                if isinstance(n, Value):
                    if n not in visited:
                        visited.add(n)
                        topo_sort(n.children)
                        topo.append(n)
        topo_sort(self.children)
        topo.append(self)
        
        self.grad = 1
        for n in reversed(topo):
            n.calc_grad()
    
    # Natural log approximation
    def ln(self):
        if self.data < 0:
            return Value('undefined', (self, None), op='ln')
        elif self.data == 0:
            return Value(-float('inf'), (self, None), op='ln')
        else: 
            return Value(99999999*(self.data**(1/99999999)-1), (self,None), op='ln')
        
    def sigmoid(self):
        return Value(1/(1+e**(-self.data)), (self, None), 'sigmoid')
    
    def relu(self):
        if self.data <= 0:
            return Value(0, (self, None), 'relu')
        return Value(self.data, (self, None), 'relu')
    
    def leaky_relu(self):
        if self.data <= 0:
            return Value(self.data*0.01, (self, None), 'leaky_relu')
        return Value(self.data, (self, None), 'leaky_relu')

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