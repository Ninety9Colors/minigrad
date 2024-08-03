from collections import deque
import copy

# Wrapper for algebraic types
class Value:
    def __init__(self, value, children=(), op=''):
        self.data = value
        self.grad = 0
        self.respect_to = None
        
        self.children = children
        self.op = op
    
    def calc_grad(self, respect_to=None):
        if self.children:
            a,b = self.children
            if self.op == '+':
                a.grad += self.grad
                b.grad += self.grad

                a.respect_to = respect_to
                b.respect_to = respect_to
            elif self.op == '*':
                a.grad += b.data*self.grad
                b.grad += a.data*self.grad

                a.respect_to = respect_to
                b.respect_to = respect_to
            elif self.op == '**':
                a.grad += (b.data*a.data**(b.data-1))*self.grad
                b.grad += (self.ln(a.data)*a.data**b.data)*self.grad

                a.respect_to = respect_to
                b.respect_to = respect_to
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
            n.calc_grad(self)
    
    # Natural log approximation
    def ln(self, o):
        o = self.data if o == None else o
        assert(o>0)
        return 99999999*(o**(1/99999999)-1)

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