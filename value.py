from collections import deque
import copy

# Wrapper for algebraic types
class Value:
    def __init__(self, value, children=(), op=''):
        self.data = value
        self.grad = 0
        self.respect = None
        
        self.children = children
        self.op = op

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