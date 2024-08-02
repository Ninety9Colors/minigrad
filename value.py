from collections import deque
import copy

OPERATIONS = ['+', '-', '*', '/', '//', '%', '**']

def operate(left, operation, right):
    if operation == '-':
        return left.calc()-right.calc()
    elif operation == '+':
        return left.calc()+right.calc()
    elif operation == '*':
        return left.calc()*right.calc()
    elif operation == '/':
        return left.calc()/right.calc()
    elif operation == '//':
        return left.calc()//right.calc()
    elif operation == '%':
        return left.calc()%right.calc()
    elif operation == '**':
        return left.calc()**right.calc()
    return None

# Wrapper for algebraic types, no type checking
class Value:
    def __init__(self, value=[0]):
        if isinstance(value, list):
            self.data = value
        elif isinstance(value, Value):
            self.data = value.data
        else:
            self.data = [value]
        self.grad = None

    def __repr__(self):
        result = ''
        for value in self.data: result += str(value) 
        return result
    
    def __add__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot add non Value object: {o=} to Value")
        return Value(['('] + self.data + [')', '+', o])

    def __sub__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot subtract non Value object: {o=} from Value")
        return Value(['('] + self.data + [')', '-', o])

    def __mul__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot multiply non Value object: {o=} with Value")
        return Value(['('] + self.data + [')', '*', o])
    
    def __truediv__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot true divide non Value object: {o=} onto Value")
        return Value(['('] + self.data + [')', '/', o])

    def __floordiv__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot floor divide non Value object: {o=} onto Value")
        return Value(['('] + self.data + [')', '//', o])

    def __mod__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot mod non Value object: {o=} onto Value")
        return Value(['('] + self.data + [')', '%', o])

    def __pow__(self, o):
        if not isinstance(o, Value):
            raise TypeError(f"Cannot pow non Value object: {o=} onto Value")
        return Value(['('] + self.data + [')', '**', o])

    
    # TODO: Finish implementing calc method, already implemented parentheses and exponent step

    # Parses data into a single value output
    # Assumes expression is in valid mathematical notation, e.g. parentheses are closed, operators have left and right side, etc.
    def calc(self, expression=None):
        # Expression should be None for client use, it has other functionality to allow for recursion
        if not expression:
            expression = self.data
        if not isinstance(expression, list):
            if isinstance(expression, Value):
                return expression.calc()
            return expression
        # Copy to avoid modifying in place
        expression = copy.deepcopy(expression)

        # Find indexes of outermost parentheses pair, if possible
        openIndex = None
        closeIndex = None
        openCount = 0
        for i, value in enumerate(expression):
            if value == '(':
                if openCount == 0:
                    openIndex = i
                openCount += 1
            elif value == ')':
                if openCount == 1:
                    closeIndex = i
                    break
                openCount -= 1
        
        # Recurse and calculate inner parentheses first, after this step no parentheses will be left
        if openIndex != None:
            expression = expression[:openIndex] + [self.calc(expression[openIndex+1:closeIndex])] + expression[closeIndex+1:]
        
        for operation in OPERATIONS:
            while operation in expression:
                index = expression.index(operation)
                left = Value(expression[index-1])
                right = Value(expression[index+1])

                expression = expression[:index-1] + [operate(left, operation, right)] + expression[index+2:]
        
        return expression[0]