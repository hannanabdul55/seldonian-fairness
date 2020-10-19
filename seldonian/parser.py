class Stack:
    def __init__(self):
        self.array = []
        pass

    def pop(self):
        if self.size() < 1:
            return None
        elem = self.array[-1]
        self.array = self.array[:-1]
        return elem

    def peek(self):
        return self.array[-1]

    def size(self):
        return len(self.array)

    def push(self, elem):
        self.array.append(elem)
        return elem

    def data(self):
        return self.array


ops = ['+', '-', '*', '/', '(', ')']


def parse_ghat(statement, vars={}):
    tokens = list(filter(lambda x: len(x) > 0, statement.split(' ')))
    stack = Stack()
    op_stack = Stack()
    var_stack = Stack()
    for tok in tokens:
        if tok in ops:
            # operator token
            op_stack.push(tok)
        else:
            # push the operand to stack
            if tok not in vars:
                stack.push(tok)
            else:
                stack.push(vars[tok])

    pass
