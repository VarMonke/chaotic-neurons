import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0

        self.parents = parents
        self.op = op

        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):

        out = Value(
            self.data + other.data,
            parents=(self, other),
            op="+",
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):

        out = Value(
            self.data * other.data,
            parents=(self, other),
            op="*",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (other * Value(-1))

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, power):

        out = Value(
            self.data ** power,
            parents=(self,),
            op=f"**{power}",
        )

        def _backward():
            self.grad += (
                power
                * (self.data ** (power - 1))
                * out.grad
            )

        out._backward = _backward

        return out
    
    def tanh(self):

        t = math.tanh(self.data)

        out = Value(
            t,
            parents=(self,),
            op="tanh",
        )

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward

        return out

    def relu(self):

        out = Value(
            max(0, self.data),
            parents=(self,),
            op="relu",
        )

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out
    
    def backward(self):

        topo = []
        visited = set()

        def build(node):

            if node not in visited:

                visited.add(node)

                for parent in node.parents:
                    build(parent)

                topo.append(node)

        build(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


a = Value(3.0)
b = Value(7.0)

w = a * b
x = w + w

x.backward()

print(a)
print(b)
print(w)
print(x)