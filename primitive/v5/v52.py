import math
import random

def dot(i, w):
    out = 0
    for i, w in zip(i, w):
        out += i * w
    return out

class Activation:

    def forward(self, v):
        pass

    def grad(self, v):
        pass

class Sigmoid(Activation):
    
    def forward(self, v):
        if v >= 0:
            z = math.exp(-v)
            return 1 / (1 + z)
        else:
            z = math.exp(v)
            return z / (1 + z)
        
    def grad(self, v):
        return v * (1 - v)

class ReLU(Activation):

    def forward(self, v):
        return max(0, v)

    def grad(self, v):
        return 1.0 if v > 0 else 0.0

class tanh(Activation):

    def forward(self, v):
        if v >= 0:
            z = math.exp(-2 * v)
            return (1 - z) / (1 + z)
        else:
            z = math.exp(2 * v)
            return (z - 1) / (z + 1)
        
    def grad(self, v):
        return 1 - v * v
        
class Layer:
    input_size: int
    width: int
    weights: list[list[ float | int | None]]
    biases: list[ float | int | None]
    outputs: list[ float | int | None]

    weight_grads: list[list[float | int | None]]
    bias_grads: list[float | int | None]
    lr: float | int | None

    prev_layer: "Layer" | None
    next_layer: "Layer" | None

    def __init__(self, width: int, activation: Activation, prev_layer: "Layer", next_layer: "Layer"):
        self.width = width
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.activation = activation
        self.initialize()

    def initialize(self):
        self.weights = [
            [
                random.uniform(-0.5, 0.5) for _ in range(self.prev_layer.width)
            ] for _ in range(self.width)
        ]

        self.biases = [random.uniform(-0.5, 0.5) for _ in range(self.width)]
        self.outputs = [0.0 for _ in range(self.width)]

        self.weight_grads = [
            [
                0.0 for _ in range(self.prev_layer.width)
            ] for _ in range(self.width)
        ]
        self.bias_grads = [0.0 for _ in range(self.width)]

    def forward(self):
        for i, w, b in zip(range(0, self.width), self.weights, self.biases):
            self.outputs[i] = self.activation.forward(dot(self.prev_layer.outputs, w) + b)
        if self.next_layer is not None:
            self.next_layer.forward()

    def backward(self, grads: list[float | int]):
        _grads = [0.0 for _ in range(self.prev_layer.width)]
        for i, grad, output in zip(range(0, self.width), grads, self.outputs):
            delta = grad * self.activation.grad(output)
            for j, inp in enumerate(self.prev_layer.outputs):
                self.weight_grads[i][j] = delta * inp
                _grads[j] += delta * self.weights[i][j]
            self.bias_grads[i] = delta
        if self.prev_layer:
            self.prev_layer.backward(_grads)

    def step(self, lr: float = 0.01):
        for i in range(self.width):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= lr * self.weight_grads[i][j]
            self.biases[i] -= lr * self.bias_grads[i]
        if self.next_layer is None:
            return
        self.next_layer.step(lr)

    def flush(self):
        for i in range(self.width):
            for j in range(len(self.weight_grads[i])):
                self.weight_grads[i][j] = 0.0
            self.bias_grads[i] = 0.0
        if self.next_layer is not None:
            self.next_layer.flush()


from itertools import product

def parity_ds(max_bits):
    dataset = []

    for bit_count in range(1, max_bits + 1):

        for sample in product(
            [0, 1],
            repeat=bit_count,
        ):

            padded = (
                list(sample)
                + [0] * (max_bits - bit_count)
            )

            parity = sum(sample) % 2

            dataset.append(
                (
                    tuple(float(x) for x in padded),
                    float(parity),
                )
            )

    random.shuffle(dataset)

    return dataset

class InputLayer:
    def __init__(self, width):
        self.width = width
        self.outputs = [0.0 for _ in range(width)]

    def backward(self, grads):
        pass


from itertools import product

def gen_ds(n_bits):
    dataset = []

    for bits in product([0, 1], repeat=n_bits):
        parity = sum(bits) % 2
        dataset.append((bits, parity))

    return dataset


def make_network(bits, inp_size, hidden_widths, activations, out_activation, lr = None):
    ds = gen_ds(bits)

    inp = InputLayer(inp_size)

    hidden_layers = []

    prev = inp
    for i, width in enumerate(hidden_widths):
        layer = Layer(
            width,
            activations[i],
            prev_layer=prev,
            next_layer=None
        )

        if hasattr(prev, "next_layer"):
            prev.next_layer = layer

        hidden_layers.append(layer)
        prev = layer

        out = Layer(width=1, activation=out_activation, prev_layer=prev, next_layer=None,)
        prev.next_layer = out
        
        return inp, hidden_layers, out
    
def predict(inp, out, hidden_layers):
    inp.outputs = list(inp.outputs)
    if hidden_layers:
        hidden_layers[0].forward()
    else:
        out.forward()
    return out.outputs[0]