import math
import random

def dot(i, w):
    out = 0
    for i, w in zip(i, w):
        out += i * w
    return out

def sigmoid(x) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

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

    def __init__(self, width: int, prev_layer: "Layer", next_layer: "Layer"):
        self.width = width
        self.prev_layer = prev_layer
        self.next_layer = next_layer
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
            self.outputs[i] = sigmoid(dot(self.prev_layer.outputs, w) + b)
        if self.next_layer is not None:
            self.next_layer.forward()

    def backward(self, grads: list[float | int]):
        _grads = [0.0 for _ in range(self.prev_layer.width)]
        for i, grad, output in zip(range(0, self.width), grads, self.outputs):
            delta = grad * output * (1 - output)
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


inp = InputLayer(5)

hidden = Layer(
    width=8,
    prev_layer=inp,
    next_layer=None,
)

out = Layer(
    width=1,
    prev_layer=hidden,
    next_layer=None,
)

hidden.next_layer = out

def predict(x):
    inp.outputs = list(x)

    hidden.forward()

    return out.outputs[0]

ds = parity_ds(5)

for epoch in range(10000):

    total_loss = 0.0
    correct = 0

    for x, y in ds:

        pred = predict(x)

        loss = (pred - y) ** 2
        total_loss += loss

        if round(pred) == y:
            correct += 1

        grad = [
            2 * (pred - y)
        ]

        out.backward(grad)

        hidden.step(0.1)

        hidden.flush()

    print(
        epoch,
        "loss:",
        total_loss / len(ds),
        "acc:",
        correct / len(ds),
    )

