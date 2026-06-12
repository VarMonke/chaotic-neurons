import math
import random

import numpy as np

from activations import Activation
        
class Layer:
    input_size: int
    width: int

    weights: np.ndarray
    biases: np.ndarray
    outputs: np.ndarray

    weight_grads: np.ndarray
    bias_grads: np.ndarray

    lr: float | None

    prev_layer: "Layer | InputLayer | None"
    next_layer: "Layer | None"

    def __init__(self, width: int, activation: Activation, prev_layer: "Layer", next_layer: "Layer"):
        self.width = width
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.activation = activation
        self.initialize()

    def initialize(self):
        self.weights = np.random.uniform(
            -0.5,
            0.5,
            (self.width, self.prev_layer.width)
        )

        self.biases = np.random.uniform(
            -0.5,
            0.5,
            self.width
        )

        self.outputs = np.zeros(self.width)

        self.weight_grads = np.zeros_like(self.weights)

        self.bias_grads = np.zeros_like(self.biases)

    def forward(self):
        self.outputs = self.activation.forward(np.dot(self.weights, self.prev_layer.outputs) + self.biases)
        if self.next_layer is not None:
            self.next_layer.forward()

    def backward(self, grads:  np.ndarray):
        delta = grads * self.activation.grad(self.outputs)
        self.weight_grads += np.outer(delta, self.prev_layer.outputs)
        self.bias_grads += delta
        _grads = np.dot(self.weights.T, delta)
        if self.prev_layer:
            self.prev_layer.backward(_grads)

    def step(self, lr: float = 0.01):
        self.weights -= lr * self.weight_grads
        self.biases -= lr * self.bias_grads
        if self.next_layer is None:
            return
        self.next_layer.step(lr)

    def flush(self):
        self.weight_grads.fill(0)
        self.bias_grads.fill(0)
        if self.next_layer is not None:
            self.next_layer.flush()


class InputLayer:
    width: int
    outputs: np.ndarray

    def __init__(self, width: int):
        self.width = width
        self.outputs = np.zeros(width)

    def backward(self, grads: np.ndarray):
        pass

