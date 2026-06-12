import math
import random

    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    inputs: list[int]
    weights: list[int]
    bias: int = 0

    def __init__(self, inputs: list[int]):
        self.inputs = inputs
        self.weights = [((random.randint(0, 1000))/1000) for n in range(len(self.inputs))]
        self.bias = ((random.randint(0, 1000))/1000)

    def forward(self):
        total = 0
        for input, weight in zip(self.inputs, self.weights):
            total += input * weight

        return sigmoid(total + self.bias)
