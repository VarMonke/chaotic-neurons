# -2.9161402272090062 [6.784192084451158, 6.784874530458721]

import math
import random

class Neuron:
    inputs: list[int]
    bias: float
    weights: list[float]

    learning_rate: float

    def __init__(self, learning_rate: float = 0.01):
        self.inputs = [1,1]
        self.learning_rate = learning_rate
        self.output = 0
        self.initialize()

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def initialize(self):
        self.weights = [((random.randint(0, 1000))/1000) for _ in range(len(self.inputs))]
        self.bias = random.randint(0, 1000) / 1000

    def set_input(self, inputs: list[int]):
        self.inputs = inputs
        return None
    
    def forward(self):
        total = 0
        for inp, weight in zip(self.inputs, self.weights):
            total += inp * weight
        
        total += self.bias

        return self.sigmoid(total)
    
    def error_correction(self, output, target_output):
        return self.learning_rate * (target_output - output)
    
    def improve_and_forward(self, target_output):
        prediction = self.forward()

        if prediction == target_output:
            pass

        else:
            for i in range(len(self.weights)):
                weight_change = self.inputs[i] * self.error_correction(prediction, target_output)
                self.weights[i] += weight_change
            
            self.bias += self.error_correction(prediction, target_output)

        return self.forward()
    
n = Neuron()
OR_LOGIC = [((1,1), 1), ((1,0), 1), ((0,1), 1), ((0,0), 0)]

for _ in range(10000):
    for inputs, target in OR_LOGIC:
        n.set_input(list(inputs))
        n.improve_and_forward(target)

n.set_input([0,1])
print(n.forward())

print(n.bias, n.weights)
