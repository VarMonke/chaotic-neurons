
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

    def new_inputs(self, inputs: list[int]):
        self.inputs = inputs

    def forward(self):
        total = 0
        for input, weight in zip(self.inputs, self.weights):
            total += input * weight

        return sigmoid(total + self.bias)
    
    def error(self, output, target):
        return (target - output)
    
    def improve(self, target):
        prediction = self.forward()

        if prediction == target:
            pass

        else:
            for i in range(len(self.weights)):
                weight_change = self.inputs[i] * self.error(prediction, target)
                self.weights[i] += weight_change
            
            self.bias += self.error(prediction, target)

        return self.forward()


OR_LOGIC = [((1,1), 1), ((1,0), 1), ((0,1), 1), ((0,0), 0)]

n = Neuron([1, 1])

for epoch in range(100):
    for inputs, target in OR_LOGIC:
        n.new_inputs(list(inputs))
        n.improve(target)

print("FINISHED LEARNING; NOW SOLVING")

n.inputs = [1,1]
print(n.forward())

n.inputs = [1,0]
print(n.forward())

n.inputs = [0,1]
print(n.forward())

n.inputs = [0,0]
print(n.forward())
