
import math
import random

    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    inputs: list[int]
    weights: list[int]
    bias: int = 0
    learning_rate = 0.001

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
        return self.learning_rate * (target - output)
    
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
AND_LOGIC = [((1,1), 1), ((1,0), 0), ((0,1), 0), ((0,0), 0)]

n_or = Neuron([1, 1])
n_and = Neuron([1,1])

def train(neuron, LOGIC, epochs, learning_rate=0.0001):
    learning_rate = learning_rate

    for epoch in range(int(epochs)):
        for inputs, target in LOGIC:
            neuron.new_inputs(list(inputs))
            neuron.improve(target)

train(n_or, OR_LOGIC, 0.001, 100000)
train(n_and, AND_LOGIC, 0.001, 100000)

# OR's input then AND's input like [OR, AND]

MIDDLE_LOGIC = [((1,1), 0), ((1,0), 1), ((0,1), 1), ((0,0), 0)]

n_middle = Neuron([1,1])

train(n_middle, MIDDLE_LOGIC, 100000)

XOR_LOGIC = [((1,1), 0), ((1,0), 1), ((0,0), 0)]


n_or_in = n_or.forward()
n_and_in = n_and.forward()
print(n_or_in)
print(n_and_in)

n_middle.new_inputs([n_or_in, n_and_in])
print(n_middle.forward())
