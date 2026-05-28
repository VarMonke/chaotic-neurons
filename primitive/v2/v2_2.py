import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class InputParameter:
    value: float | int
    cost: float | int
    source_neuron: "Neuron"
    

class Neuron:
    inputs: list[InputParameter]
    weights: list[float | int]
    bias: float | int
    learning_rate: float | int

    shape: int # this dictates size of the input param list

    def __init__(self, shape, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.shape = shape
        self.initialize()

    def initialize(self):
        self.weights = [((random.randint(-1000, 1000))/1000) for _ in range(self.shape)]
        self.bias = random.randint(-1000, 1000) / 1000

    def forward(self):
        total = 0
        for i, w in zip(self.inputs, self.weights):
            total += i.value * w
        
        total += self.bias 

        return sigmoid(total)
    
class StartNeuron(Neuron):
    def backward(self):
        pass

class EndNeuron(Neuron):

    def calculate_cost(self, target_output):
        return (target_output - self.forward()) * self.learning_rate
    
    def update_self(self, cost):

        for i in range(self.shape):
            weight_change = self.inputs[i].value * cost
            self.weights[i] += weight_change
            self.inputs[i].cost = self.inputs[i].value * weight_change
        
        self.bias += cost

    def backward(self, target_output):
        cost = self.calculate_cost(target_output)
        self.update_self(cost)

        for inp in self.inputs:
            inp.source_neuron.update_self(inp)

class MiddleNeuron(Neuron):
    
    def update_self(self, inp: InputParameter):
        for i in range(len(self.weights)):
            weight_change = (self.inputs[i].value * inp.cost) * self.learning_rate
            self.weights[i] += weight_change
            self.inputs[i].cost = self.inputs[i].value * weight_change
        
        self.bias += inp.cost

        self.backward()

    def backward(self):
        for inp in self.inputs:
            inp.source_neuron.update_self(inp)