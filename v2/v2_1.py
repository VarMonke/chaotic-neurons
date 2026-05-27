import math
import random

class InputParameter:
    value: float | int
    neuron: "Neuron"

    def __init__(self, values, neuron):
        self.value = values
        self.neuron = neuron

class Neuron:
    inputs: list[InputParameter]
    bias: float
    weights: list[float]

    learning_rate: float

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.output = 0
        self.initialize()

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def initialize(self):
        self.weights = [((random.randint(-1000, 1000))/1000) for _ in range(len(self.inputs))]
        self.bias = random.randint(-1000, 1000) / 1000

    def set_input(self, inputs: list[int]):
        self.inputs = inputs
        return None
    
    def error_correction(self, output, target_output):
        return self.learning_rate * (target_output - output)

class BaseNeuron(Neuron):
    
    def forward(self):
        total = 0
        for inp, weight in zip(self.inputs, self.weights):
            total += inp.value * weight
        
        total += self.bias

        return self.sigmoid(total)
    
    def improve_and_forward(self, target_output):
        prediction = self.forward()

        if prediction == target_output:
            pass

        else:
            for i in range(len(self.weights)):
                weight_change = self.inputs[i].value * self.error_correction(prediction, target_output)
                self.weights[i] += weight_change
            
            self.bias += self.error_correction(prediction, target_output)

        return self.forward()


class MNeuron(Neuron):

    def forward(self):
        total = 0
        for inp, weight in zip(self.inputs, self.weights):
            total += inp.value * weight
        
        total += self.bias

        return self.sigmoid(total)
    


"""
n1 gives the output x1 based on i = [a, b], w = [x, y] and b1
n1 gives the output x1 based on i = [p, q], w = [r, s] and b2

then you do xi = sigmoid(i dot w + bi) for each

then we compute the n3 outcome as OUTPUT = x1 * w3a + x2 * w3b + b3

now we can figure out our cost function which is

cost = (TARGET - OUTPUT) * learning_rate

now we can update b3 directly, by doing b3 += cost

but n3 has 0 control over x3, so what it does? it just calculates the "theoretical cost" by x1 and x2, so it updates the weights of x1, and x2.
THEN  we could maybe add a "cost" parameter to the InputParameter

so we update a InputParameters cost

for inp in self.inputs:
    self.inp.cost = cost * self.inp.value #i.e we just updated our cost based on the value of the inputs

now we can do a neuron.update_parents method which just tells the parent that, this output from you(which is now the n3's input) causes me this much hassle,
go improve your weights and biases and go tell whoever is before you as well to improve their weights and biases

so

def update_parents(self):
    # this is a neuron function
    for inp in self.inputs:
        inp.cost = self.error_cost * inp.value
        inp.source_neuron.update_self(inp)

def update_self():
    # update your weights and biases by using 
    inp.cost 
    and then call self.update_parents
    
    
    
we can't control x1 also.. there might be an upstream neuron that controls x1 by the weights and biases

so every neuron has only 3 properties it can change, it's bias and it's weights, everything else is not it's authority.

so... the last neuron of our chain finds a cost like the single-neuron system and updates it's weights and bias,
and is very happy with itself, but it has to tell it's parents about their outputs being shitty. how do you compute their shittiness?

you see how much n3 could self correct, so this n3 decided to update it's internal weight related to x1
by a specific amount because that is how much it could change, now if this change is positive,
we need more of x1, else we need less. but hmm, n1 can only change the weights and biases so x1 can change..
so we can calculate the n1_causation_value or something which is just n3's difference in x1's weight?
"""