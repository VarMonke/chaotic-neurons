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
        self.weights = [((random.randint(-1000, 1000))/1000) for _ in range(len(self.inputs))]
        self.bias = random.randint(-1000, 1000) / 1000

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
AND_LOGIC = [((1,1), 1), ((1,0), 0), ((0,1), 0), ((0,0), 0)]
XOR_LOGIC = [((1,1), 0), ((1,0), 1), ((0,1), 1), ((0,0), 0)]

for _ in range(10000):
    for inputs, target in XOR_LOGIC:
        n.set_input(list(inputs))
        n.improve_and_forward(target)

n.set_input([0,1])
print(n.forward())

print(n.bias, n.weights)

"""
here, i made the the weights be -ve as well, and i learnt 2 things

The no.of epochs i'm running is high enough that even if a weight starts at -1, the model can pick up correlation and update it to be the right value finally
seconds, XOR doesn't show you one relationship, XOR has 2 relationships, i.e XOR is a combination of 2 things happening, that a single neuron cant comprehend and it'll just give you an output close to 0.5 in every case cause it's confused about the 2 things, both activated so it reduce weights and bias, but both not activated should be a one so it increases bias, and in the 0,1 and 1,0 case it tends to icnrease bias and weight of one side, but then this gets decreased in the other side.

HMMM
"""