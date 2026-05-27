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
    
OR_LOGIC = [((1,1), 1), ((1,0), 1), ((0,1), 1), ((0,0), 0)]
AND_LOGIC = [((1,1), 1), ((1,0), 0), ((0,1), 0), ((0,0), 0)]
XOR_LOGIC = [((1,1), 0), ((1,0), 1), ((0,1), 1), ((0,0), 0)]

def stuff(n, LOGIC):
    for _ in range(10000):
        for inputs, target in LOGIC:
            n.set_input(list(inputs))
            n.improve_and_forward(target)

n_or = Neuron()
n_and = Neuron()
stuff(n_or, OR_LOGIC)
stuff(n_and, AND_LOGIC)

FIGURER_LOGIC = []

for inputs, xor_target in XOR_LOGIC:
    n_or.set_input(inputs)
    n_and.set_input(inputs)

    or_out = n_or.forward()
    and_out = n_and.forward()

    FIGURER_LOGIC.append(
        ((or_out, and_out), xor_target)
    )
    #print(FIGURER_LOGIC)

n_figure_out = Neuron(learning_rate=0.1)

def teach_figurer():
    for _ in range(10000):
        for inputs, target in FIGURER_LOGIC:
            n_figure_out.set_input(list(inputs))
            n_figure_out.improve_and_forward(target)



teach_figurer()

print("TAUGHT, NOW TESTING")

def test_figurer(inp):
    n_figure_out.set_input(inp)
    print(n_figure_out.forward())

    print(n_figure_out.bias, n_figure_out.weights)


test_figurer([1,1])
test_figurer([1,0])
test_figurer([0,1])
test_figurer([0,0])

"""
ahh and yeah the test_figurer is meh

TAUGHT, NOW TESTING
0.0010957105301340484
-6.260675533170095 [13.11089131001769, -13.665471705769798]
0.9989418936633585
-6.260675533170095 [13.11089131001769, -13.665471705769798]
2.2191374753845144e-09
-6.260675533170095 [13.11089131001769, -13.665471705769798]
0.0019063141482235503
-6.260675533170095 [13.11089131001769, -13.665471705769798]

hmmm so for linearly correlated concepts,
a single neuron can reason with enough precision and accuracy to pinpoint the outpuit within a 0.01 error range,
while as if concepts without any correlation or negatriove correlation will more often than not have internal features that have some correlation which can be reasoned about
"""