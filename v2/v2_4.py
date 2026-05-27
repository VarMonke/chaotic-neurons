import random
import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

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

    def __init__(self, shape, learning_rate = 0.1):
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
    def update_self(self, inp):
        pass

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

        contributions = []

        for input_param, weight in zip(self.inputs, self.weights):
            contributions.append(input_param.value * weight)

        total_contribution = sum(abs(c) for c in contributions)

        for i in range(len(self.weights)):

            weight_change = self.inputs[i].value * inp.cost

            self.weights[i] += weight_change

            if total_contribution == 0:
                self.inputs[i].cost = 0
            else:
                self.inputs[i].cost = (
                    contributions[i]
                    / total_contribution
                ) * inp.cost

        self.bias += inp.cost

        self.backward()

    def backward(self):
        for inp in self.inputs:
            inp.source_neuron.update_self(inp)


XOR = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
]

# Build network
def make_input(value, neuron=None):
    inp = InputParameter()
    inp.value = value
    inp.cost = 0
    inp.source_neuron = neuron
    return inp


from itertools import product

PARITY3 = []

for bits in product([0, 1], repeat=3):
    target = sum(bits) % 2
    PARITY3.append((bits, target))

start_a = StartNeuron(shape=0)
start_b = StartNeuron(shape=0)
start_c = StartNeuron(shape=0)

hidden = [
    MiddleNeuron(shape=3),
    MiddleNeuron(shape=3),
    MiddleNeuron(shape=3)
]

out = EndNeuron(
    shape=3,
    learning_rate=0.01
)

EPOCHS = 100000

for epoch in range(EPOCHS):

    total_error = 0

    for (a, b, c), target in PARITY3:

        for neuron in hidden:

            neuron.inputs = [
                make_input(a, start_a),
                make_input(b, start_b),
                make_input(c, start_c)
            ]

        hidden_outputs = []

        for neuron in hidden:
            hidden_outputs.append(neuron.forward())

        out.inputs = [
            make_input(hidden_outputs[0], hidden[0]),
            make_input(hidden_outputs[1], hidden[1]),
            make_input(hidden_outputs[2], hidden[2])
        ]

        prediction = out.forward()

        total_error += abs(
            target - prediction
        )

        out.backward(target)

    if epoch % 1000 == 0:
        print(
            f"epoch={epoch}",
            f"error={total_error:.6f}"
        )

print()
print("========== TEST ==========")
print()

correct = 0

for (a, b, c), target in PARITY3:

    for neuron in hidden:

        neuron.inputs = [
            make_input(a, start_a),
            make_input(b, start_b),
            make_input(c, start_c)
        ]

    hidden_outputs = []

    for neuron in hidden:
        hidden_outputs.append(
            neuron.forward()
        )

    out.inputs = [
        make_input(hidden_outputs[0], hidden[0]),
        make_input(hidden_outputs[1], hidden[1]),
        make_input(hidden_outputs[2], hidden[2])
    ]

    prediction = out.forward()

    predicted_class = round(prediction)

    if predicted_class == target:
        correct += 1

    print(
        f"input={(a,b,c)}",
        f"target={target}",
        f"pred={prediction:.4f}",
        f"class={predicted_class}"
    )

print()
print(
    f"Accuracy: {correct}/8 "
    f"({100*correct/8:.2f}%)"
)