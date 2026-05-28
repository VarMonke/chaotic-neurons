import math
import random


def dot(inputs: list["Value"], weights: list[float]) -> float:
    res = 0
    for i, w in zip(inputs, weights):
        res += i.value * w

    return res

def sigmoid(x) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
    
def calculate_output(inputs: list["Value"], weights: list[float], bias: float | int) -> float:
    return sigmoid(dot(inputs, weights) + bias)

def delta(target: float, output: float) -> float:
    return 2 * (output - target) * (output * (1 - output))

def grad(delta: float, param: float) -> float:
    return delta * param

def d_sigmoid(x: float) -> float:
    return x * (1 - x)

class Value:
    value: float
    grad: float

    source_neuron: "Neuron"


class Neuron:
    inputs: list[Value]
    last_output: float
    weights: list[float]
    bias: float
    shape: int

    learning_rate: float

    def __init__(self, shape, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate
        self.shape = shape
        self.initialize()
    
    def initialize(self) -> None:
        self.weights = [((random.randint(-500, 500))/1000) for _ in range(self.shape)]
        self.bias = random.randint(-500, 500) / 1000

    def duplicate_values(self) -> tuple[list[Value], list[float]]:
        return ([v for v in self.inputs], [w for w in self.weights])
    
    def forward(self) -> float:
        output = calculate_output(self.inputs, self.weights, self.bias)
        self.last_output = output

        return output
    
    def backward(self, *args, **kwargs):
        pass

class EndNeuron(Neuron):

    def cost(self, output: float, target: float) -> float:
        return (target - output) ** 2
    
    def backward(self, output: float, target: float) -> float:

        local_delta = delta(target, output)
        
        forward_inp, forward_weights = self.duplicate_values()
        self.bias -= self.learning_rate * grad(local_delta, 1)

        for i in range(self.shape):
            self.weights[i] -= self.learning_rate * grad(local_delta, forward_inp[i].value)
            inp_grad = grad(local_delta, forward_weights[i])
            #print(abs(inp_grad))
            self.inputs[i].source_neuron.backward(inp_grad)
            

class MiddleNeuron(Neuron):

    def backward(self, inp_grad):
        # in one sense, our local_delta is inp_grad * learning_rate
        local_delta = inp_grad * d_sigmoid(self.last_output)

        forward_inp, forward_weights = self.duplicate_values()

        self.bias -= self.learning_rate * grad(local_delta, 1)

        for i in range(self.shape):
            self.weights[i] -= self.learning_rate * grad(local_delta, forward_inp[i].value)
            inp_grad = grad(local_delta, forward_weights[i])
            self.inputs[i].source_neuron.backward(inp_grad)



from itertools import product
import random

PARITY_DATA = [
    (bits, sum(bits) % 2)
    for bits in product([0, 1], repeat=4)
]

# input leaves
inputs = [Value() for _ in range(4)]
for inp in inputs:
    inp.source_neuron = Neuron(0)

# hidden layer 1
hidden1 = [
    MiddleNeuron(shape=4, learning_rate=0.1)
    for _ in range(16)
]

# hidden layer 2
hidden2 = [
    MiddleNeuron(shape=16, learning_rate=0.1)
    for _ in range(12)
]


hidden3 = [
    MiddleNeuron(shape=12, learning_rate=0.1)
    for _ in range(8)
]


hidden4 = [
    MiddleNeuron(shape=8, learning_rate=0.1)
    for _ in range(4)
]


hidden5 = [
    MiddleNeuron(shape=4, learning_rate=0.1)
    for _ in range(2)
]

# output
out = EndNeuron(shape=2, learning_rate=0.1)


def predict(bits):

    # inputs
    for inp, val in zip(inputs, bits):
        inp.value = val

    # layer 1 (16)
    l1 = []
    for h in hidden1:
        h.inputs = inputs

        v = Value()
        v.value = h.forward()
        v.source_neuron = h

        l1.append(v)

    # layer 2 (12)
    l2 = []
    for h in hidden2:
        h.inputs = l1

        v = Value()
        v.value = h.forward()
        v.source_neuron = h

        l2.append(v)

    # layer 3 (8)
    l3 = []
    for h in hidden3:
        h.inputs = l2

        v = Value()
        v.value = h.forward()
        v.source_neuron = h

        l3.append(v)

    # layer 4 (4)
    l4 = []
    for h in hidden4:
        h.inputs = l3

        v = Value()
        v.value = h.forward()
        v.source_neuron = h

        l4.append(v)

    # layer 5 (2)
    l5 = []
    for h in hidden5:
        h.inputs = l4

        v = Value()
        v.value = h.forward()
        v.source_neuron = h

        l5.append(v)

    # output
    out.inputs = l5
    pred = out.forward()

    return pred, l1, l2, l3, l4, l5


def evaluate():

    correct = 0
    loss = 0

    for bits, target in PARITY_DATA:

        pred, *_ = predict(bits)

        loss += (target - pred) ** 2

        if int(pred > 0.5) == target:
            correct += 1

    return correct, loss


# TRAIN
for epoch in range(100000):

    random.shuffle(PARITY_DATA)

    for bits, target in PARITY_DATA:

        pred, *_ = predict(bits)

        out.backward(pred, target)

    if epoch % 1000 == 0:

        correct, loss = evaluate()

        print(
            f"epoch={epoch:6d} "
            f"acc={correct}/16 "
            f"loss={loss:.6f}"
        )

        if correct == 16:
            print("\nSOLVED\n")
            break


print("\n===== FINAL TEST =====\n")

correct = 0

for bits, target in PARITY_DATA:

    pred, *_ = predict(bits)

    cls = int(pred > 0.5)

    if cls == target:
        correct += 1

    print(
        f"{bits} "
        f"target={target} "
        f"pred={pred:.6f} "
        f"class={cls}"
    )

print(f"\nAccuracy: {correct}/16")


print("\n===== NETWORK STATS =====\n")

print("Output weights:")
print(out.weights)

print("\nOutput bias:")
print(out.bias)


for name, layer in [
    ("H1", hidden1),
    ("H2", hidden2),
    ("H3", hidden3),
    ("H4", hidden4),
    ("H5", hidden5),
]:

    print(f"\n{name} weight stats")

    all_weights = []

    for neuron in layer:
        all_weights.extend(neuron.weights)

    print("min:", min(all_weights))
    print("max:", max(all_weights))
    print("avg abs:", sum(abs(w) for w in all_weights)/len(all_weights))


print("\n===== ACTIVATION RANGES =====\n")

for name, layer in [
    ("H1", hidden1),
    ("H2", hidden2),
    ("H3", hidden3),
    ("H4", hidden4),
    ("H5", hidden5),
]:

    print(f"\n{name}")

    for idx, neuron in enumerate(layer):

        vals = []

        for bits, _ in PARITY_DATA:

            predict(bits)

            vals.append(neuron.last_output)

        print(
            idx,
            f"min={min(vals):.4f}",
            f"max={max(vals):.4f}",
            f"spread={max(vals)-min(vals):.4f}"
        )


print("\n===== REPRESENTATION TEST =====\n")

for bits in [
    (0,0,0,0),
    (1,1,1,1),
    (0,0,0,1),
    (1,0,1,1),
]:

    pred, l1, l2, l3, l4, l5 = predict(bits)

    print("\nINPUT:", bits)

    print(
        "L1:",
        [round(v.value, 3) for v in l1[:6]]
    )

    print(
        "L2:",
        [round(v.value, 3) for v in l2[:6]]
    )

    print(
        "L3:",
        [round(v.value, 3) for v in l3[:6]]
    )

    print(
        "L4:",
        [round(v.value, 3) for v in l4]
    )

    print(
        "L5:",
        [round(v.value, 3) for v in l5]
    )

    print(
        "OUT:",
        round(pred, 6)
    )