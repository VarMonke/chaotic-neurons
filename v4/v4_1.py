from typing import (
    Any,
    Self,
    Tuple,
    List,
    Union,
    Optional,
)

import math
import random

def dot(inputs: list["IOParameter"], weights: list["Weight"]) -> float:
    res = 0
    for i, w in zip(inputs, weights):
        res += i.value * w.value

    return res

def sigmoid(x) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def d_sigmoid(x: float) -> float:
    return x * (1 - x)

class InputVector():
    bits: List[int]

    def __init__(self, value: int) -> None:
        self.bits = [int(d) for d in str(value)]

class Value:
    value: float
    history: list[float]

    def __init__(self, value: float) -> None:
        self.value = value
        self.history = [value]

    def __add__(self, other: "Value") -> "Value":
        return Value(self.value + other.value)
    
    def __iadd__(self, other):
        if isinstance(other, Value):
            return Value(self.value + other.value)
        return Value(self.value + other)
    
    def __sub__(self, other: "Value") -> "Value":
        return Value(self.value - other.value)
    
    def __isub__(self, other):
        if isinstance(other, Value):
            return Value(self.value - other.value)
        return Value(self.value - other)
    
    def flush(self) -> None:
        self.history = []

    def update(self, value: float) -> "Value":
        self.value = value
        self.history.append(value)
        return self
    
class Loss(Value):
    ...

class IOParameter(Value):
    source: "Neuron"

class Weight(Value):
    accumulated_grad: float

    def flush(self) -> None:
        self.accumulated_grad = 0.0
        return super().flush()
    
class Bias(Weight):
    ...

class InputNode:
    output: IOParameter

    def __init__(self):
        self.output = IOParameter(0.0)

class Neuron:
    weights: List[Weight]
    bias: Bias
    
    last_inputs: List[Optional[IOParameter]]
    output: Optional[IOParameter]

    accumulated_grad: float
    stepped: bool

    children: List["Neuron"]
    parents: List["Neuron"]

    received_children: int
    expected_children: int

    delta: Value

    shape: int
    learning_rate: float

    def __init__(self, shape: int, parents: List["Neuron"], children: List["Neuron"], expected_children: int, learning_rate: float = 0.01):
        self.shape = shape
        self.learning_rate = learning_rate
        self.parents = parents
        self.children = children
        self.expected_children = expected_children
        self.initialize()

    def initialize(self):
        self.weights = [Weight(random.uniform(-0.5, 0.5)) for _ in range(self.shape)]
        self.bias = Bias(random.uniform(-0.5, 0.5))
        output = IOParameter(0.0)
        output.source = self
        self.output = output

        self.accumulated_grad = 0.0
        self.stepped = False
        self.received_children = 0
        self.delta = Value(0.0)

    def forward(self, inputs: List[Union[IOParameter, InputVector]]) -> IOParameter:
        assert len(inputs) == self.shape
        self.last_inputs = inputs
        self.clear_pass_state()
        return self.output.update(sigmoid(dot(inputs, self.weights) + self.bias.value))

    def backward(self, incoming_grad: float) -> None:
        self.accumulated_grad += incoming_grad
        self.received_children += 1

        if self.received_children == self.expected_children:
            self.received_children = 0
            delta = self.delta.update(self.accumulated_grad * d_sigmoid(self.output.value))

            for i, inp in enumerate(self.last_inputs):
                if hasattr(inp, "source"):
                    inp.source.backward(delta.value * self.weights[i].value)

            self.accumulated_grad = 0.0

    def step(self):
        if self.stepped:
            return
        
        self.bias.value -= self.learning_rate * self.delta.value

        for i, weight in enumerate(self.weights):
            local_grad = self.delta.value * self.last_inputs[i].value
            weight.value -= self.learning_rate * local_grad

        self.stepped = True

        for child in self.children:
            child.step()

    def clear_pass_state(self):
        self.accumulated_grad = 0.0
        self.stepped = False

