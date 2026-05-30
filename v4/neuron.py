from v4_1 import *

import math

# ------------------
# build neuron
# ------------------

n = Neuron(
    shape=1,
    parents=[],
    children=[],
    expected_children=1,
    learning_rate=0.1,
)

# deterministic values

n.weights[0].value = 0.3
n.bias.value = 0.1

# fake input node

inp = InputNode()
inp.output.update(0.7)

target = 1.0

# ------------------
# forward
# ------------------

pred = n.forward([inp.output]).value
loss = (pred - target) ** 2

print("prediction =", pred)
print("loss =", loss)

# ------------------
# backprop gradient
# ------------------

loss_grad = 2 * (pred - target)

n.backward(loss_grad)

backprop_grad = (
    n.delta.value
    * inp.output.value
)

print("\nBACKPROP")
print("delta =", n.delta.value)
print("grad =", backprop_grad)

# ------------------
# numerical gradient
# ------------------

eps = 1e-6

original_weight = n.weights[0].value

# loss(w + eps)

n.weights[0].value = original_weight + eps

pred_plus = n.forward([inp.output]).value
loss_plus = (pred_plus - target) ** 2

# loss(w - eps)

n.weights[0].value = original_weight - eps

pred_minus = n.forward([inp.output]).value
loss_minus = (pred_minus - target) ** 2

# restore

n.weights[0].value = original_weight

numeric_grad = (
    loss_plus - loss_minus
) / (2 * eps)

print("\nNUMERICAL")
print("grad =", numeric_grad)

# ------------------
# compare
# ------------------

error = abs(
    backprop_grad - numeric_grad
)

relative_error = error / (
    abs(backprop_grad)
    + abs(numeric_grad)
    + 1e-12
)

print("\nCOMPARE")
print("abs error      =", error)
print("relative error =", relative_error)