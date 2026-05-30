from v4_1 import *

XOR = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
]

inp1 = InputNode()
inp2 = InputNode()

h1 = Neuron(
    shape=2,
    parents=[],
    children=[],
    expected_children=1,
    learning_rate=0.1,
)

h2 = Neuron(
    shape=2,
    parents=[],
    children=[],
    expected_children=1,
    learning_rate=0.1,
)

out = Neuron(
    shape=2,
    parents=[],
    children=[],
    expected_children=1,   # <-- loss is the conceptual child
    learning_rate=0.1,
)

h1.children.append(out)
h2.children.append(out)

for epoch in range(10000):

    total_loss = 0.0

    for (x1, x2), target in XOR:

        inp1.output.update(float(x1))
        inp2.output.update(float(x2))

        h1_out = h1.forward([
            inp1.output,
            inp2.output,
        ])

        h2_out = h2.forward([
            inp1.output,
            inp2.output,
        ])

        out_val = out.forward([
            h1_out,
            h2_out,
        ])

        loss = (target - out_val.value) ** 2
        total_loss += loss

        # d/dy (y - target)^2
        loss_grad = 2 * (out_val.value - target)

        out.backward(loss_grad)

        h1.step()
        h2.step()
        out.step()

    if epoch % 500 == 0:
        print(epoch, total_loss)

print("\nPredictions:")

for (x1, x2), target in XOR:

    inp1.output.update(float(x1))
    inp2.output.update(float(x2))

    h1_out = h1.forward([
        inp1.output,
        inp2.output,
    ])

    h2_out = h2.forward([
        inp1.output,
        inp2.output,
    ])

    out_val = out.forward([
        h1_out,
        h2_out,
    ])

    print(
        (x1, x2),
        "->",
        round(out_val.value, 4),
        "(target:", target, ")"
    )