import math
import random

from primitive.v5.v52 import *


inp, hidden_layers, out = make_network(
    bits=2,
    inp_size=2,
    hidden_widths=[2],
    activations=[Sigmoid()],
    out_activation=Sigmoid()
)

ds = gen_ds(3)

lr = 0.1

for epoch in range(100):

    random.shuffle(ds)

    for bits, target in ds:

        state = 0.0
        for bit in bits:

            inp.outputs = [
                float(bit),
                state
            ]

            state = predict(inp, out, hidden_layers)
            print(
                "bits =", bits,
                "pred =", round(state, 3),
                "target =", target
            )
        pred = state

        loss_grad = pred - target

        out.backward([loss_grad])

        hidden_layers[0].step(lr)

        hidden_layers[0].flush()