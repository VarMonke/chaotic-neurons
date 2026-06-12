import numpy as np

from layer import InputLayer, Layer


class FFNetwork:

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        hidden_activation,
        output_activation,
    ):

        self.input_layer = InputLayer(input_size)
        self.layers: list[Layer] = []
        prev = self.input_layer

        for width in hidden_sizes:
            layer = Layer(
                width=width,
                activation=hidden_activation,
                prev_layer=prev,
                next_layer=None,
            )

            if hasattr(prev, "next_layer"):
                prev.next_layer = layer

            self.layers.append(layer)
            prev = layer

        out = Layer(
            width=output_size,
            activation=output_activation,
            prev_layer=prev,
            next_layer=None,
        )

        if hasattr(prev, "next_layer"):
            prev.next_layer = out

        self.layers.append(out)
        self.output_layer = out

    def predict(self, x):

        self.input_layer.outputs = np.array(x, dtype=float)
        if self.layers:
            self.layers[0].forward()
        return self.output_layer.outputs

    def backward(self, grads):
        self.output_layer.backward(np.array(grads, dtype=float))

    def step(self, lr=0.01):
        if self.layers:
            self.layers[0].step(lr)

    def flush(self):
        if self.layers:
            self.layers[0].flush()

    def train_step(
        self,
        x,
        y,
        criterion,
        lr=0.01,
    ):

        pred = self.predict(x)
        target = np.array(y, dtype=float)
        loss = criterion.forward(pred, target)
        grad = criterion.grad(pred, target)
        self.backward(grad)
        self.step(lr)
        self.flush()
        return loss, pred