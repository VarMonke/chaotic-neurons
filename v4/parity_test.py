from v4_1 import *


from itertools import product
import random


# =====================================================
# DATASET GENERATION
# =====================================================

from itertools import product
import random

def make_parity_dataset(max_bits):

    dataset = []

    for bit_count in range(1, max_bits + 1):

        for sample in product(
            [0, 1],
            repeat=bit_count,
        ):

            padded = (
                list(sample)
                + [0] * (max_bits - bit_count)
            )

            parity = sum(sample) % 2

            dataset.append(
                (
                    tuple(float(x) for x in padded),
                    float(parity),
                )
            )

    random.shuffle(dataset)

    return dataset


# =====================================================
# NETWORK WRAPPER
# =====================================================

class Network:

    def __init__(
        self,
        bits: int,
        widths: list[int],
        learning_rates: list[float] | None = None,
    ):

        self.bits = bits

        self.input_nodes = [
            InputNode()
            for _ in range(bits)
        ]

        if learning_rates is None:
            learning_rates = [0.1] * (len(widths) + 1)

        self.layers = []

        previous_width = bits

        for width, lr in zip(widths, learning_rates):

            layer = [
                Neuron(
                    shape=previous_width,
                    parents=[],
                    children=[],
                    expected_children=1,
                    learning_rate=lr,
                )
                for _ in range(width)
            ]

            self.layers.append(layer)

            previous_width = width

        self.output = Neuron(
            shape=previous_width,
            parents=[],
            children=[],
            expected_children=1,
            learning_rate=learning_rates[-1],
        )

        self._connect()

    # -------------------------------------------------

    def _connect(self):

        for current_layer, next_layer in zip(
            self.layers[:-1],
            self.layers[1:],
        ):

            for neuron in current_layer:
                neuron.children = next_layer
                neuron.expected_children = len(next_layer)

        if self.layers:

            for neuron in self.layers[-1]:
                neuron.children = [self.output]
                neuron.expected_children = 1

    # -------------------------------------------------

    def forward(self, inputs):

        for node, value in zip(
            self.input_nodes,
            inputs,
        ):
            node.output.update(float(value))

        activations = []

        current = [
            node.output
            for node in self.input_nodes
        ]

        for layer in self.layers:

            current = [
                neuron.forward(current)
                for neuron in layer
            ]

            activations.append(current)

        prediction = self.output.forward(current)

        return prediction, activations

    # -------------------------------------------------

    def predict(self, inputs):

        prediction, _ = self.forward(inputs)

        return prediction.value

    # -------------------------------------------------

    def train(self, dataset, epochs):

        for epoch in range(epochs):

            random.shuffle(dataset)

            total_loss = 0.0

            for inputs, target in dataset:

                prediction, _ = self.forward(inputs)

                loss = (
                    prediction.value - target
                ) ** 2

                total_loss += loss

                loss_grad = (
                    2
                    * (prediction.value - target)
                )

                self.output.backward(loss_grad)

                for layer in self.layers:
                    for neuron in layer:
                        neuron.step()

                self.output.step()

            if epoch % 500 == 0:
                print(
                    f"epoch={epoch}"
                    f" loss={total_loss:.6f}"
                )

    # -------------------------------------------------

    def evaluate(self, dataset):

        correct = 0

        print("\nPREDICTIONS\n")

        for inputs, target in dataset:

            prediction = self.predict(inputs)

            pred = int(prediction > 0.5)

            if pred == target:
                correct += 1

            print(
                inputs,
                "->",
                round(prediction, 4),
                "pred:",
                pred,
                "target:",
                int(target),
            )

        print(
            f"\nAccuracy: "
            f"{correct}/{len(dataset)} "
            f"({100*correct/len(dataset):.2f}%)"
        )

    # -------------------------------------------------

    def inspect(self, inputs):

        prediction, activations = (
            self.forward(inputs)
        )

        print("\nINPUT:", inputs)

        for layer_idx, layer in enumerate(
            activations
        ):

            print(
                f"L{layer_idx}:",
                [
                    round(x.value, 4)
                    for x in layer
                ]
            )

        print(
            "OUT:",
            round(prediction.value, 4)
        )

    # -------------------------------------------------

    def inspect_all_inputs(self):

        print(
            "\n========================"
        )
        print("INSPECTION")
        print(
            "========================"
        )

        for i in range(2 ** self.bits):

            bits = tuple(
                int(bit)
                for bit in format(
                    i,
                    f"0{self.bits}b"
                )
            )

            self.inspect(bits)

    # -------------------------------------------------

    def activation_stats(self, dataset):

        storage = [
            [
                []
                for _ in layer
            ]
            for layer in self.layers
        ]

        for inputs, _ in dataset:

            _, activations = self.forward(
                inputs
            )

            for layer_idx, layer in enumerate(
                activations
            ):

                for neuron_idx, neuron in enumerate(
                    layer
                ):

                    storage[layer_idx][neuron_idx].append(
                        neuron.value
                    )

        print(
            "\n========================"
        )
        print("ACTIVATION STATS")
        print(
            "========================"
        )

        for layer_idx, layer_stats in enumerate(
            storage
        ):

            print(
                f"\nLayer {layer_idx}"
            )

            for neuron_idx, values in enumerate(
                layer_stats
            ):

                print(
                    neuron_idx,
                    "mean=",
                    round(
                        sum(values)
                        / len(values),
                        4,
                    ),
                    "min=",
                    round(min(values), 4),
                    "max=",
                    round(max(values), 4),
                )

    # -------------------------------------------------

    def neuron_signature(
        self,
        neuron,
        dataset,
    ):

        for inputs, target in sorted(
            dataset,
            key=lambda x: sum(x[0]),
        ):

            self.forward(inputs)

            print(
                inputs,
                "ones=",
                int(sum(inputs)),
                "target=",
                int(target),
                "activation=",
                round(
                    neuron.output.value,
                    4,
                ),
            )

    # -------------------------------------------------

    def inspect_neurons(
        self,
        dataset,
    ):

        print(
            "\n========================"
        )
        print("NEURON SIGNATURES")
        print(
            "========================"
        )

        for layer_idx, layer in enumerate(
            self.layers
        ):

            print(
                f"\n######## "
                f"LAYER {layer_idx}"
                f" ########"
            )

            for neuron_idx, neuron in enumerate(
                layer
            ):

                print(
                    f"\n----- "
                    f"[{layer_idx}]"
                    f"[{neuron_idx}] "
                    f"-----"
                )

                self.neuron_signature(
                    neuron,
                    dataset,
                )


# =====================================================
# ONE-LINE EXPERIMENT
# =====================================================

def run_experiment(
    bits=4,
    widths=[8, 4, 2],
    epochs=30000,
    task="parity",
):

    dataset = make_parity_dataset(
        bits,
    )

    network = Network(
        bits=bits,
        widths=widths,
    )

    network.train(
        dataset,
        epochs,
    )

    network.evaluate(dataset)

    network.inspect_all_inputs()

    network.activation_stats(
        dataset
    )

    network.inspect_neurons(
        dataset
    )

    return network


# =====================================================
# EXAMPLES
# =====================================================

net = run_experiment(
    bits=5,
    widths=[8, 8],
    epochs=30000,
)

# net = run_experiment(
#     bits=8,
#     widths=[32, 16, 8],
#     epochs=50000,
# )

# net = run_experiment(
#     bits=12,
#     widths=[64, 32, 16],
#     epochs=100000,
# )