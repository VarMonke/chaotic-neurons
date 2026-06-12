"""
Standalone diagnostics for the tiny neural network from the prompt.

This file does not modify the network core. Save your core code in a Python file,
then run this helper against it:

    python /home/ubuntu/nn_diagnostics.py ./your_network.py --bits 3 --hidden 8 8 --epochs 2000 --lr 0.2

What it gives you:
- static/smoke diagnostics for dataset generation, activations, make_network, predict
- an external network builder that wires Layer.next_layer without editing your code
- training metrics, truth-table predictions, gradient checks, and feature sensitivity
- simple summaries that help interpret what the network learned
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import random
import sys
from itertools import product
from types import ModuleType


def load_module(path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    try:
        spec.loader.exec_module(module)
        return module
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise

    shimmed = ModuleType(path.stem)
    sys.modules[path.stem] = shimmed
    source = path.read_text()
    exec(compile("from __future__ import annotations\n" + source, str(path), "exec"), shimmed.__dict__)
    return shimmed


def call_check(label: str, fn):
    try:
        value = fn()
        return True, label, value
    except Exception as exc:
        return False, label, f"{type(exc).__name__}: {exc}"


def fmt(value: float) -> str:
    return f"{value: .6f}"


def make_dataset(bits: int) -> list[tuple[tuple[float, ...], float]]:
    return [
        (tuple(float(bit) for bit in sample), float(sum(sample) % 2))
        for sample in product([0, 1], repeat=bits)
    ]


def get_activation(module: ModuleType, name: str):
    key = name.lower()
    if key == "sigmoid":
        return module.Sigmoid()
    if key == "relu":
        return module.ReLU()
    if key == "tanh":
        return module.tanh()
    raise ValueError(f"Unknown activation: {name}")


def build_network(
    module: ModuleType,
    input_size: int,
    hidden_widths: list[int],
    hidden_activation_name: str,
    output_activation_name: str,
):
    input_layer = module.InputLayer(input_size)
    previous = input_layer
    hidden_layers = []

    for width in hidden_widths:
        activation = get_activation(module, hidden_activation_name)
        layer = module.Layer(
            width=width,
            activation=activation,
            prev_layer=previous,
            next_layer=None,
        )
        if hidden_layers:
            hidden_layers[-1].next_layer = layer
        hidden_layers.append(layer)
        previous = layer

    output_layer = module.Layer(
        width=1,
        activation=get_activation(module, output_activation_name),
        prev_layer=previous,
        next_layer=None,
    )

    if hidden_layers:
        hidden_layers[-1].next_layer = output_layer

    return input_layer, hidden_layers, output_layer


def forward(input_layer, hidden_layers, output_layer, sample: tuple[float, ...]) -> float:
    input_layer.outputs = list(sample)
    if hidden_layers:
        hidden_layers[0].forward()
    else:
        output_layer.forward()
    return float(output_layer.outputs[0])


def trainable_start(hidden_layers, output_layer):
    if hidden_layers:
        return hidden_layers[0]
    return output_layer


def loss_and_output(input_layer, hidden_layers, output_layer, sample, target: float) -> tuple[float, float]:
    output = forward(input_layer, hidden_layers, output_layer, sample)
    loss = 0.5 * (output - target) * (output - target)
    return loss, output


def train_epoch(input_layer, hidden_layers, output_layer, dataset, lr: float) -> float:
    random.shuffle(dataset)
    total_loss = 0.0
    start = trainable_start(hidden_layers, output_layer)

    for sample, target in dataset:
        start.flush()
        loss, output = loss_and_output(input_layer, hidden_layers, output_layer, sample, target)
        total_loss += loss
        output_layer.backward([output - target])
        start.step(lr)

    return total_loss / len(dataset)


def accuracy(input_layer, hidden_layers, output_layer, dataset) -> float:
    correct = 0
    for sample, target in dataset:
        output = forward(input_layer, hidden_layers, output_layer, sample)
        prediction = 1.0 if output >= 0.5 else 0.0
        correct += int(prediction == target)
    return correct / len(dataset)


def truth_table(input_layer, hidden_layers, output_layer, dataset) -> list[tuple[tuple[float, ...], float, float, float]]:
    rows = []
    for sample, target in sorted(dataset):
        output = forward(input_layer, hidden_layers, output_layer, sample)
        prediction = 1.0 if output >= 0.5 else 0.0
        rows.append((sample, target, output, prediction))
    return rows


def layer_chain(hidden_layers, output_layer):
    return hidden_layers + [output_layer]


def weight_summary(hidden_layers, output_layer) -> list[str]:
    lines = []
    for index, layer in enumerate(layer_chain(hidden_layers, output_layer)):
        flat_weights = [weight for row in layer.weights for weight in row]
        avg_abs_weight = sum(abs(weight) for weight in flat_weights) / len(flat_weights)
        avg_abs_bias = sum(abs(bias) for bias in layer.biases) / len(layer.biases)
        lines.append(
            f"layer {index}: width={layer.width}, avg|w|={fmt(avg_abs_weight)}, avg|b|={fmt(avg_abs_bias)}"
        )
    return lines


def neuron_weight_dump(hidden_layers, output_layer) -> list[str]:
    lines = []
    for layer_index, layer in enumerate(layer_chain(hidden_layers, output_layer)):
        layer_name = "output" if layer.next_layer is None else f"hidden {layer_index}"
        lines.append(f"{layer_name} layer")
        for neuron_index, weights in enumerate(layer.weights):
            parts = [f"x{input_index}:{fmt(float(weight))}" for input_index, weight in enumerate(weights)]
            bias = fmt(float(layer.biases[neuron_index]))
            lines.append(f"  neuron {neuron_index:02d}  bias:{bias}  weights={{" + ", ".join(parts) + "}")
    return lines


def gradient_norm_summary(hidden_layers, output_layer) -> list[str]:
    lines = []
    for index, layer in enumerate(layer_chain(hidden_layers, output_layer)):
        grad_sq = 0.0
        for row in layer.weight_grads:
            for grad in row:
                grad_sq += grad * grad
        for grad in layer.bias_grads:
            grad_sq += grad * grad
        lines.append(f"layer {index}: grad_norm={fmt(math.sqrt(grad_sq))}")
    return lines


def finite_difference_check(input_layer, hidden_layers, output_layer, sample, target: float) -> tuple[float, float, float]:
    start = trainable_start(hidden_layers, output_layer)
    start.flush()
    loss_and_output(input_layer, hidden_layers, output_layer, sample, target)
    output_layer.backward([output_layer.outputs[0] - target])

    layer = layer_chain(hidden_layers, output_layer)[-1]
    analytic_grad = float(layer.weight_grads[0][0])
    original_weight = float(layer.weights[0][0])
    epsilon = 1e-4

    layer.weights[0][0] = original_weight + epsilon
    loss_plus, _ = loss_and_output(input_layer, hidden_layers, output_layer, sample, target)

    layer.weights[0][0] = original_weight - epsilon
    loss_minus, _ = loss_and_output(input_layer, hidden_layers, output_layer, sample, target)

    layer.weights[0][0] = original_weight
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    return analytic_grad, numerical_grad, abs(analytic_grad - numerical_grad)


def feature_sensitivity(input_layer, hidden_layers, output_layer, dataset) -> list[tuple[int, float, float]]:
    bit_count = len(dataset[0][0])
    rows = []
    for bit_index in range(bit_count):
        signed_total = 0.0
        absolute_total = 0.0
        for sample, _ in dataset:
            flipped = list(sample)
            flipped[bit_index] = 1.0 - flipped[bit_index]
            base = forward(input_layer, hidden_layers, output_layer, sample)
            changed = forward(input_layer, hidden_layers, output_layer, tuple(flipped))
            delta = changed - base
            signed_total += delta
            absolute_total += abs(delta)
        rows.append((bit_index, signed_total / len(dataset), absolute_total / len(dataset)))
    return rows


def activation_snapshot(input_layer, hidden_layers, output_layer, sample: tuple[float, ...]) -> list[str]:
    forward(input_layer, hidden_layers, output_layer, sample)
    lines = [f"input={sample} output={fmt(float(output_layer.outputs[0]))}"]
    for index, layer in enumerate(hidden_layers):
        values = ", ".join(fmt(float(value)) for value in layer.outputs[:8])
        suffix = "" if len(layer.outputs) <= 8 else ", ..."
        lines.append(f"hidden {index}: [{values}{suffix}]")
    return lines

def hidden_activation_table(input_layer, hidden_layers, output_layer, bits: int) -> list[str]:
    samples = [tuple(float(bit) for bit in sample) for sample in product([0, 1], repeat=bits)]
 
    headers = ["input", "target"]
    for layer_index, layer in enumerate(hidden_layers):
        for neuron_index in range(layer.width):
            headers.append(f"h{layer_index}.{neuron_index}")
    headers.append("output")
    headers.append("pred")
 
    rows = []
    for sample in samples:
        target = float(sum(sample) % 2)
        output = forward(input_layer, hidden_layers, output_layer, sample)
        pred = 1.0 if output >= 0.5 else 0.0
 
        row = [
            "".join(str(int(bit)) for bit in sample),
            str(int(target)),
        ]
        for layer in hidden_layers:
            row.extend(fmt(float(value)) for value in layer.outputs)
        row.extend([fmt(output), str(int(pred))])
        rows.append(row)
 
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
 
    def render(row):
        return " | ".join(cell.rjust(width) for cell, width in zip(row, widths))
 
    lines = [render(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    return lines


def static_diagnostics(module: ModuleType) -> list[tuple[bool, str, object]]:
    checks = [
        call_check("dot([1, 2], [3, 4]) == 11", lambda: module.dot([1, 2], [3, 4])),
        call_check("Sigmoid.forward(0)", lambda: module.Sigmoid().forward(0)),
        call_check("Sigmoid.grad(0.5)", lambda: module.Sigmoid().grad(0.5)),
        call_check("ReLU.forward(-1)", lambda: module.ReLU().forward(-1)),
        call_check("tanh.forward(0)", lambda: module.tanh().forward(0)),
        call_check("gen_ds(3) size", lambda: len(module.gen_ds(3))),
        call_check(
            "make_network smoke",
            lambda: module.make_network(2, 2, [4], [module.Sigmoid()], module.Sigmoid()),
        ),
        call_check(
            "predict smoke",
            lambda: (
                lambda built: module.predict(built[0], built[2], built[1])
            )(build_network(module, 2, [4], "sigmoid", "sigmoid")),
        ),
    ]
    return checks


def print_static_diagnostics(module: ModuleType) -> None:
    print("\nSTATIC / SMOKE DIAGNOSTICS")
    print("-" * 80)
    for ok, label, value in static_diagnostics(module):
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} {label}: {value}")


def print_training_run(args, module: ModuleType) -> None:
    random.seed(args.seed)
    dataset = make_dataset(args.bits)
    input_layer, hidden_layers, output_layer = build_network(
        module=module,
        input_size=args.bits,
        hidden_widths=args.hidden,
        hidden_activation_name=args.hidden_activation,
        output_activation_name=args.output_activation,
    )

    print("\nNETWORK SETUP")
    print("-" * 80)
    print(f"bits={args.bits} hidden={args.hidden} lr={args.lr} epochs={args.epochs}")
    print(f"hidden_activation={args.hidden_activation} output_activation={args.output_activation}")
    print(f"dataset_size={len(dataset)}")

    print("\nBEFORE TRAINING")
    print("-" * 80)
    print(f"accuracy={accuracy(input_layer, hidden_layers, output_layer, dataset):.2%}")
    for line in weight_summary(hidden_layers, output_layer):
        print(line)

    report_every = max(1, args.epochs // 10)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(input_layer, hidden_layers, output_layer, dataset, args.lr)
        if epoch == 1 or epoch % report_every == 0 or epoch == args.epochs:
            acc = accuracy(input_layer, hidden_layers, output_layer, dataset)
            print(f"epoch={epoch:5d} loss={loss:.6f} accuracy={acc:.2%}")

    sample, target = dataset[0]
    analytic, numerical, diff = finite_difference_check(input_layer, hidden_layers, output_layer, sample, target)

    print("\nAFTER TRAINING")
    print("-" * 80)
    print(f"accuracy={accuracy(input_layer, hidden_layers, output_layer, dataset):.2%}")
    print(f"finite_diff output weight[0][0]: analytic={fmt(analytic)} numerical={fmt(numerical)} abs_diff={fmt(diff)}")

    print("\nGRADIENT NORMS FROM LAST CHECK")
    print("-" * 80)
    for line in gradient_norm_summary(hidden_layers, output_layer):
        print(line)

    print("\nTRUTH TABLE")
    print("-" * 80)
    for sample, target, output, prediction in truth_table(input_layer, hidden_layers, output_layer, dataset):
        status = "ok" if prediction == target else "bad"
        print(f"x={tuple(int(bit) for bit in sample)} target={int(target)} out={output:.4f} pred={int(prediction)} {status}")

    print("\nFEATURE FLIP SENSITIVITY")
    print("-" * 80)
    print("High avg_abs_delta means flipping that input changes the output more.")
    for bit_index, signed_delta, absolute_delta in feature_sensitivity(input_layer, hidden_layers, output_layer, dataset):
        print(f"bit={bit_index} avg_signed_delta={fmt(signed_delta)} avg_abs_delta={fmt(absolute_delta)}")

    print("\nACTIVATION SNAPSHOT")
    print("-" * 80)
    for line in activation_snapshot(input_layer, hidden_layers, output_layer, tuple(1.0 for _ in range(args.bits))):
        print(line)

    print("\nHIDDEN ACTIVATIONS BY PARITY INPUT")
    print("-" * 80)
    for line in hidden_activation_table(input_layer, hidden_layers, output_layer, args.bits):
        print(line)
        
    print("\nWEIGHT SUMMARY")
    print("-" * 80)
    for line in weight_summary(hidden_layers, output_layer):
        print(line)

    print("\nPER-NEURON WEIGHTS")
    print("-" * 80)
    for line in neuron_weight_dump(hidden_layers, output_layer):
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostics against the tiny neural-network module.")
    parser.add_argument("module_path", type=pathlib.Path, help="Path to the Python file containing your core NN code.")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[8, 8])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-activation", choices=["sigmoid", "relu", "tanh"], default="sigmoid")
    parser.add_argument("--output-activation", choices=["sigmoid", "relu", "tanh"], default="sigmoid")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_module(args.module_path.resolve())
    print_static_diagnostics(module)
    if not args.skip_training:
        print_training_run(args, module)


if __name__ == "__main__":
    main()
