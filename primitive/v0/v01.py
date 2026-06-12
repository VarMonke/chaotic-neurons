class Neuron:
    input_neurons: list["Neuron"] | "Neuron" | "str"
    output_neuron: "Neuron"


    def __init__(self, inputs: list["Neuron"]):
        self.input_neurons = inputs


    def black_box_operations(self, p1, p2, p3, pn):
        # go magic
        ...