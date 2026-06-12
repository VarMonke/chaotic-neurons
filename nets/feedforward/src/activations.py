import numpy as np

class Activation:
    def forward(self, v: np.ndarray) -> np.ndarray:
        pass

    def grad(self, v: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(Activation):
    def forward(self, v):
        return 1 / (1 + np.exp(-v))

    def grad(self, v):
        return v * (1 - v)


class ReLU(Activation):
    def forward(self, v):
        return np.maximum(0, v)

    def grad(self, v):
        return (v > 0).astype(float)


class Tanh(Activation):
    def forward(self, v):
        return np.tanh(v)

    def grad(self, v):
        return 1 - v * v