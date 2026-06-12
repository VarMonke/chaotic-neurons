import numpy as np


class Loss:
    def forward(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        pass

    def grad(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        pass


class MSE(Loss):
    def forward(self, pred, target):
        return np.mean((pred - target) ** 2)

    def grad(self, pred, target):
        return 2 * (pred - target) / pred.size


class MAE(Loss):
    def forward(self, pred, target):
        return np.mean(np.abs(pred - target))

    def grad(self, pred, target):
        return np.sign(pred - target) / pred.size


class BinaryCrossEntropy(Loss):
    EPS = 1e-8

    def forward(self, pred, target):

        pred = np.clip(pred, self.EPS, 1 - self.EPS)

        return -np.mean(
            target * np.log(pred)
            + (1 - target) * np.log(1 - pred)
        )

    def grad(self, pred, target):

        pred = np.clip(pred, self.EPS, 1 - self.EPS)

        return (
            (pred - target)
            / (pred * (1 - pred) * pred.size)
        )


class Huber(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, pred, target):

        err = pred - target

        abs_err = np.abs(err)

        quadratic = np.minimum(abs_err, self.delta)

        linear = abs_err - quadratic

        return np.mean(
            0.5 * quadratic**2
            + self.delta * linear
        )

    def grad(self, pred, target):

        err = pred - target

        return np.where(
            np.abs(err) <= self.delta,
            err,
            self.delta * np.sign(err)
        ) / pred.size