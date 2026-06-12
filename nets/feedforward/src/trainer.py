import random
import numpy as np


class Trainer:

    def __init__(
        self,
        network,
        loss,
        lr=0.01,
        epochs=1000,
        shuffle=True,
        verbose=True,
    ):

        self.network = network
        self.loss = loss

        self.lr = lr
        self.epochs = epochs

        self.shuffle = shuffle
        self.verbose = verbose

        self.history = []

    def train(self, dataset):

        """
        dataset format:

        [
            ([inputs...], [targets...]),
            ...
        ]
        """

        for epoch in range(self.epochs):

            if self.shuffle:
                random.shuffle(dataset)

            epoch_loss = 0.0

            for x, y in dataset:

                pred = self.network.predict(x)

                target = np.array(y, dtype=float)

                loss = self.loss.forward(pred, target)

                grad = self.loss.grad(pred, target)

                self.network.backward(grad)

                self.network.step(self.lr)

                self.network.flush()

                epoch_loss += loss

            epoch_loss /= len(dataset)

            self.history.append(epoch_loss)

            if self.verbose:

                if epoch % max(1, self.epochs // 20) == 0:

                    print(
                        f"[epoch {epoch+1}/{self.epochs}] "
                        f"loss={epoch_loss:.6f}"
                    )

    def evaluate(self, dataset):

        total_loss = 0.0

        for x, y in dataset:

            pred = self.network.predict(x)

            target = np.array(y, dtype=float)

            loss = self.loss.forward(pred, target)

            total_loss += loss

        return total_loss / len(dataset)

    def predict(self, x):

        return self.network.predict(x)