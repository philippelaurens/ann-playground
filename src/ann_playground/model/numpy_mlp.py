"""
Module : numpy_mlp.py
Description : Object-Oriented multi-class Multi-Layer Perceptron (MLP) implementation
              from scratch using pure NumPy with backpropagation and He/Xavier initializations.

Classes :
    - DenseLayer : Fully-connected neural network layer with forward and backward passes.
    - NumpyMLP : Sequential multi-class Multi-Layer Perceptron model.
"""

from typing import List, Tuple

import numpy as np


class DenseLayer:
    """Fully-connected dense layer with forward and backward gradient propagation."""

    def __init__(self, n_in: int, n_out: int, activation: str = "relu") -> None:
        """Initialize weights and biases.

        Args:
            n_in: Number of input features.
            n_out: Number of output neurons.
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'softmax').
        """
        self.activation: str = activation
        limit: float = (
            np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        )
        self.weights: np.ndarray = np.random.randn(n_in, n_out) * limit
        self.biases: np.ndarray = np.zeros((1, n_out))
        self.inputs: np.ndarray = np.empty((0,))
        self.z: np.ndarray = np.empty((0,))
        self.output: np.ndarray = np.empty((0,))
        self.dweights: np.ndarray = np.empty((0,))
        self.dbiases: np.ndarray = np.empty((0,))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the forward pass activation.

        Args:
            inputs: Input matrix of shape (n_samples, n_in).

        Returns:
            Activated output matrix of shape (n_samples, n_out).
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases

        if self.activation == "relu":
            self.output = np.maximum(0.0, self.z)
        elif self.activation == "sigmoid":
            self.output = 1.0 / (1.0 + np.exp(-np.clip(self.z, -20.0, 20.0)))
        elif self.activation == "tanh":
            self.output = np.tanh(self.z)
        elif self.activation == "softmax":
            exp_shifted = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        else:
            self.output = self.z
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """Compute layer gradients and propagate error backward.

        Args:
            dvalues: Upstream derivative matrix.

        Returns:
            Gradients with respect to layer inputs.
        """
        if self.activation == "relu":
            dz = dvalues * (self.z > 0)
        elif self.activation == "sigmoid":
            dz = dvalues * (self.output * (1.0 - self.output))
        elif self.activation == "tanh":
            dz = dvalues * (1.0 - self.output**2)
        elif self.activation == "softmax":
            dz = dvalues
        else:
            dz = dvalues

        self.dweights = np.dot(self.inputs.T, dz) / self.inputs.shape[0]
        self.dbiases = np.sum(dz, axis=0, keepdims=True) / self.inputs.shape[0]
        return np.dot(dz, self.weights.T)


class NumpyMLP:
    """Multi-Layer Perceptron implemented from scratch in pure NumPy."""

    def __init__(self, layer_sizes: List[int], activation: str = "relu") -> None:
        """Assemble sequential dense layers ending with a softmax classifier.

        Args:
            layer_sizes: List of neuron counts from input layer to output layer.
            activation: Activation function for hidden layers.
        """
        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(
                DenseLayer(layer_sizes[i], layer_sizes[i + 1], activation)
            )
        self.layers.append(
            DenseLayer(layer_sizes[-2], layer_sizes[-1], activation="softmax")
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagate input across all layers.

        Args:
            x: Input array of shape (n_samples, n_features).

        Returns:
            Probability distribution matrix of shape (n_samples, n_classes).
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true_one_hot: np.ndarray, y_pred: np.ndarray) -> None:
        """Backpropagate categorical cross-entropy gradient.

        Args:
            y_true_one_hot: One-hot encoded ground truth array.
            y_pred: Softmax probabilities output by forward pass.
        """
        dvalues = y_pred - y_true_one_hot
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def train_step(
        self, x: np.ndarray, y_true: np.ndarray, lr: float
    ) -> Tuple[float, float]:
        """Execute one gradient descent step over a given batch.

        Args:
            x: Batch feature array.
            y_true: One-hot encoded batch labels.
            lr: Learning rate.

        Returns:
            Tuple of (loss_value, accuracy_score).
        """
        y_pred = self.forward(x)
        epsilon = 1e-12
        loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
        acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))

        self.backward(y_true, y_pred)

        for layer in self.layers:
            layer.weights -= lr * layer.dweights
            layer.biases -= lr * layer.dbiases

        return float(loss), acc

    def evaluate(self, x: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        """Compute evaluation metrics on a validation or test set.

        Args:
            x: Feature array.
            y_true: One-hot encoded ground truth array.

        Returns:
            Tuple of (loss_value, accuracy_score).
        """
        y_pred = self.forward(x)
        epsilon = 1e-12
        loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
        acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))
        return float(loss), acc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Compute categorical predictions.

        Args:
            x: Feature array.

        Returns:
            Class indices with the highest probability.
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
