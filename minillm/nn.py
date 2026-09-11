"""Small neural-network building blocks implemented with NumPy.

The functions in this module are intentionally explicit. They are the small
mathematical operations used by :class:`minillm.model.MiniGPT`.
"""

from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert scores into probabilities along ``axis`` safely."""
    shifted_values = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted_values)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def gelu(values: np.ndarray) -> np.ndarray:
    """Apply the tanh approximation of the Gaussian Error Linear Unit."""
    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (values + 0.044715 * values**3)
    return 0.5 * values * (1.0 + np.tanh(inner))


def gelu_gradient(values: np.ndarray) -> np.ndarray:
    """Return the derivative of :func:`gelu` for ``values``."""
    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (values + 0.044715 * values**3)
    tanh_inner = np.tanh(inner)
    inner_gradient = coefficient * (1.0 + 3.0 * 0.044715 * values**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * values * (1.0 - tanh_inner**2) * inner_gradient


def layer_norm(
    values: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Normalize each vector in ``values`` over its final dimension."""
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    inverse_standard_deviation = 1.0 / np.sqrt(variance + epsilon)
    normalized = (values - mean) * inverse_standard_deviation
    output = normalized * scale + bias
    cache = (normalized, inverse_standard_deviation, scale)
    return output, cache


def layer_norm_backward(
    gradient: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through :func:`layer_norm`.

    Returns gradients for the input, scale, and bias, in that order.
    """
    normalized, inverse_standard_deviation, scale = cache
    feature_count = gradient.shape[-1]
    normalized_gradient = gradient * scale
    variance_gradient = np.sum(
        normalized_gradient * normalized * -0.5 * inverse_standard_deviation,
        axis=-1,
        keepdims=True,
    )
    mean_gradient = np.sum(
        normalized_gradient * -inverse_standard_deviation,
        axis=-1,
        keepdims=True,
    )
    mean_gradient += variance_gradient * np.mean(
        -2.0 * normalized / inverse_standard_deviation,
        axis=-1,
        keepdims=True,
    )
    input_gradient = (
        normalized_gradient * inverse_standard_deviation
        + variance_gradient * 2.0 * normalized / (feature_count * inverse_standard_deviation)
        + mean_gradient / feature_count
    )
    reduction_axes = tuple(range(gradient.ndim - 1))
    scale_gradient = np.sum(gradient * normalized, axis=reduction_axes)
    bias_gradient = np.sum(gradient, axis=reduction_axes)
    return input_gradient, scale_gradient, bias_gradient


class Adam:
    """Adam optimizer for a dictionary of NumPy parameter arrays."""

    def __init__(
        self,
        parameters: dict[str, np.ndarray],
        learning_rate: float = 3e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Create an optimizer that updates ``parameters`` in place."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < beta1 < 1 or not 0 < beta2 < 1:
            raise ValueError("beta1 and beta2 must be between 0 and 1")
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_count = 0

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        """Apply one clipped, bias-corrected Adam update to every parameter."""
        missing = set(self.parameters) - set(gradients)
        if missing:
            raise KeyError(f"missing gradients for: {sorted(missing)}")
        self.step_count += 1
        for name, parameter in self.parameters.items():
            clipped_gradient = np.clip(gradients[name], -1.0, 1.0)
            self.first_moment[name] = self.beta1 * self.first_moment[name] + (1.0 - self.beta1) * clipped_gradient
            self.second_moment[name] = self.beta2 * self.second_moment[name] + (1.0 - self.beta2) * clipped_gradient**2
            first_moment_hat = self.first_moment[name] / (1.0 - self.beta1**self.step_count)
            second_moment_hat = self.second_moment[name] / (1.0 - self.beta2**self.step_count)
            parameter -= self.learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + self.epsilon)
