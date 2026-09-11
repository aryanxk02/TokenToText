import numpy as np

from minillm import MiniGPT, ModelConfig


def test_selected_gradient_matches_finite_difference():
    """The analytic MLP gradient should agree with a numerical estimate."""
    model = MiniGPT(ModelConfig(vocab_size=5, context_length=3, d_model=4, n_heads=2, d_ff=6), seed=3)
    x = np.array([[1, 2, 3]])
    y = np.array([[2, 3, 4]])
    _, grads = model.loss_and_backward(x, y)
    name, index, eps = "w2", (0, 0), 1e-5
    original = model.params[name][index]
    model.params[name][index] = original + eps
    plus, _ = model.loss_and_backward(x, y)
    model.params[name][index] = original - eps
    minus, _ = model.loss_and_backward(x, y)
    model.params[name][index] = original
    numerical = (plus - minus) / (2 * eps)
    assert np.isclose(grads[name][index], numerical, rtol=2e-3, atol=2e-4)
