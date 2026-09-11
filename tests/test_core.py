import numpy as np

from minillm import CharTokenizer, MiniGPT, ModelConfig
from minillm.nn import softmax


def test_tokenizer_round_trip_and_unknown():
    """The tokenizer should preserve known text and mark unknown characters."""
    tokenizer = CharTokenizer("ab ba")
    assert tokenizer.decode(tokenizer.encode("ab ba")) == "ab ba"
    assert tokenizer.encode("z")[0] == tokenizer.stoi[tokenizer.UNK]


def test_forward_shape_and_causal_mask():
    """The model should return correct shapes and hide future positions."""
    model = MiniGPT(ModelConfig(vocab_size=7, context_length=8, d_model=8, n_heads=2, d_ff=16), seed=0)
    x = np.array([[1, 2, 3, 4]])
    logits, cache = model.forward(x)
    assert logits.shape == (1, 4, 7)
    weights = cache[5]
    assert np.allclose(weights[0, :, 0, 1:], 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_softmax_is_probability_distribution():
    """Softmax outputs positive values that sum to one."""
    result = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.all(result > 0)
    assert np.allclose(result.sum(), 1.0)


def test_tiny_model_learns():
    """A tiny predictable corpus should produce a lower loss after training."""
    tokenizer = CharTokenizer("abababababababab")
    model = MiniGPT(ModelConfig(tokenizer.vocab_size, context_length=4, d_model=8, n_heads=2, d_ff=16), seed=1)
    from minillm.train import train
    losses = train(model, tokenizer.encode("abababababababab") * 8, steps=80, batch_size=8, log_every=0)
    assert losses[-1] < losses[0]
