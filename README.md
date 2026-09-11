# MiniLLM

Build a tiny language model from first principles: tokenizer → embeddings → causal attention → transformer block → training → generation.

MiniLLM is an educational, NumPy-only project inspired by [TinyTorch](https://mlsysbook.ai/tinytorch/). It deliberately avoids PyTorch, TensorFlow, JAX, and automatic differentiation so every important operation is visible and testable.

## What you will build

```text
text
  ↓
character tokenizer
  ↓
token + position embeddings
  ↓
causal multi-head self-attention
  ↓
residual + LayerNorm + MLP
  ↓
next-token logits
  ↓
cross-entropy + manual backpropagation
  ↓
Adam updates
```

The model is small enough to train on a laptop and is intended to learn short patterns from a tiny text corpus. It is not a useful general-purpose LLM.

## Requirements

- Python 3.10+
- NumPy

Create the project virtual environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

With the environment activated, run the checks and example:

```bash
python -m pytest -q
python -m examples.train_tiny --steps 400
```

Generate text after training:

```bash
python -m examples.generate --checkpoint artifacts/minillm.npz --prompt "the "
```

On the next terminal session, activate the existing environment with
`source .venv/bin/activate` before running project commands. The `.gitignore`
keeps `.venv/`, caches, logs, and generated checkpoints out of version control.

## Learning path

1. Read `minillm/tokenizer.py`: text becomes integer token IDs.
2. Read `minillm/nn.py`: inspect softmax, GELU, LayerNorm, and Adam.
3. Read `minillm/model.py`: follow the forward pass and its explicit backward pass.
4. Run `python -m pytest -q`: the tests check shapes, masking, gradients, and learning.
5. Change the corpus, context length, or model size and observe the trade-offs.

For a complete explanation of the architecture, tensor shapes, equations, training loop, generation, checkpoints, tests, and extensions, see the [MiniLLM guide](docs/guide.md).

## Code style

The implementation favors small classes, descriptive names, type hints, and docstrings on every function and method. `ForwardCache` gives names to intermediate tensors used by backpropagation, while `Trainer` owns the optimizer and batch generator. Read the docstrings first; they describe the contract of each public operation before the implementation details.

## Project layout

```text
minillm/
  tokenizer.py   character-level vocabulary and encode/decode
  nn.py          NumPy primitives and optimizer
  model.py       causal transformer and manual gradients
  data.py        next-token training batches
  train.py       reproducible training loop and checkpoints
  generate.py    temperature/top-k inference
examples/
  train_tiny.py
  generate.py
tests/
```

## Design choices

- Character tokens make tokenization transparent and require no downloaded vocabulary.
- A single transformer block keeps the equations approachable.
- Manual gradients make the training mechanics explicit. `tests/test_gradients.py` compares selected derivatives against finite differences.
- NumPy arrays use batch-first shapes: `(batch, time, features)`.

## Inspired by TinyTorch

The structure follows the same educational idea as TinyTorch: build a narrow vertical slice, test each layer, then connect the pieces into a working system. TinyTorch’s broader curriculum covers a full tensor/autograd framework and many more architectures; MiniLLM focuses on the smallest complete language-model path.

## License

MIT. The reference TinyTorch material is linked for inspiration; this implementation is original and independent.
