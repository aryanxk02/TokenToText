# MiniLLM: A From-Scratch Language Model

This guide explains the complete MiniLLM implementation in this repository. It is written as a companion to the code: each concept points to the file where it is implemented.

MiniLLM is an educational decoder-only transformer written with Python and NumPy. It is designed to make the complete path from raw text to generated text visible:

```text
raw text
  │
  ▼
character IDs
  │
  ▼
training windows: input tokens → next-token targets
  │
  ▼
token + position embeddings
  │
  ▼
causal self-attention
  │
  ▼
residual connection + LayerNorm + MLP
  │
  ▼
logits over the vocabulary
  │
  ▼
cross-entropy loss → manual gradients → Adam
  │
  ▼
next-token sampling
```

This is a teaching model, not a competitive language model. Its value is that every major operation can be inspected, modified, and tested.

## Code-reading principles

The code is intentionally written for a first-time reader:

- Classes own state: `CharTokenizer`, `BatchGenerator`, `Adam`, `Trainer`, and `MiniGPT`.
- Public methods use descriptive names and type hints.
- Every function and method has a docstring describing its job and inputs.
- Long calculations are split into named helper methods.
- `ForwardCache` replaces an anonymous tuple with named intermediate tensors.
- The short `train()` function remains as a convenience wrapper around the object-oriented `Trainer` API.

When reading a new file, start with its module docstring, then the class docstrings, then the public methods. Only after that should you trace the numerical details.

## 1. Installation and first run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.train_tiny --steps 400
python -m examples.generate \
  --checkpoint artifacts/minillm.npz \
  --prompt "the " \
  --tokens 120
```

The training example uses a small built-in corpus and writes a checkpoint to `artifacts/minillm.npz`.

For a quick smoke test:

```bash
python -m examples.train_tiny --steps 25 --checkpoint /tmp/minillm.npz
python -m examples.generate --checkpoint /tmp/minillm.npz --prompt "the " --tokens 40
```

The output will not be grammatical at first. The purpose of the first run is to verify that the complete pipeline works and that the loss decreases.

## 2. Repository map

```text
minillm/
  tokenizer.py   Character vocabulary, encode, decode, serialization
  data.py        Random next-token training windows
  nn.py          NumPy primitives and Adam
  model.py       Transformer forward pass, loss, backward pass, generation
  train.py       Reusable training loop and checkpoint writing
examples/
  train_tiny.py  Train on the built-in toy corpus
  generate.py    Load a checkpoint and sample text
tests/
  test_core.py   Tokenizer, shapes, mask, softmax, learning behavior
  test_gradients.py  Finite-difference gradient check
```

## 3. Tokenization

The tokenizer is intentionally character-level. Given:

```text
hello
```

it constructs a vocabulary containing two reserved tokens plus each distinct character:

```text
["<pad>", "<unk>", "e", "h", "l", "o"]
```

The mapping might therefore be:

```text
h → 3
e → 2
l → 4
o → 5
```

`CharTokenizer.encode()` converts a string into integer IDs. Unknown characters become `<unk>`. `decode()` converts IDs back to text.

The tokenizer lives in [minillm/tokenizer.py](../minillm/tokenizer.py).

### Why character tokens?

Character tokenization is not the most efficient choice for a real LLM, but it has three educational advantages:

1. There is no downloaded vocabulary.
2. Every input character is visible.
3. The vocabulary is small, so experiments run quickly.

The trade-off is that sequences are longer and the model must learn spelling and word boundaries one character at a time.

## 4. Next-token prediction

Language models are trained by shifting a sequence by one position.

For a token sequence:

```text
[the,  , m, o, d, e, l]
```

the training pair is:

```text
input:  [the,  , m, o, d, e]
target: [   , m, o, d, e, l]
```

At every position, the model predicts the token immediately to the right. `BatchGenerator` in `minillm/data.py` samples random windows of length `context_length` and creates these shifted pairs.

For a batch, the shapes are:

```text
input token IDs:  (batch, time)
target token IDs: (batch, time)
```

The target at position `t` is never allowed to influence the prediction at position `t`. The causal attention mask enforces this.

## 5. Model configuration

`ModelConfig` in [minillm/model.py](../minillm/model.py) controls the model:

```python
ModelConfig(
    vocab_size=tokenizer.vocab_size,
    context_length=32,
    d_model=32,
    n_heads=4,
    d_ff=64,
)
```

### Configuration fields

- `vocab_size`: number of possible token IDs.
- `context_length`: maximum number of tokens processed at once.
- `d_model`: width of each token representation.
- `n_heads`: number of independent attention heads.
- `d_ff`: hidden width of the feed-forward network.

`d_model` must be divisible by `n_heads`. Each head has width:

```text
head_dim = d_model / n_heads
```

The default model is deliberately small. Increasing these values increases capacity and computation.

## 6. Embeddings

The model begins with two learned lookup tables:

```text
token_embedding:    (vocab_size, d_model)
position_embedding: (context_length, d_model)
```

For input IDs with shape `(batch, time)`, token lookup produces:

```text
(batch, time, d_model)
```

The position vector for position `t` is added to every example in the batch:

```text
x₀[b, t] = token_embedding[token_id[b, t]] + position_embedding[t]
```

The token embedding answers “what token is this?” The position embedding answers “where in the context does it occur?”

## 7. Causal self-attention

The attention implementation is in `MiniGPT.forward()`.

The model projects each token representation into queries, keys, and values:

```text
Q, K, V = x₀ Wqkv
```

The combined parameter `wqkv` has shape:

```text
(d_model, 3 * d_model)
```

It is split into three arrays, each with shape `(batch, time, d_model)`, then reshaped into heads:

```text
(batch, n_heads, time, head_dim)
```

For each head, scaled dot-product attention is:

```text
scores = QKᵀ / √head_dim
weights = softmax(scores)
attention_output = weights V
```

The scaling factor prevents dot products from becoming too large as the head dimension grows.

### The causal mask

Without a mask, the prediction at position `t` could look at future tokens. MiniLLM sets the upper triangle of the score matrix to a large negative number before softmax:

```text
position 0 may see: 0
position 1 may see: 0, 1
position 2 may see: 0, 1, 2
```

After softmax, masked positions have probability zero. `tests/test_core.py` checks this behavior.

The heads are concatenated and projected back to `d_model` using `wo`:

```text
attention_output = concat(heads) Wo + bo
```

## 8. Residual connections and LayerNorm

The attention output is added to its input:

```text
x₁ = x₀ + attention_output
```

This is a residual connection. It gives the optimizer a direct path through the network and makes deeper networks easier to train.

LayerNorm normalizes each token independently across its feature dimension:

```text
mean = average(x)
variance = average((x - mean)²)
normalized = (x - mean) / √(variance + ε)
output = gamma * normalized + beta
```

`gamma` and `beta` are learned vectors of length `d_model`. The implementation returns a cache during the forward pass so the backward pass can reuse the intermediate statistics.

## 9. Feed-forward network

The feed-forward network processes each position independently:

```text
z = LayerNorm(x₁) W₁ + b₁
h = GELU(z)
mlp = h W₂ + b₂
x₂ = LayerNorm(x₁) + mlp
```

The inner width is `d_ff`, usually larger than `d_model`. GELU is a smooth activation function commonly used in transformer models.

MiniLLM uses one transformer block. A larger model would repeat this attention-plus-MLP block several times.

## 10. Output logits and loss

The final normalized representation is projected to vocabulary-sized logits:

```text
logits = final_representation Wlm + blm
```

Shape:

```text
(batch, time, vocab_size)
```

Logits are unnormalized scores. Softmax converts them into probabilities:

```text
p(token) = softmax(logits)
```

The cross-entropy loss for the correct target token is:

```text
loss = -log(p(correct_token))
```

MiniLLM computes this directly in `loss_and_backward()`.

## 11. Manual backpropagation

MiniLLM does not use PyTorch autograd or another automatic differentiation system. `loss_and_backward()` explicitly computes gradients in reverse order:

```text
loss
  ↓
output projection
  ↓
LayerNorm
  ↓
MLP and GELU
  ↓
LayerNorm
  ↓
attention output projection
  ↓
softmax attention weights
  ↓
Q, K, V projections
  ↓
embeddings
```

The backward pass returns a dictionary with one gradient array per parameter. Each gradient has exactly the same shape as its parameter.

### Why explicit gradients?

Manual derivatives make implementation mistakes visible. The gradient test perturbs one parameter by a small amount and compares:

```text
numerical_gradient ≈ (loss(parameter + ε) - loss(parameter - ε)) / (2ε)
```

The current tests cover the MLP path, and the implementation has also been checked manually across embeddings, attention, LayerNorm, MLP, and the output head.

## 12. Adam optimization

The `Adam` class in [minillm/nn.py](../minillm/nn.py) maintains two moving averages for each parameter:

```text
m = moving average of gradient
v = moving average of squared gradient
```

Bias-corrected values are used to update each parameter:

```text
parameter -= learning_rate * m_hat / (√v_hat + ε)
```

The implementation clips each gradient element to `[-1, 1]`. This is a simple guard against unstable updates in a tiny educational model.

## 13. The training loop

`Trainer` in `minillm/train.py` connects data, model, loss, gradients, and Adam:

```python
trainer = Trainer(model, token_ids, batch_size=16, learning_rate=3e-3)
losses = trainer.fit(
    steps=400,
    checkpoint="artifacts/minillm.npz",
    tokenizer_data=tokenizer.to_dict(),
)
```

The small `train()` wrapper in the same module calls this API for short scripts. The important signal is the loss trend. A healthy run should generally show the loss decreasing, although individual batches can fluctuate.

### Useful experiments

- Change `context_length` and see how much history the model uses.
- Double `d_model` and compare loss versus runtime.
- Change `n_heads` while keeping `d_model` divisible by it.
- Replace the toy corpus with a small plain-text file.
- Train for 25, 100, and 400 steps and compare generated text.
- Set a fixed seed, then change it and observe different samples.

## 14. Generation

Generation is autoregressive:

```text
prompt
  → predict next token
  → append token
  → predict again
  → repeat
```

The model only uses the most recent `context_length` tokens. This keeps inference compatible with the fixed positional embedding table.

### Temperature

The final logits are divided by temperature before softmax:

```text
adjusted_logits = logits / temperature
```

- Lower temperature: safer, more repetitive output.
- Higher temperature: more varied, more random output.

The example uses `0.7` by default.

### Top-k sampling

With top-k sampling, only the k highest-scoring token candidates remain eligible. The example uses `top_k=8` inside the Python API. This reduces unlikely or nonsensical samples while retaining variation.

## 15. Checkpoints

`model.save()` stores:

- Every model parameter
- The model configuration
- The tokenizer vocabulary

`MiniGPT.load()` reconstructs the model and restores its parameters:

```python
model, tokenizer_data = MiniGPT.load("artifacts/minillm.npz")
tokenizer = CharTokenizer.from_dict(tokenizer_data)
```

Checkpoints are NumPy `.npz` files and are intended for local experiments. Do not load untrusted checkpoint files with `allow_pickle=True`.

## 16. Tests and correctness checks

Run:

```bash
source .venv/bin/activate
python -m pytest -q
```

The tests cover:

- Tokenizer round-trip behavior
- Unknown-character handling
- Forward-pass shapes
- Causal attention masking
- Softmax normalization
- Loss reduction during training
- A finite-difference gradient check

If `pytest` is unavailable, the core test functions can still be invoked directly, but installing the requirements is recommended.

## 17. Common problems

### `ModuleNotFoundError: No module named 'minillm'`

Run examples from the repository root as modules:

```bash
source .venv/bin/activate
python -m examples.train_tiny
```

Do not execute `examples/train_tiny.py` as a standalone file unless the repository root has been added to `PYTHONPATH`.

### Loss becomes NaN

Try:

- Lowering the learning rate.
- Reducing `d_model` or `d_ff`.
- Checking that the corpus is not empty.
- Checking that `context_length < len(token_ids)`.
- Confirming that input IDs are valid vocabulary indices.

The implementation already uses a stabilized softmax and clips gradients.

### Generated text is nonsense

This is expected with very few training steps. Try:

```bash
source .venv/bin/activate
python -m examples.train_tiny --steps 400
```

Also use a larger, cleaner corpus and adjust temperature.

### Runtime is slow

The implementation favors readability over performance. Reduce batch size, context length, or model width while experimenting.

## 18. Limitations and next steps

The current implementation intentionally omits many production features. Good extensions are:

1. Add a validation split and report train/validation loss.
2. Add more automated attention-gradient tests.
3. Add input validation for empty prompts and invalid `top_k` values.
4. Add a learning-rate warmup or decay schedule.
5. Add dropout during training.
6. Support multiple transformer blocks.
7. Add a subword tokenizer such as byte-pair encoding.
8. Add key/value caching for faster generation.
9. Add checkpoint metadata and safer loading for untrusted files.
10. Compare the NumPy implementation with a PyTorch reference implementation.

## 19. A useful mental model

At every position, MiniLLM asks:

> Given the tokens to my left, what token is most likely to come next?

The tokenizer turns text into IDs. Embeddings turn IDs into vectors. Attention lets each position gather information from earlier positions. The MLP transforms that gathered information. The output head scores every possible next token. Cross-entropy tells the model how wrong it was, and backpropagation plus Adam adjusts the parameters.

That entire loop is the core of a decoder-only language model.
