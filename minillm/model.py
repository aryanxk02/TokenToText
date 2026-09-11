"""A small decoder-only transformer with explicit backpropagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nn import gelu, gelu_gradient, layer_norm, layer_norm_backward, softmax


@dataclass
class ModelConfig:
    """Hyperparameters that define the shape of a :class:`MiniGPT` model."""

    vocab_size: int
    context_length: int = 32
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64

    def __post_init__(self) -> None:
        """Reject dimensions that cannot form a valid attention layer."""
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("n_heads must be positive and divide d_model")


@dataclass
class ForwardCache:
    """Named intermediate values saved for the backward pass."""

    token_ids: np.ndarray
    input_vectors: np.ndarray
    queries: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    attention_weights: np.ndarray
    attention_context: np.ndarray
    attention_output: np.ndarray
    residual_one: np.ndarray
    layer_norm_one_cache: tuple
    normalized_one: np.ndarray
    mlp_pre_activation: np.ndarray
    residual_two: np.ndarray
    layer_norm_two_cache: tuple
    normalized_two: np.ndarray

    def __getitem__(self, index: int):
        """Support the old tuple-style cache indexes during migration."""
        values = (
            self.token_ids,
            self.input_vectors,
            self.queries,
            self.keys,
            self.values,
            self.attention_weights,
            self.attention_context,
            self.attention_output,
            self.residual_one,
            self.layer_norm_one_cache,
            self.normalized_one,
            self.mlp_pre_activation,
            self.residual_two,
            self.layer_norm_two_cache,
            self.normalized_two,
        )
        return values[index]


class MiniGPT:
    """A one-block causal transformer that predicts the next token."""

    def __init__(self, config: ModelConfig, seed: int = 0) -> None:
        """Create reproducible model parameters from ``config`` and ``seed``."""
        self.config = config
        self.parameters = self._initialize_parameters(seed)

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Expose model parameters using the original public name."""
        return self.parameters

    def _initialize_parameters(self, seed: int) -> dict[str, np.ndarray]:
        """Create all trainable arrays with simple small initial values."""
        random_generator = np.random.default_rng(seed)
        config = self.config
        return {
            "token_embedding": random_generator.normal(0, 0.02, (config.vocab_size, config.d_model)),
            "position_embedding": random_generator.normal(0, 0.02, (config.context_length, config.d_model)),
            "wqkv": random_generator.normal(0, 0.02, (config.d_model, 3 * config.d_model)),
            "bo": np.zeros(config.d_model),
            "wo": random_generator.normal(0, 0.02, (config.d_model, config.d_model)),
            "ln1_g": np.ones(config.d_model),
            "ln1_b": np.zeros(config.d_model),
            "w1": random_generator.normal(0, 0.02, (config.d_model, config.d_ff)),
            "b1": np.zeros(config.d_ff),
            "w2": random_generator.normal(0, 0.02, (config.d_ff, config.d_model)),
            "b2": np.zeros(config.d_model),
            "ln2_g": np.ones(config.d_model),
            "ln2_b": np.zeros(config.d_model),
            "lm_head": random_generator.normal(0, 0.02, (config.d_model, config.vocab_size)),
            "lm_b": np.zeros(config.vocab_size),
        }

    def _validate_token_ids(self, token_ids: np.ndarray) -> None:
        """Raise a clear error when token IDs cannot be processed."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, time)")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one time step")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds "
                f"context_length {self.config.context_length}"
            )
        if np.any(token_ids < 0) or np.any(token_ids >= self.config.vocab_size):
            raise ValueError("token_ids contain an ID outside the vocabulary")

    def _split_attention_inputs(self, combined: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split and reshape combined QKV projections into attention heads."""
        batch_size, time_steps, _ = combined.shape
        head_count = self.config.n_heads
        head_size = self.config.d_model // head_count
        query, key, value = np.split(combined, 3, axis=-1)
        shape = (batch_size, time_steps, head_count, head_size)
        query = query.reshape(shape).transpose(0, 2, 1, 3)
        key = key.reshape(shape).transpose(0, 2, 1, 3)
        value = value.reshape(shape).transpose(0, 2, 1, 3)
        return query, key, value

    def _attention_forward(
        self,
        input_vectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute causal multi-head attention and return its intermediates."""
        config = self.config
        parameters = self.parameters
        query, key, value = self._split_attention_inputs(input_vectors @ parameters["wqkv"])
        head_size = config.d_model // config.n_heads
        scores = query @ key.transpose(0, 1, 3, 2) / np.sqrt(head_size)
        time_steps = input_vectors.shape[1]
        future_mask = np.triu(np.ones((time_steps, time_steps), dtype=bool), 1)
        scores = np.where(future_mask[None, None], -1e9, scores)
        weights = softmax(scores, axis=-1)
        attention_context = weights @ value
        attention_context = attention_context.transpose(0, 2, 1, 3)
        attention_context = attention_context.reshape(input_vectors.shape)
        attention_output = attention_context @ parameters["wo"] + parameters["bo"]
        return attention_output, attention_context, query, key, value, weights

    def forward(self, token_ids: np.ndarray, training: bool = True) -> tuple[np.ndarray, ForwardCache]:
        """Run the model and return logits plus a cache for training.

        ``training`` is retained as a clear API boundary for future features
        such as dropout. The current model has no training-only operations.
        """
        del training
        self._validate_token_ids(token_ids)
        time_steps = token_ids.shape[1]
        parameters = self.parameters
        positions = np.arange(time_steps)
        input_vectors = parameters["token_embedding"][token_ids]
        input_vectors = input_vectors + parameters["position_embedding"][positions]
        attention_output, attention_context, query, key, value, weights = self._attention_forward(input_vectors)
        residual_one = input_vectors + attention_output
        normalized_one, norm_one_cache = layer_norm(
            residual_one, parameters["ln1_g"], parameters["ln1_b"]
        )
        mlp_pre_activation = normalized_one @ parameters["w1"] + parameters["b1"]
        mlp_output = gelu(mlp_pre_activation) @ parameters["w2"] + parameters["b2"]
        residual_two = normalized_one + mlp_output
        normalized_two, norm_two_cache = layer_norm(
            residual_two, parameters["ln2_g"], parameters["ln2_b"]
        )
        logits = normalized_two @ parameters["lm_head"] + parameters["lm_b"]
        cache = ForwardCache(
            token_ids, input_vectors, query, key, value, weights, attention_context,
            attention_output, residual_one, norm_one_cache, normalized_one, mlp_pre_activation,
            residual_two, norm_two_cache, normalized_two,
        )
        return logits, cache

    def _empty_gradients(self) -> dict[str, np.ndarray]:
        """Create zero-filled arrays that mirror every model parameter."""
        return {
            name: np.zeros_like(parameter)
            for name, parameter in self.parameters.items()
        }

    def _backward_attention(
        self,
        gradient: np.ndarray,
        cache: ForwardCache,
        gradients: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Backpropagate through the attention projection and causal softmax."""
        batch_size, time_steps, _ = gradient.shape
        head_count = self.config.n_heads
        head_size = self.config.d_model // head_count
        parameters = self.parameters
        attention_gradient = gradient @ parameters["wo"].T
        gradients["wo"] = cache.attention_context.reshape(-1, self.config.d_model).T @ gradient.reshape(-1, self.config.d_model)
        gradients["bo"] = gradient.sum(axis=(0, 1))
        attention_gradient = attention_gradient.reshape(batch_size, time_steps, head_count, head_size)
        attention_gradient = attention_gradient.transpose(0, 2, 1, 3)
        values = cache.values
        weights = cache.attention_weights
        value_gradient = weights.transpose(0, 1, 3, 2) @ attention_gradient
        weight_gradient = attention_gradient @ values.transpose(0, 1, 3, 2)
        score_gradient = weights * (
            weight_gradient - (weight_gradient * weights).sum(axis=-1, keepdims=True)
        )
        future_mask = np.triu(np.ones((time_steps, time_steps), dtype=bool), 1)
        score_gradient *= ~future_mask[None, None]
        score_gradient /= np.sqrt(head_size)
        query_gradient = score_gradient @ cache.keys
        key_gradient = score_gradient.transpose(0, 1, 3, 2) @ cache.queries
        query_gradient = query_gradient.transpose(0, 2, 1, 3).reshape(batch_size, time_steps, self.config.d_model)
        key_gradient = key_gradient.transpose(0, 2, 1, 3).reshape(batch_size, time_steps, self.config.d_model)
        value_gradient = value_gradient.transpose(0, 2, 1, 3).reshape(batch_size, time_steps, self.config.d_model)
        combined_gradient = np.concatenate([query_gradient, key_gradient, value_gradient], axis=-1)
        gradients["wqkv"] = cache.input_vectors.reshape(-1, self.config.d_model).T @ combined_gradient.reshape(-1, 3 * self.config.d_model)
        return combined_gradient @ parameters["wqkv"].T

    def loss_and_backward(
        self,
        token_ids: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Compute cross-entropy loss and all parameter gradients."""
        if targets.shape != token_ids.shape:
            raise ValueError("targets must have the same shape as token_ids")
        logits, cache = self.forward(token_ids)
        batch_size, time_steps, vocabulary_size = logits.shape
        probabilities = softmax(logits, axis=-1)
        flat_targets = targets.reshape(-1)
        flat_probabilities = probabilities.reshape(-1, vocabulary_size)
        target_indices = np.arange(batch_size * time_steps)
        loss = -np.log(np.maximum(flat_probabilities[target_indices, flat_targets], 1e-12)).mean()
        logits_gradient = probabilities.copy()
        logits_gradient.reshape(-1, vocabulary_size)[target_indices, flat_targets] -= 1.0
        logits_gradient /= batch_size * time_steps
        gradients = self._empty_gradients()
        gradients["lm_head"] = cache.normalized_two.reshape(-1, self.config.d_model).T @ logits_gradient.reshape(-1, vocabulary_size)
        gradients["lm_b"] = logits_gradient.sum(axis=(0, 1))
        residual_two_gradient = logits_gradient @ self.parameters["lm_head"].T
        residual_two_gradient, gradients["ln2_g"], gradients["ln2_b"] = layer_norm_backward(
            residual_two_gradient, cache.layer_norm_two_cache
        )
        normalized_one_gradient = residual_two_gradient.copy()
        mlp_gradient = residual_two_gradient
        gelu_input_gradient = mlp_gradient @ self.parameters["w2"].T
        gradients["w2"] = gelu(cache.mlp_pre_activation).reshape(-1, self.config.d_ff).T @ mlp_gradient.reshape(-1, self.config.d_model)
        gradients["b2"] = mlp_gradient.sum(axis=(0, 1))
        gelu_input_gradient *= gelu_gradient(cache.mlp_pre_activation)
        gradients["w1"] = cache.normalized_one.reshape(-1, self.config.d_model).T @ gelu_input_gradient.reshape(-1, self.config.d_ff)
        gradients["b1"] = gelu_input_gradient.sum(axis=(0, 1))
        normalized_one_gradient += gelu_input_gradient @ self.parameters["w1"].T
        residual_one_gradient, gradients["ln1_g"], gradients["ln1_b"] = layer_norm_backward(
            normalized_one_gradient, cache.layer_norm_one_cache
        )
        input_gradient = residual_one_gradient + self._backward_attention(
            residual_one_gradient, cache, gradients
        )
        np.add.at(gradients["token_embedding"], cache.token_ids, input_gradient)
        gradients["position_embedding"][:time_steps] = input_gradient.sum(axis=0)
        return float(loss), gradients

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        random_generator: np.random.Generator,
        temperature: float = 0.8,
        top_k: int = 0,
    ) -> list[int]:
        """Sample new token IDs autoregressively from a prompt."""
        if not prompt_ids:
            raise ValueError("prompt_ids must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        generated_ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            context_ids = np.asarray(
                [generated_ids[-self.config.context_length :]], dtype=np.int64
            )
            logits, _ = self.forward(context_ids, training=False)
            next_logits = logits[0, -1] / temperature
            next_logits = self._apply_top_k(next_logits, top_k)
            next_probabilities = softmax(next_logits)
            next_id = random_generator.choice(len(next_probabilities), p=next_probabilities)
            generated_ids.append(int(next_id))
        return generated_ids

    @staticmethod
    def _apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
        """Mask all but the highest ``top_k`` logits, when requested."""
        if top_k == 0:
            return logits
        effective_top_k = min(top_k, len(logits))
        keep_indices = np.argpartition(logits, -effective_top_k)[-effective_top_k:]
        filtered_logits = np.full_like(logits, -1e9)
        filtered_logits[keep_indices] = logits[keep_indices]
        return filtered_logits

    def save(self, path: str, tokenizer_data: dict) -> None:
        """Save parameters, configuration, and tokenizer metadata to ``path``."""
        np.savez(
            path,
            **self.parameters,
            tokenizer=np.array([tokenizer_data], dtype=object),
            config=np.array([self.config.__dict__], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> tuple["MiniGPT", dict]:
        """Load a model checkpoint created by :meth:`save`."""
        data = np.load(path, allow_pickle=True)
        model = cls(ModelConfig(**data["config"][0]))
        for name in model.parameters:
            if name not in data:
                raise ValueError(f"checkpoint is missing parameter: {name}")
            model.parameters[name][...] = data[name]
        tokenizer_data = data["tokenizer"][0]
        if hasattr(tokenizer_data, "item"):
            tokenizer_data = tokenizer_data.item()
        return model, tokenizer_data
