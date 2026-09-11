"""Data utilities for next-token language-model training."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class BatchGenerator:
    """Generate random input and target windows from a tokenized corpus."""

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        batch_size: int,
        random_generator: np.random.Generator,
    ) -> None:
        """Validate and store the settings used for random sampling."""
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(token_ids) <= context_length:
            raise ValueError("corpus must be longer than context_length")
        self.values = np.asarray(token_ids, dtype=np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.random_generator = random_generator

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield random batches forever."""
        while True:
            yield self.next_batch()

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return inputs and one-token-shifted targets."""
        maximum_start = len(self.values) - self.context_length
        starts = self.random_generator.integers(0, maximum_start, size=self.batch_size)
        inputs = np.stack([
            self.values[start : start + self.context_length]
            for start in starts
        ])
        targets = np.stack([
            self.values[start + 1 : start + self.context_length + 1]
            for start in starts
        ])
        return inputs, targets


def make_batches(
    token_ids: list[int],
    context_length: int,
    batch_size: int,
    random_generator: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return a backwards-compatible iterator of random training batches."""
    return iter(BatchGenerator(token_ids, context_length, batch_size, random_generator))
