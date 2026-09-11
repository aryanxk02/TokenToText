"""Training orchestration for the MiniLLM model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import BatchGenerator
from .nn import Adam


class Trainer:
    """Own the batch generator and optimizer for one training run."""

    def __init__(
        self,
        model,
        token_ids: list[int],
        batch_size: int = 16,
        seed: int = 1,
        learning_rate: float = 3e-3,
    ) -> None:
        """Create a trainer connected to ``model`` and a tokenized corpus."""
        self.model = model
        self.random_generator = np.random.default_rng(seed)
        self.batch_generator = BatchGenerator(
            token_ids,
            model.config.context_length,
            batch_size,
            self.random_generator,
        )
        self.optimizer = Adam(model.parameters, learning_rate=learning_rate)

    def fit(
        self,
        steps: int = 400,
        log_every: int = 50,
        checkpoint: str | None = None,
        tokenizer_data: dict | None = None,
    ) -> list[float]:
        """Train for ``steps`` and optionally save a checkpoint."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if log_every < 0:
            raise ValueError("log_every must not be negative")
        losses = []
        for step in range(1, steps + 1):
            inputs, targets = self.batch_generator.next_batch()
            loss, gradients = self.model.loss_and_backward(inputs, targets)
            self.optimizer.step(gradients)
            losses.append(loss)
            self._log_progress(step, steps, loss, log_every)
        if checkpoint is not None:
            self._save_checkpoint(checkpoint, tokenizer_data)
        return losses

    def _log_progress(self, step: int, total_steps: int, loss: float, log_every: int) -> None:
        """Print occasional progress without cluttering short experiments."""
        should_log = log_every and (step == 1 or step % log_every == 0)
        if should_log:
            print(f"step {step:4d}/{total_steps} | loss {loss:.4f}")

    def _save_checkpoint(self, checkpoint: str, tokenizer_data: dict | None) -> None:
        """Create the checkpoint directory and save the current model."""
        if tokenizer_data is None:
            raise ValueError("tokenizer_data is required when saving a checkpoint")
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(checkpoint, tokenizer_data)


def train(
    model,
    token_ids: list[int],
    steps: int = 400,
    batch_size: int = 16,
    seed: int = 1,
    learning_rate: float = 3e-3,
    log_every: int = 50,
    checkpoint: str | None = None,
    tokenizer: dict | None = None,
) -> list[float]:
    """Train a model using a convenient functional wrapper around :class:`Trainer`."""
    trainer = Trainer(model, token_ids, batch_size, seed, learning_rate)
    return trainer.fit(steps, log_every, checkpoint, tokenizer)
