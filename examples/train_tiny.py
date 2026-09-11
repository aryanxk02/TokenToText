from __future__ import annotations

import argparse

from minillm import CharTokenizer, MiniGPT, ModelConfig
from minillm.train import train


CORPUS = ("the little model learns one small pattern. "
          "the model reads tokens and predicts the next token.\n") * 40


def main() -> None:
    """Train MiniLLM on a small built-in corpus and save its checkpoint."""
    parser = argparse.ArgumentParser(description="Train MiniLLM on a tiny built-in corpus")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--checkpoint", default="artifacts/minillm.npz")
    args = parser.parse_args()
    tokenizer = CharTokenizer(CORPUS)
    model = MiniGPT(ModelConfig(tokenizer.vocab_size, context_length=32, d_model=32, n_heads=4, d_ff=64), seed=7)
    train(model, tokenizer.encode(CORPUS), steps=args.steps, checkpoint=args.checkpoint, tokenizer=tokenizer.to_dict())
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
