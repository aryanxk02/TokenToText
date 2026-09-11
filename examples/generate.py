from __future__ import annotations

import argparse

import numpy as np

from minillm import CharTokenizer, MiniGPT


def main() -> None:
    """Load a MiniLLM checkpoint and print sampled text."""
    parser = argparse.ArgumentParser(description="Generate text from a MiniLLM checkpoint")
    parser.add_argument("--checkpoint", default="artifacts/minillm.npz")
    parser.add_argument("--prompt", default="the ")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    model, tokenizer_data = MiniGPT.load(args.checkpoint)
    tokenizer = CharTokenizer.from_dict(tokenizer_data)
    ids = model.generate(tokenizer.encode(args.prompt), args.tokens, np.random.default_rng(2), args.temperature, top_k=8)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
