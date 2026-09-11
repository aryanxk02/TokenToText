"""A readable character-level tokenizer for the MiniLLM examples."""

from __future__ import annotations


class CharTokenizer:
    """Map individual characters to integer IDs and back again."""

    UNKNOWN_TOKEN = "<unk>"
    PADDING_TOKEN = "<pad>"
    UNK = UNKNOWN_TOKEN
    PAD = PADDING_TOKEN

    def __init__(self, text: str) -> None:
        """Build a sorted vocabulary from the characters in ``text``."""
        if not text:
            raise ValueError("text must contain at least one character")
        self.tokens = [self.PADDING_TOKEN, self.UNKNOWN_TOKEN] + sorted(set(text))
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.itos = self.tokens
        self.stoi = self.token_to_id

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Convert ``text`` into token IDs, using ``<unk>`` when necessary."""
        unknown_id = self.token_to_id[self.UNKNOWN_TOKEN]
        return [self.token_to_id.get(character, unknown_id) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs into text and omit padding tokens."""
        decoded_tokens = []
        for token_id in token_ids:
            if 0 <= token_id < self.vocab_size:
                decoded_tokens.append(self.tokens[token_id])
            else:
                decoded_tokens.append(self.UNKNOWN_TOKEN)
        return "".join(decoded_tokens).replace(self.PADDING_TOKEN, "")

    def to_dict(self) -> dict[str, list[str]]:
        """Return the vocabulary in a JSON-like dictionary."""
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CharTokenizer":
        """Rebuild a tokenizer from a dictionary created by :meth:`to_dict`."""
        tokens = data.get("tokens", data.get("itos"))
        if not tokens:
            raise ValueError("tokenizer data must contain a non-empty 'tokens' list")
        tokenizer = cls.__new__(cls)
        tokenizer.tokens = list(tokens)
        tokenizer.token_to_id = {token: index for index, token in enumerate(tokens)}
        tokenizer.itos = tokenizer.tokens
        tokenizer.stoi = tokenizer.token_to_id
        return tokenizer
