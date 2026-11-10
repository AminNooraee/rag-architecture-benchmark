"""Deterministic offline tokenizer for the official LightRAG adapter.

LightRAG's default tokenizer is tiktoken-based and downloads BPE models from
the network. For reproducible offline benchmark runs the adapter injects a
character-level tokenizer so that ``chunk_token_size`` /
``chunk_overlap_token_size`` act as exact character windows with zero network
access and zero model state.
"""

from __future__ import annotations

from typing import Any


class CharacterTokenizer:
    """Offline tokenizer that maps each Unicode code point to one token.

    The token id is the code point itself, which is unique per character and
    round-trips exactly through :meth:`decode`, so LightRAG's token-count
    based chunking and its token->text reconstruction both stay lossless.

    Stateless and therefore safe to call concurrently from multiple OS
    threads (LightRAG runs token counting off the event loop), and trivially
    deepcopy-safe (no locks or mutable state).
    """

    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def build_offline_character_tokenizer() -> Any:
    """Build a ``lightrag.utils.Tokenizer`` backed by :class:`CharacterTokenizer`.

    Raises ImportError when the optional LightRAG dependency is missing; the
    adapter converts that into a ConfigurationError at the call site.
    """
    from lightrag.utils import Tokenizer  # type: ignore[import-untyped]

    return Tokenizer(model_name="rag-benchmark-char-1", tokenizer=CharacterTokenizer())
