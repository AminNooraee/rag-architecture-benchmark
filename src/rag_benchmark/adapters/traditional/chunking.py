"""Deterministic Persian-safe character-based chunking for BenchmarkDocument.

Baseline chunking guarantees:
- Persian Unicode and ZWNJ (\\u200c) are preserved verbatim (no normalization).
- Output is deterministic: the same document and parameters always produce the
  same chunks, chunk IDs, and ordering across runs (no process-level hashing).
- Chunk IDs are derived only from the stable document_id and chunk ordinal.
- Empty and whitespace-only chunks are never emitted.
- Only boundary whitespace is trimmed from each chunk; interior whitespace is
  preserved exactly as in the source text.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_benchmark.schemas.document import BenchmarkDocument


def validate_chunking_params(chunk_size: int, overlap: int) -> None:
    """Reject invalid chunking parameters.

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )


@dataclass(frozen=True)
class TextChunk:
    """A single deterministic chunk of a benchmark document.

    char_start/char_end refer to the raw (untrimmed) character span in the
    original document text, providing stable source provenance.
    """

    document_id: str
    chunk_id: str
    text: str
    ordinal: int
    source: str
    char_start: int
    char_end: int


def make_chunk_id(document_id: str, ordinal: int) -> str:
    """Build a stable, deterministic chunk ID from stable inputs."""
    return f"{document_id}::chunk-{ordinal:06d}"


def chunk_document(
    document: BenchmarkDocument,
    chunk_size: int,
    overlap: int,
) -> tuple[TextChunk, ...]:
    """Chunk a document deterministically using a sliding character window.

    The window walks the text from left to right. When a window would split a
    word (i.e. the character at the window edge is not whitespace), the cut is
    backed up to the last whitespace position within the window. Pathological
    runs longer than chunk_size without any whitespace are hard-cut. Each
    chunk keeps `overlap` trailing characters as the leading span of the next
    chunk. Whitespace-only windows are skipped, so the result never contains
    empty chunks; a document with empty or whitespace-only text yields no
    chunks at all.
    """
    validate_chunking_params(chunk_size, overlap)

    text = document.text
    total = len(text)
    chunks: list[TextChunk] = []
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        cut = end
        if end < total and not text[end].isspace():
            for candidate in range(end - 1, start, -1):
                if text[candidate].isspace():
                    cut = candidate
                    break
        raw = text[start:cut]
        chunk_text = raw.strip()
        if chunk_text:
            ordinal = len(chunks)
            chunks.append(
                TextChunk(
                    document_id=document.document_id,
                    chunk_id=make_chunk_id(document.document_id, ordinal),
                    text=chunk_text,
                    ordinal=ordinal,
                    source=document.source,
                    char_start=start,
                    char_end=cut,
                )
            )
        if cut == total:
            # Window reached the end of the text; the only remaining tail is
            # at most `overlap` characters of already-covered text.
            break
        next_start = cut - overlap
        if next_start <= start:
            next_start = cut
        start = next_start

    return tuple(chunks)
