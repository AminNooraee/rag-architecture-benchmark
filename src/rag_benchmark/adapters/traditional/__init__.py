"""Traditional Vector RAG baseline architecture (Phase 3C)."""

from rag_benchmark.adapters.traditional.adapter import TraditionalVectorRAGAdapter
from rag_benchmark.adapters.traditional.chunking import (
    TextChunk,
    chunk_document,
    make_chunk_id,
    validate_chunking_params,
)
from rag_benchmark.adapters.traditional.config import TraditionalRAGConfig
from rag_benchmark.adapters.traditional.providers import (
    LLM_SYSTEM_INSTRUCTION,
    EmbeddingProvider,
    LLMGenerationResult,
    LLMProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
    build_llm_messages,
    resolve_api_key_from_env,
)
from rag_benchmark.adapters.traditional.vector_index import (
    ExactCosineIndex,
    IndexEntry,
    cosine_similarity,
    validate_vector,
)

__all__ = [
    "LLMGenerationResult",
    "LLMProvider",
    "LLM_SYSTEM_INSTRUCTION",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleLLMProvider",
    "ExactCosineIndex",
    "EmbeddingProvider",
    "IndexEntry",
    "TextChunk",
    "TraditionalRAGConfig",
    "TraditionalVectorRAGAdapter",
    "build_llm_messages",
    "chunk_document",
    "cosine_similarity",
    "make_chunk_id",
    "resolve_api_key_from_env",
    "validate_chunking_params",
    "validate_vector",
]
