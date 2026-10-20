# Persian RAG Architecture Benchmark

> **Phase 3A — Benchmark Harness Foundation**

A rigorous, production-oriented evaluation harness for comparing Retrieval-Augmented Generation (RAG) architectures on Persian-language corpora.

---

## 1. Project Purpose & Persian Benchmark Goal

Evaluating RAG systems in the Persian language presents unique structural and linguistic challenges:
- Complex morphology, zero-width non-joiners (نیم‌فاصله `\u200c`), non-ASCII glyphs (گ، چ، پ، ژ), and diverse script variations.
- Disparate information retrieval boundaries: local factual lookups, cross-document relationship reasoning, global thematic summarization, and hierarchical abstraction.
- Multi-architecture architectural divergence: classical dense vector search, community-driven graph extraction, and recursive tree clustering make disparate retrieval trade-offs.

The goal of this benchmark harness is to provide a unified, reproducible, and strictly fair foundation for assessing retrieval and generation quality across all candidate RAG architectures.

Later phases will benchmark the following architectures against this harness:
- **Traditional Vector RAG** (Dense chunking and cosine/HNSW vector search)
- **LightRAG** (Dual-level knowledge graph with low overhead)
- **Neo4j GraphRAG** (Explicit property graph with Cypher queries)
- **Microsoft GraphRAG** (Hierarchical community report generation)
- **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval)
- **Hierarchical RAG** (Multi-level document and section indexing)

---

## 2. Canonical Schemas

All systems exchange data through strictly validated, serializable Pydantic v2 data models.

### Document Schema (`BenchmarkDocument`, `DocumentSection`)
Represents source knowledge assets with structural subsections and metadata:
- Full preservation of Persian Unicode and ZWNJ (`\u200c`).
- No destructive normalization or character stripping.
- Section-level hierarchical provenance (`section_id`, `heading`, `text`, `order`, `metadata`).

### Question Schema (`BenchmarkQuestion`, `AdapterQueryInput`)
Represents evaluation queries covering 8 standardized question types:
- `direct_factual`
- `detailed_factual`
- `multi_hop`
- `relationship`
- `cross_document`
- `global_summary`
- `hierarchical`
- `unanswerable`

Scopes include `local`, `multi_document`, and `global`.

#### Ground-Truth Isolation & Architectural Fairness
> **CRITICAL ARCHITECTURAL CONTRACT**:
> The adapter's normal `query()` invocation must **NEVER** receive gold answers, acceptable answers, or ground-truth supporting evidence.
>
> `BenchmarkQuestion.to_query_input()` generates an `AdapterQueryInput` that contains only the question text, identifier, and safe query-time metadata. Ground-truth reference information exists exclusively for downstream evaluation layers.

### Retrieved Evidence Schema (`RetrievedEvidence`)
Represents evidence units independent of underlying RAG topology:
- Supports `chunk`, `document`, `entity`, `relationship`, `community_report`, `summary_node`, `tree_node`, and `cypher_result`.
- Preserves provenance back to the source document (`document_id`, `chunk_id`).
- Strict JSON guarantees (forbidding `NaN`, `Infinity`, and `-Infinity`).

### Query Result Schema (`QueryResult`)
Canonical execution outcome:
- Captures status (`success`, `failed`, `timeout`, `error`), answer text in Persian Unicode, retrieved evidence, `QueryUsage` (token and API call counts), and `QueryTiming` (retrieval, generation, and total latency in milliseconds).
- **Failed queries**: A failed query produces a valid `QueryResult` containing structured diagnostics (`StructuredError`). Queries are never silently discarded.

---

## 3. BaseRAGAdapter Contract

All candidate RAG architectures implement `BaseRAGAdapter`:

```python
class BaseRAGAdapter(ABC):
    @property
    @abstractmethod
    def system_name(self) -> str: ...

    @property
    @abstractmethod
    def variant_name(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    @abstractmethod
    def prepare(self, config: BenchmarkConfig) -> None: ...

    @abstractmethod
    def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult: ...

    @abstractmethod
    def query(self, query_input: AdapterQueryInput) -> QueryResult: ...

    @abstractmethod
    def health(self) -> HealthCheckResult: ...

    @abstractmethod
    def cleanup(self) -> None: ...
```

### Fairness Rules
> **The adapter normalizes execution and output; it must not add architecture-specific improvements that alter benchmark fairness unless declared as a benchmark variant.**

Specifically, an adapter must **NOT** secretly inject:
- Reranking
- HyDE (Hypothetical Document Embeddings)
- Query rewriting
- Multi-query expansion
- Hidden additional retrieval stages

unless an explicit benchmark variant is declared to evaluate that capability.

### Capabilities and Optional Operations
Adapters report supported capabilities via `AdapterCapabilities`:
- `supports_incremental_insert`
- `supports_update`
- `supports_delete`
- `supports_persistence`
- `supports_async`
- `supports_streaming`
- `supports_native_citations`

Optional lifecycle operations (`insert()`, `update()`, `delete()`) raise `UnsupportedOperationError` when called on systems that do not support them.

---

## 4. Run Artifact Layout

Each benchmark execution outputs artifacts into a deterministic, isolated run directory:

```text
runs/<run_id>/
  manifest.json        # Reproducible run metadata, git SHA, and sanitized config
  raw_results.jsonl    # Line-delimited JSON results (Persian UTF-8 unescaped)
  errors.jsonl         # Line-delimited JSON structured errors for diagnostics
```

*(Future evaluation phases will add `metrics.json`, `retrieval_metrics.json`, `generation_metrics.json`, `efficiency.json`, and `production_score.json`. Phase 3A does NOT emit fake metrics.)*

### Overwrite Protection & Secret Safety
- **Collision Protection**: If `runs/<run_id>/` contains a completed run manifest, `RunArtifactWriter` raises `ArtifactCollisionError` unless `allow_overwrite=True` is explicitly supplied.
- **Secret Sanitization**: All credentials (API keys, passwords, bearer tokens) are strictly excluded from configuration serializations and manifests via recursive scrubbing.

---

## 5. Development & Quality Gates

The harness relies on a modern toolchain:
- **Python**: `>=3.11` (tested with Python 3.13)
- **Package Manager**: `uv`
- **Typing**: `mypy --strict`
- **Linter/Formatter**: `ruff`
- **Testing**: `pytest`

### Executing Quality Gates

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
uv run pytest
```
