# Persian RAG Architecture Benchmark

> **Phase 3A — Benchmark Harness Foundation (Remediated)**

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

## 2. Canonical Schemas & Strict Data Contracts

All benchmark components exchange data through strictly validated Pydantic v2 models adhering to a shared strict JSON contract.

### Strict JSON Contract
All metadata, configuration extras, and diagnostics are recursively limited to JSON-native primitives:
- `null`, `boolean`, `integer`, `finite float` (no `NaN`, `Infinity`, or `-Infinity`), `string`
- `list` of valid JSON values (Python `tuple` and `set` are strictly rejected)
- `dict` with **string keys only** and valid JSON values
- Datetimes, custom classes, and non-string enums are rejected (`StrEnum` behaves as string and is valid JSON text).
- Nested structures are returned as recursively immutable representations (`FrozenDict`, `FrozenList`), permanently preventing post-construction in-place mutation (e.g. `metadata["k"] = v` or `metadata["list"].append(...)`).

### Document Schema (`BenchmarkDocument`, `DocumentSection`)
Represents source knowledge assets with structural subsections and metadata:
- Full preservation of Persian Unicode and ZWNJ (`\u200c`).
- No destructive normalization or character stripping.
- Section-level hierarchical provenance (`section_id`, `heading`, `text`, `order`, `metadata`).

### Question Schema (`BenchmarkQuestion`, `AdapterQueryInput`)
Represents evaluation queries covering 8 standardized question types:
- `direct_factual`, `detailed_factual`, `multi_hop` (enforces `reasoning_hops >= 2`), `relationship`, `cross_document`, `global_summary`, `hierarchical`, `unanswerable` (enforces `answerable=False` and empty `gold_answer`).

#### Ground-Truth Isolation & Architectural Fairness
> **CRITICAL ARCHITECTURAL CONTRACT**:
> The adapter's normal `query()` invocation must **NEVER** receive gold answers, acceptable answers, or ground-truth supporting evidence.
>
> `BenchmarkQuestion.to_query_input()` generates an `AdapterQueryInput` that contains only the question text, identifier, and safe query-time metadata allowlisted in `query_metadata`. Evaluation-only metadata (`metadata`), gold answers, and reference evidence are strictly withheld. If forbidden ground-truth keys are detected in `query_metadata`, `GroundTruthLeakageError` is raised.

### Retrieved Evidence Schema (`RetrievedEvidence`)
Represents evidence units independent of underlying RAG topology:
- Supports `chunk`, `document`, `entity`, `relationship`, `community_report`, `summary_node`, `tree_node`, and `cypher_result`.
- Preserves provenance back to source documents (`document_id`, `chunk_id`).
- Strict finite scores and 1-based ranks.

### Query Result Schema (`QueryResult`)
Canonical execution outcome:
- Cross-field state matrix:
  - **SUCCESS**: requires `status=SUCCESS`, `error=None`, and `answer=string`.
  - **FAILED / ERROR / TIMEOUT**: requires `status!=SUCCESS`, `error=StructuredError`, and `answer=None`.
  - Nested `StructuredError` must match the parent result's `question_id`, `system`, and `variant`.
- **Failure Evidence Preservation**: If retrieval succeeds but generation fails, `create_failure()` retains all retrieved evidence items, retrieval latency, and stage usage metrics.

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

An adapter must **NOT** secretly inject reranking, HyDE, query rewriting, multi-query expansion, or hidden retrieval stages unless explicitly declared in a benchmark variant.

### Capability Trustworthiness & Activation Lifecycle
Adapters declare supported features via `AdapterCapabilities` (`supports_incremental_insert`, `supports_update`, `supports_delete`, `supports_persistence`, `supports_streaming`, `supports_native_citations`).
- **Trustworthiness Check**: Calling `validate_capabilities()` fails early with `InvalidCapabilityDeclarationError` if an adapter advertises a capability without overriding the corresponding base method.
- **Mandatory Activation**: `BaseRAGAdapter.activate(config)` guarantees that `validate_capabilities()` executes before `prepare(config)`.
  > All adapters must be activated through the validated lifecycle before health/index/query operations are used by the benchmark runner.
- *Note*: Asynchronous execution (`supports_async`) is deferred until Phase 4 when an explicit async query contract is scheduled.

---

## 4. Run Artifact Layout & Single-Writer Model

Each benchmark execution outputs artifacts into an isolated, safe run directory:

```text
runs/<run_id>/
  writer.lock          # Exclusive process lock (single-writer enforcement)
  manifest.json        # Atomic JSON manifest (UTC timestamps, sanitized config, git SHA)
  raw_results.jsonl    # Line-delimited JSON results (Persian UTF-8 unescaped)
  errors.jsonl         # Line-delimited JSON structured errors for diagnostics
```

*(Future evaluation phases will add `metrics.json`, `retrieval_metrics.json`, etc. Phase 3A does not emit fake metrics.)*

### Single-Writer Concurrency Policy
> **`RunArtifactWriter` is a single-writer durable artifact sink. Parallel query workers must send completed results to the owning writer rather than opening independent writers for the same run.**

Exclusive process ownership is enforced via `writer.lock` with a unique owner token. If another process attempts to open or initialize the run concurrently, `WriterConcurrencyError` is raised.

On the same host, stale locks from dead processes are detected via PID inspection and safely taken over using an atomic claim file. Note: In rare cases where an OS PID is recycled and allocated to an unrelated running process, the harness conservatively treats the lock as active to prevent catastrophic concurrent corruption.

### Safe `run_id` Validation
To prevent directory traversal and filesystem corruption, `run_id` must be a safe, single-component identifier:
- Rejects `.`, `..`, path traversal components (`../`), absolute paths, drive letters (`C:`), colons, slashes (`/`, `\`), and Windows reserved device names (`CON`, `NUL`, etc.).
- Verifies that the resolved run directory remains strictly enclosed within the resolved base directory.

### Explicit Lifecycle Modes
`RunArtifactWriter` enforces three unambiguous modes:
1. **CREATE**: Fails with `ArtifactCollisionError` if `manifest.json`, `raw_results.jsonl`, or `errors.jsonl` already exists.
2. **RESUME**: Resumes an actively `RUNNING` run. Reconstructs query counts and recorded question IDs from durable JSONL records, validating cross-file consistency. Rejects terminal runs (`COMPLETED` or `FAILED`).
3. **OVERWRITE**: Sequentially clears previous Phase 3A run artifacts (`manifest.json`, `raw_results.jsonl`, `errors.jsonl`) before initializing clean state. *(Note: OVERWRITE currently manages only Phase 3A core artifacts; future phases will extend cleanup for additional metric or index files).*

### Result Uniqueness & Identity Binding
Within a single run, each question may have only one canonical final result: `(run_id, question_id)`. Appending duplicate question IDs is rejected. Appended results and errors must match the writer's bound `system` and `variant`.

### Atomic Manifest Writes
Manifest writing uses atomic same-directory replacement:
1. Serializes to strict JSON string without touching the destination.
2. Writes to a temporary file in the target directory.
3. Flushes and executes `os.fsync()`.
4. Atomically replaces the destination using `os.replace()`.
If an write error occurs, the pre-existing valid manifest remains completely intact.

### Centralized Secret Redaction
A centralized sanitization layer (`rag_benchmark.security.sanitizer`) protects credentials:
- Exact and compound pattern matching for credentials (`api_key`, `token`, `password`, `secret`, `client_secret`, `private_key`).
- Redacts sensitive values to `"[REDACTED]"` for manifest transparency.
- Sanitizes embedded URL credentials (userinfo and sensitive query params).
- Prevents false positives by explicitly protecting legitimate provenance keys (`author`, `authors`, `tokenizer`, `tokenizer_name`, `authorization_mode`, `token_count`, `input_tokens`, `output_tokens`).
- *Limitation*: Sanitization is designed for structured fields, URLs, and standard token prefixes; it is not an arbitrary free-text NLP classifier.

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
