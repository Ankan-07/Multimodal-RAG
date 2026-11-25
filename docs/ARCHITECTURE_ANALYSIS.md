# Multimodal RAG — Architecture Analysis, Bottlenecks & Scaling Roadmap

> **Legacy note**: This document describes the older summary-based pipeline.
> The current hybrid MiniLM + Gemini + BM25 design is described in the root
> `README.md` and `docs/README.md`.

> **Scope**: Deep-dive review of the current system design, a bottleneck catalogue for bulk ingestion / scale scenarios, and a concrete path to a production-grade architecture.

---

## 1. Current Architecture

### 1.1 High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  INGESTION  (index-time)                                             │
│                                                                      │
│  PDF File                                                            │
│     │                                                                │
│     ▼                                                                │
│  partition_pdf()  ──  unstructured hi_res strategy                  │
│     │               (layout model + OCR + table detection)          │
│     ▼                                                                │
│  chunk_by_title()  ── title-boundary semantic chunking              │
│     │                                                                │
│     ▼                                                                │
│  separate_content_types()  ── splits each chunk into:               │
│     │                          text | tables (HTML) | images (b64)  │
│     ▼                                                                │
│  create_ai_enhanced_summary()  ── VLM call per multimedia chunk     │
│     │    (nvidia/nemotron-nano-12b via OpenRouter)                   │
│     │    text-only chunks skip this step                            │
│     ▼                                                                │
│  LangChain Document                                                  │
│     ├── page_content  =  AI-generated searchable description        │
│     └── metadata["original_content"]  =  JSON blob with            │
│              raw_text | tables_html | image_base64                  │
│     ▼                                                                │
│  HuggingFaceEmbeddings("all-MiniLM-L6-v2")                       │
│     ▼                                                                │
│  ChromaDB  (HNSW, cosine, persisted to dbv2/chroma_db)             │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  QUERY  (query-time)                                                 │
│                                                                      │
│  User Query                                                          │
│     ▼                                                                │
│  HuggingFaceEmbeddings (same model, re-instantiated)              │
│     ▼                                                                │
│  ChromaDB.as_retriever(k=3)  ── top-3 cosine-nearest chunks        │
│     ▼                                                                │
│  generate_final_answer()                                             │
│     ├── builds multipart HumanMessage:                              │
│     │     text prompt + HTML tables (as text) + base64 images       │
│     └── calls google/gemma-3-12b-it:free via OpenRouter            │
│     ▼                                                                │
│  Answer + Source Attribution  →  Streamlit UI                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Map

| File | Role | Stage |
|---|---|---|
| `document_processor.py` | PDF partition, title chunking, content type separation | Extract |
| `summarizer.py` | Per-chunk VLM summarization, Document assembly | Enrich |
| `vector_store.py` | Chroma create / load | Store |
| `retriever.py` | Similarity retrieval, answer generation | Retrieve + Generate |
| `pipeline.py` | Orchestration CLI | Entry point |
| `app.py` | Streamlit UI | Frontend |
| `config.py` | All constants | Cross-cutting |

### 1.3 Key Design Strengths

- **Dual-storage pattern** — `page_content` holds the VLM-generated searchable description (optimised for embedding recall), while `metadata["original_content"]` preserves raw text/tables/images for faithful answer generation. Decouples retrieval quality from answer fidelity.
- **Title-boundary chunking** — `chunk_by_title` preserves document structure semantics rather than hard-cutting at character limits.
- **Selective enrichment** — AI summarization is only invoked for chunks containing tables or images; text-only chunks skip it, saving API cost.
- **Config-centralised** — model names, chunk sizes, K, paths all live in `config.py`.
- **Modular refactor** — `main.py` (monolith) has been cleanly split into domain modules, making each stage independently testable.

---

## 2. Current Architecture Diagram (as-is)

```
┌──────────────┐     upload     ┌────────────────────────────────────┐
│   Browser    │──────────────▶│            app.py                   │
│  (Streamlit) │               │  session_state.vector_db (in-mem)   │
└──────────────┘               └────────────┬───────────────────────┘
                                            │ orchestrates
                    ┌────────────────────────▼──────────────────────┐
                    │           Ingestion (synchronous, serial)      │
                    │                                                │
                    │  partition_document()   [CPU/GPU — slow]      │
                    │       ↓                                        │
                    │  create_chunks_by_title()                      │
                    │       ↓                                        │
                    │  for chunk in chunks:                          │
                    │      separate_content_types()                  │
                    │      create_ai_enhanced_summary()  ←──[API]   │
                    │       ↓ (sequential, 1 LLM call per chunk)    │
                    │  create_vector_store()   [in-memory Chroma]   │
                    └────────────────────────────────────────────────┘

                    ┌────────────────────────────────────────────────┐
                    │           Query (synchronous)                  │
                    │                                                │
                    │  SentenceTransformer embed query               │
                    │       ↓                                        │
                    │  Chroma HNSW top-3                            │
                    │       ↓                                        │
                    │  generate_final_answer()   ←──────── [API]    │
                    └────────────────────────────────────────────────┘
```

---

## 3. Bottleneck Catalogue — Bulk Ingestion & Scale

### 3.1 Summarization Is Serial and Unbounded  ⚠️ Critical

**Where**: `summarizer.py → summarize_chunks()` — a plain `for` loop.

**Problem**: For a 100-page document that produces 80 chunks (of which 40 contain images/tables), you fire 40 sequential API calls with no concurrency. Each call is a round-trip to OpenRouter — at ~2–5 seconds per call, 40 chunks = 80–200 seconds of pure waiting. At 10 documents this is 800–2,000 seconds. The user watches a spinner the entire time.

**Compounding factor**: The VLM model is a free-tier endpoint (`nvidia/nemotron-nano-12b-v2-vl:free`). Free-tier endpoints typically enforce per-minute rate limits and queue requests aggressively under load, making serial queuing even slower than parallel with backoff.

**Signal in code**:
```python
# summarizer.py:35
for chunk in chunks:          # ← no asyncio, no ThreadPoolExecutor
    ...create_ai_enhanced_summary(...)
```

---

### 3.2 PDF Partitioning Is Single-Threaded and CPU/GPU Bound  ⚠️ Critical

**Where**: `document_processor.py → partition_document()`.

**Problem**: `unstructured`'s `hi_res` strategy loads a layout detection model (detectron2 or YOLO-based), runs it page-by-page, performs OCR (via pytesseract/paddle), and then does table structure inference. For a 100-page, image-dense PDF this easily takes 5–20 minutes on CPU. There is no multi-document parallelism — `app.py` processes each uploaded file in a sequential loop:

```python
# app.py:53
for uploaded_file in uploaded_files:
    elements = partition_document(temp_path)   # blocking, sequential
```

Uploading 5 documents means 5 fully sequential `partition_pdf` calls.

---

### 3.3 Base64 Images Stored Inside ChromaDB Metadata  ⚠️ High

**Where**: `summarizer.py` packs raw image bytes as base64 strings into `metadata["original_content"]`.

**Problem**: ChromaDB is a vector database — its metadata layer is not designed for large binary payloads. A single scanned-page image at moderate resolution is 50–200 KB as JPEG base64. A document with 30 images adds ~6 MB of metadata to the Chroma collection. Problems at scale:

- **Write amplification**: Every upsert serialises the full JSON blob into SQLite (Chroma's backing store).
- **Memory pressure**: When the collection is loaded, SQLite reads these blobs into memory. 1,000 documents with 30 images each = ~6 GB of just metadata in RAM.
- **Query overhead**: Chroma deserialises metadata for every returned document, even when the UI only needs 3 results.
- **No deduplication**: The same image appearing in multiple documents is stored multiple times.

---

### 3.4 Embedding Model Re-instantiated on Every Call  ⚠️ Medium

**Where**: `vector_store.py → _embedding_model()`.

**Problem**: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` downloads weights from HuggingFace Hub and loads them into RAM every time `create_vector_store()` or `load_vector_store()` is called. In the Streamlit app this happens every time the user processes a new batch (even on the same session). Loading a SentenceTransformer model takes 2–5 seconds and ~90 MB of RAM.

---

### 3.5 LLM Client Instantiated Per-Chunk  ⚠️ Medium

**Where**: `summarizer.py → create_ai_enhanced_summary()` — `ChatOpenAI(...)` is constructed inside the function.

**Problem**: The HTTP client (httpx under the hood) is created, configured, and GC'd for every single chunk. For 40 chunks, 40 client objects are created. This defeats connection pooling and adds per-call TLS handshake overhead.

---

### 3.6 In-Memory ChromaDB (No Persistence Between Sessions)  ⚠️ Medium

**Where**: `app.py → st.session_state.vector_db`.

**Problem**: The Streamlit app calls `create_vector_store()` which writes to `dbv2/chroma_db`, but the `vector_db` object in session state is that in-memory handle. When the browser tab closes or refreshes, the session state is lost — the user must re-upload and re-process all files. `load_vector_store()` exists in `vector_store.py` but is never called from `app.py`.

---

### 3.7 No Re-ranking — Top-K = 3 Flat Retrieval  ⚠️ Medium

**Where**: `retriever.py → retrieve_documents()`.

**Problem**: Cosine similarity retrieval with `all-MiniLM-L6-v2` (384-dim) can surface lexically-similar but semantically-wrong chunks. With `k=3` and no re-ranking, a false positive in the top-3 directly degrades answer quality. At scale (thousands of documents, millions of chunks), the HNSW approximate search also introduces recall degradation.

---

### 3.8 No Error Recovery or Idempotency in Ingestion  ⚠️ Medium

**Where**: `app.py` deduplication is filename-only; no chunk-level deduplication.

**Problem**: If ingestion fails mid-way (network error during summarization, API rate limit, crash), there is no checkpoint. The next run re-processes from the start. For a 500-page document that failed on chunk 300, you waste the work done on chunks 1–299. The filename deduplication in `st.session_state.processed_files` only guards against re-uploading the same filename in the same session — it offers no cross-session or cross-run protection.

---

### 3.9 No Query-Time Optimisation  ⚠️ Low–Medium

**Where**: `retriever.py`.

**Problem**: The query goes directly to the embedding model and HNSW without any of:
- **Query expansion** — the raw user query may use different vocabulary than the document's searchable summaries.
- **HyDE (Hypothetical Document Embeddings)** — generating a hypothetical answer and embedding it often outperforms embedding the raw question.
- **MMR (Maximal Marginal Relevance)** — top-3 results may all be from the same section of the same document, providing redundant context.

---

### 3.10 Free-Tier LLMs As Single Point of Failure  ⚠️ Structural

Both models (`nvidia/nemotron-nano-12b-v2-vl:free` and `google/gemma-3-12b-it:free`) are free OpenRouter endpoints. Free-tier models:

- Have unpredictable availability and rate limits.
- May be deprecated or removed without notice.
- Produce variable output quality.
- Lack SLA guarantees.

Using them as the only option means the entire pipeline fails silently when they are unavailable.

---

## 4. Proposed Optimal Architecture

The following architecture addresses every bottleneck above and is designed to handle hundreds of documents / millions of chunks without code rewrites.

### 4.1 Architecture Diagram (to-be)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PLANE                                    │
│                                                                              │
│  PDF Upload                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Object Storage  (S3 / MinIO / Azure Blob)              │                │
│  │  • Raw PDFs                                             │                │
│  │  • Extracted images (JPEG/WebP, keyed by hash)         │                │
│  └─────────────────────────┬───────────────────────────────┘                │
│                            │ triggers                                        │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Task Queue  (Celery + Redis / RQ / Cloud Tasks)        │                │
│  │                                                         │                │
│  │  Task: ingest_document(doc_id, s3_path)                │                │
│  │  ├── subtask: partition_pdf   (worker pool, hi_res)    │                │
│  │  ├── subtask: chunk_and_separate                       │                │
│  │  └── subtask: summarize_batch  (async, concurrent)     │                │
│  └─────────────────────────┬───────────────────────────────┘                │
│                            │                                                 │
│                            ▼                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Embedding Service  (dedicated, batched, cached)                     │   │
│  │  • Model: BGE-M3 (1024-dim, multilingual) or text-embedding-3-small  │   │
│  │  • Batch size: 64–256 chunks per forward pass                        │   │
│  │  • Cache: Redis / DiskCache — skip re-embedding unchanged docs       │   │
│  └───────────────────────┬──────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Vector Database  (Qdrant / Weaviate / Pinecone)                     │   │
│  │  • Vectors only — no binary payloads in metadata                     │   │
│  │  • Metadata: doc_id, chunk_id, page_num, content_types, image_keys   │   │
│  │  • Separate full-text index for BM25 hybrid retrieval                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY PLANE                                        │
│                                                                              │
│  User Query                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Query Processor                                                      │   │
│  │  ① HyDE — generate hypothetical answer, embed it                    │   │
│  │  ② Query expansion — synonyms / sub-questions                        │   │
│  │  ③ Embed original query                                              │   │
│  │  ④ Combine all query vectors (multi-vector retrieval)               │   │
│  └───────────────────────┬──────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Hybrid Retrieval  (dense + sparse)                                  │   │
│  │  • Vector search (HNSW) — top-50 candidates                         │   │
│  │  • BM25 keyword search — top-50 candidates                          │   │
│  │  • RRF / linear fusion — merge to top-20                            │   │
│  └───────────────────────┬──────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Re-ranker  (cross-encoder: BAAI/bge-reranker-v2-m3)                │   │
│  │  • Scores each of top-20 against the query                          │   │
│  │  • Returns top-5 re-ranked chunks                                   │   │
│  └───────────────────────┬──────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Context Assembler                                                   │   │
│  │  • Fetches raw text from DB                                         │   │
│  │  • Fetches images from Object Storage by key (lazy, on-demand)      │   │
│  │  • Builds multimodal prompt                                         │   │
│  └───────────────────────┬──────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM Gateway  (with fallback chain)                                  │   │
│  │  Primary: Claude claude-sonnet-4-6 / GPT-4o (paid, reliable)       │   │
│  │  Fallback: Gemma 3 12B / Nemotron (free, via OpenRouter)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           SERVING PLANE                                      │
│                                                                              │
│  ┌───────────────────┐    REST / WebSocket    ┌────────────────────────┐   │
│  │   FastAPI Backend  │◀─────────────────────▶│   Streamlit / Next.js  │   │
│  │   (stateless)      │                       │   Frontend              │   │
│  └───────────┬────────┘                       └────────────────────────┘   │
│              │                                                               │
│  ┌───────────▼────────┐                                                     │
│  │  PostgreSQL         │  ← document registry, job status, user sessions   │
│  └────────────────────┘                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Bottleneck-by-Bottleneck Solutions

### 5.1 Parallelise Summarization

**Solution**: Replace the serial `for` loop with `asyncio.gather` or a `ThreadPoolExecutor`. LangChain's `ChatOpenAI` supports async natively.

```python
# summarizer.py — improved
import asyncio
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(
    model=config.SUMMARIZER_MODEL,
    base_url=config.OPENROUTER_BASE_URL,
    temperature=0.4,
)  # ← singleton, built once, connection pool reused

async def _summarize_one(chunk_data: dict) -> str:
    ...
    response = await _llm.ainvoke([HumanMessage(content=message_content)])
    return response.content

async def summarize_chunks_async(chunks):
    tasks = [_summarize_one(separate_content_types(c)) for c in chunks]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

Add **exponential backoff + jitter** for rate-limit errors (HTTP 429):

```python
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(RateLimitError))
async def _summarize_one(chunk_data): ...
```

**Expected gain**: 10–30× throughput on a 40-chunk document, bounded by the API's concurrent rate limit rather than latency.

---

### 5.2 Parallelise PDF Partitioning

**Solution**: Use `ProcessPoolExecutor` (not threads — GIL-bound OCR is CPU-heavy) to partition multiple PDFs concurrently, up to `os.cpu_count()` workers.

```python
# pipeline.py
from concurrent.futures import ProcessPoolExecutor

def ingest_multiple_pdfs(pdf_paths: list[str]) -> list[Document]:
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        element_batches = list(pool.map(partition_document, pdf_paths))
    ...
```

For single very large PDFs, consider splitting the PDF at the page level (using `pypdf`) and partitioning page-batches in parallel, then merging.

---

### 5.3 Move Images Out of ChromaDB Metadata

**Solution**: Extract images to a content-addressed object store keyed by SHA-256 hash of the image bytes. Store only the key in Chroma metadata.

```python
# image_store.py
import hashlib, os

IMAGE_DIR = "content/images"  # swap for S3 bucket at scale

def store_image(base64_str: str) -> str:
    """Returns a stable key; deduplicates identical images."""
    img_bytes = base64.b64decode(base64_str)
    key = hashlib.sha256(img_bytes).hexdigest() + ".jpg"
    path = os.path.join(IMAGE_DIR, key)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(img_bytes)
    return key

def load_image(key: str) -> str:
    """Returns base64 on demand."""
    with open(os.path.join(IMAGE_DIR, key), "rb") as f:
        return base64.b64encode(f.read()).decode()
```

Metadata stored in Chroma becomes:
```json
{
  "raw_text": "...",
  "tables_html": ["<table>..."],
  "image_keys": ["a3f9...jpg", "b2e1...jpg"]
}
```

**Impact**: Chroma metadata shrinks from ~200 KB/chunk (with images) to ~2 KB. This alone reduces memory pressure by 100×.

---

### 5.4 Singleton Embedding Model + Batch Embed

**Solution**: Load the model once as a module-level singleton and use batch embedding.

```python
# vector_store.py
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _MODEL

def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embedding_model().encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).tolist()
```

Batch embedding runs the entire transformer forward pass over 64 chunks at once, fully utilising CPU SIMD / GPU matrix operations. At batch_size=64, throughput is typically 20–50× faster than calling `.encode()` per chunk.

---

### 5.5 Persist and Reload Vector Store in Streamlit

**Solution**: On startup, check if a Chroma collection already exists and load it. Only create a new one when documents are actually added.

```python
# app.py — session state init
if "vector_db" not in st.session_state:
    if os.path.exists(config.VECTOR_STORE_DIR):
        st.session_state.vector_db = load_vector_store()
    else:
        st.session_state.vector_db = None
```

This means refreshing the browser, restarting the app, or opening a second tab all resume from the same index.

---

### 5.6 Add a Re-ranking Stage

**Solution**: Retrieve top-20 with the bi-encoder, then score and re-rank with a cross-encoder. Cross-encoders jointly encode query + document and produce a relevance score far more accurate than cosine similarity.

```python
# retriever.py
from sentence_transformers import CrossEncoder

_RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3")

def retrieve_and_rerank(vector_store, query: str, fetch_k: int = 20, top_k: int = 5):
    candidates = vector_store.as_retriever(
        search_kwargs={"k": fetch_k}
    ).invoke(query)
    
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = _RERANKER.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

**Recommended model**: `BAAI/bge-reranker-v2-m3` — multilingual, ~280 MB, runs on CPU in ~50 ms for 20 candidates.

---

### 5.7 Hybrid Retrieval (Dense + BM25)

**Solution**: Run dense (vector) and sparse (BM25) retrieval in parallel and fuse with Reciprocal Rank Fusion (RRF). This recovers keyword-heavy queries that cosine similarity alone misses.

```python
# retriever.py
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store, documents: list[Document]):
        self.vector_store = vector_store
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = documents

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        dense_docs = self.vector_store.similarity_search(query, k=50)
        
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_top = sorted(
            enumerate(sparse_scores), key=lambda x: x[1], reverse=True
        )[:50]
        sparse_docs = [self.documents[i] for i, _ in sparse_top]
        
        return reciprocal_rank_fusion(dense_docs, sparse_docs)[:k]
```

---

### 5.8 Checkpoint-Based Ingestion (Idempotency)

**Solution**: Track ingestion progress in a lightweight SQLite/JSON state file keyed by `(doc_id, chunk_index)`. On failure/restart, skip already-completed chunks.

```python
# ingestion_state.py
import json, os

class IngestionCheckpoint:
    def __init__(self, path="ingestion_state.json"):
        self.path = path
        self.state = json.load(open(path)) if os.path.exists(path) else {}

    def is_done(self, doc_id: str, chunk_idx: int) -> bool:
        return self.state.get(doc_id, {}).get(str(chunk_idx), False)

    def mark_done(self, doc_id: str, chunk_idx: int):
        self.state.setdefault(doc_id, {})[str(chunk_idx)] = True
        json.dump(self.state, open(self.path, "w"))
```

---

### 5.9 LLM Fallback Chain

**Solution**: Define a priority list of models with automatic fallback on error/timeout.

```python
# llm_gateway.py
from langchain_openai import ChatOpenAI

MODEL_CHAIN = [
    {"model": "anthropic/claude-sonnet-4-6", "base_url": config.OPENROUTER_BASE_URL},
    {"model": "google/gemma-3-12b-it:free",  "base_url": config.OPENROUTER_BASE_URL},
    {"model": "nvidia/llama-3.1-nemotron-70b-instruct:free", "base_url": config.OPENROUTER_BASE_URL},
]

def get_llm_with_fallback(message_content, timeout=30):
    for cfg in MODEL_CHAIN:
        try:
            llm = ChatOpenAI(**cfg, temperature=0, timeout=timeout)
            return llm.invoke([HumanMessage(content=message_content)])
        except Exception as e:
            logger.warning(f"Model {cfg['model']} failed: {e}, trying next...")
    raise RuntimeError("All models failed")
```

---

### 5.10 Upgrade the Embedding Model

**Current**: `all-MiniLM-L6-v2` — 384-dim, English-optimised, 22M params, ~90 MB.

**Problem at scale**: 384 dimensions cause HNSW recall to degrade as the collection grows past ~100K chunks (curse of dimensionality at low dimension). English-only fails on multilingual documents.

**Recommended upgrade path**:

| Model | Dims | Size | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | Better than MiniLM on BEIR benchmarks, same speed |
| `BAAI/bge-base-en-v1.5` | 768 | 430 MB | Significant quality jump, ~2× slower |
| `BAAI/bge-m3` | 1024 | 570 MB | Multilingual, dense+sparse+multi-vector, best quality |
| `text-embedding-3-small` | 1536 | API | No local inference, $0.02/1M tokens |

For a project at 100K+ chunks, `bge-base-en-v1.5` gives the best quality-to-speed ratio on CPU.

---

## 6. Queue-Based Architecture for True Scale

For production ingestion of hundreds of documents per day, a synchronous pipeline inside Streamlit is unacceptable. The correct model is **background job processing**.

```
┌─────────────────┐      HTTP POST /ingest      ┌────────────────────┐
│  Frontend/API   │ ─────────────────────────▶ │  FastAPI Backend   │
│  (Streamlit /   │                             │  (stateless)       │
│   REST client)  │ ◀─────── job_id ──────────  └────────┬───────────┘
└─────────────────┘                                      │ enqueue
                                                         ▼
                                              ┌────────────────────┐
                                              │  Redis / RabbitMQ  │
                                              │  Task Queue        │
                                              └────────┬───────────┘
                                                       │ dequeue
                           ┌───────────────────────────┼──────────────────────────┐
                           │                           │                          │
                    ┌──────▼──────┐           ┌────────▼────┐           ┌─────────▼────┐
                    │  Worker 1   │           │  Worker 2   │           │  Worker N    │
                    │ (PDF parse) │           │ (Summarize) │           │  (Embed)     │
                    └──────┬──────┘           └────────┬────┘           └─────────┬────┘
                           │                           │                          │
                           └───────────────────────────▼──────────────────────────┘
                                                 ┌──────────┐
                                                 │  Qdrant  │
                                                 │  (vector │
                                                 │   store) │
                                                 └──────────┘

  Frontend polls GET /job/{job_id}/status  →  pending | processing | complete | failed
```

**Recommended stack**:
- **Queue**: Celery + Redis (battle-tested, great visibility)
- **Vector DB**: Qdrant (self-hosted Docker or Qdrant Cloud) — supports filtering, named vectors, payload indexing, and horizontal scaling natively
- **Object store**: MinIO (self-hosted S3-compatible) for images
- **Metadata DB**: PostgreSQL — document registry, job history, chunk metadata

---

## 7. Upgrade Priority Matrix

| # | Change | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | Async/parallel summarization + tenacity retry | 🔴 Critical | Low | **Do now** |
| 2 | Move images out of Chroma metadata → file/object store | 🔴 Critical | Low | **Do now** |
| 3 | Singleton embedding model + batch embedding | 🟠 High | Low | **Do now** |
| 4 | Persist & reload vector store in Streamlit (`load_vector_store`) | 🟠 High | Low | **Do now** |
| 5 | LLM client singleton (not per-chunk) | 🟡 Medium | Low | **Do now** |
| 6 | Checkpoint-based ingestion state | 🟠 High | Medium | Sprint 1 |
| 7 | Re-ranking with `bge-reranker-v2-m3` | 🟠 High | Medium | Sprint 1 |
| 8 | Upgrade embedding model to `bge-base-en-v1.5` | 🟡 Medium | Low | Sprint 1 |
| 9 | LLM fallback chain | 🟡 Medium | Medium | Sprint 1 |
| 10 | Hybrid retrieval (dense + BM25) | 🟠 High | Medium | Sprint 2 |
| 11 | ProcessPoolExecutor for multi-PDF partitioning | 🟠 High | Medium | Sprint 2 |
| 12 | Migrate from Chroma to Qdrant | 🟡 Medium | High | Sprint 2 |
| 13 | FastAPI backend + Celery queue | 🟠 High | High | Sprint 3 |
| 14 | HyDE query preprocessing | 🟡 Medium | Medium | Sprint 3 |
| 15 | Object storage (MinIO/S3) for images | 🟡 Medium | High | Sprint 3 |

---

## 8. Quick Wins (Code Changes Under 30 Minutes Each)

### 8.1 Use `load_vector_store` on App Startup

```python
# app.py — change session state init block
if "vector_db" not in st.session_state:
    if os.path.exists(config.VECTOR_STORE_DIR):
        st.session_state.vector_db = load_vector_store()  # resume existing index
    else:
        st.session_state.vector_db = None
```

### 8.2 Hoist LLM Clients to Module Level

```python
# summarizer.py — top of file, outside any function
_SUMMARIZER_LLM = ChatOpenAI(
    model=config.SUMMARIZER_MODEL,
    base_url=config.OPENROUTER_BASE_URL,
    temperature=0.4,
)

# retriever.py — top of file
_ANSWER_LLM = ChatOpenAI(
    model=config.ANSWER_MODEL,
    base_url=config.OPENROUTER_BASE_URL,
    temperature=0,
)
```

### 8.3 Async Batch Summarization

```python
# summarizer.py — add async path
async def summarize_chunks_async(chunks):
    sem = asyncio.Semaphore(10)  # max 10 concurrent API calls

    async def bounded(chunk):
        async with sem:
            return await _create_summary_async(chunk)

    return await asyncio.gather(*[bounded(c) for c in chunks])
```

### 8.4 MMR Retrieval (Built Into LangChain)

```python
# retriever.py — one-line change
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)
```

This fetches 20 candidates and returns 5 that are maximally diverse — no extra dependencies required.

### 8.5 Image Deduplication via SHA-256 Key

```python
# document_processor.py — in separate_content_types
import hashlib

def _image_key(base64_str: str) -> str:
    return hashlib.sha256(base64.b64decode(base64_str)).hexdigest()[:16]
```

Store only the key in metadata; write the actual image to disk once.

---

## 9. Summary

The current architecture is well-designed for a proof-of-concept. The dual-storage pattern (AI summary for embedding, raw content for answering) is architecturally sound and worth preserving. The main risks for scale are:

1. **Serial API calls** during summarization — fix with async concurrency.
2. **Binary blobs in vector DB metadata** — fix with a separate image store.
3. **Single-process, in-memory everything** — fix with background workers and a persistent vector DB like Qdrant.
4. **No re-ranking** — fix with a cross-encoder, which is the single highest-ROI retrieval quality improvement.
5. **No resilience** — fix with retry logic (tenacity), LLM fallback chains, and checkpoint-based ingestion.

The modular codebase structure makes all of these improvements surgical — each module has a single responsibility and can be upgraded independently without touching the others.
