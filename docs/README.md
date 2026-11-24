# Documentation Index

The current architecture is a hybrid retrieval design:

- MiniLM text vectors in `text_chunks_minilm`
- Gemini Embedding 2 image/table vectors in `visual_chunks_gemini`
- BM25L keyword index over text chunks and table text
- RRF fusion across text, visual, and BM25 candidates
- No reranker by default

See the root `README.md` and these modules for the current implementation:

- `ingestion.py`
- `vector_store.py`
- `retriever.py`
- `gemini_embeddings.py`
- `bm25_store.py`

The other files in this `docs/` folder describe the older summary-based
pipeline. They are kept as historical notes and are not a description of the
current runtime path.
