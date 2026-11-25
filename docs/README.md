# Documentation Index

The current architecture is a hybrid retrieval design:

- MiniLM text vectors in `text_chunks_minilm`
- Gemini Embedding 2 image/table vectors in `visual_chunks_gemini`
- BM25L keyword index over text chunks and table text
- RRF fusion across text, visual, and BM25 candidates
- No reranker by default
- Answer model: `openrouter/free` via OpenRouter

See the root `README.md` and these docs for the current implementation:

- `app.md` — Streamlit UI
- `config.md` — configuration and model names
- `document_processor.md` — PDF partition and content separation
- `pipeline.md` — CLI pipeline
- `retriever.md` — hybrid retrieval, RRF, and answer generation
- `vector_store.md` — two Chroma collections + BM25
- `ARCHITECTURE_ANALYSIS.md` — architecture analysis, bottlenecks, roadmap
