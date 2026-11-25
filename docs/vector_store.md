# vector_store.py

This module implements the hybrid store (`HybridVectorStore`) that backs both
the Streamlit app and the CLI pipeline.

The store keeps two Chroma collections because the two embedding models use
different vector spaces and dimensions:

- `text_chunks_minilm` — MiniLM vectors (384-d), `hnsw:space: cosine`.
- `visual_chunks_gemini` — Gemini Embedding 2 vectors (768-d), cosine.

Text embeddings use `HuggingFaceEmbeddings` from `langchain-huggingface`
with `all-MiniLM-L6-v2`. The model runs locally and loads once per store
creation; in the app the store is cached per process with
`@st.cache_resource`, so the model is not reloaded per upload.

The visual embedder (`GeminiEmbedder`) is lazily created on first use, so a
text-only store opens without a `GEMINI_API_KEY`.

`add_text_records` embeds chunk text with MiniLM and upserts into the text
collection. `add_visual_records` embeds each image (bytes) and each table
(text, capped at `TABLE_EMBED_MAX_CHARS` for embedding) with Gemini and
upserts into the visual collection. It returns the number of records that
could not be embedded; a total failure of all visual records raises a
`RuntimeError`.

`add_batches` writes several batches and rebuilds the BM25 index once in a
`finally` block, so a visual failure cannot leave BM25 stale for the text
records already written.

`query_text`, `query_visual`, and `query_bm25` are the three retrieval entry
points used by `retriever.py` (top-k per source). All query methods share
`_query_collection`, which normalizes the Chroma response into hit dicts with
a cosine-derived score (`1 - distance`).

`rebuild_bm25` reconstructs the BM25 index from the persisted Chroma records
(text chunks plus table records), so the keyword index stays in sync after an
application restart.

`get_chunk_context` returns the `original_content` JSON for a `chunk_id`
(stored once per chunk, on the first record). `delete_source` removes every
record for one source and rebuilds BM25; the app calls it before re-ingesting
a file.
