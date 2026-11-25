# retriever.py

This module handles the query side: hybrid retrieval, fusion, answer
generation, and source extraction for the UI.

`retrieve_documents` queries three independent sources through the vector
store:

1. `query_text` — MiniLM text vectors (`text_chunks_minilm`), top 5.
2. `query_visual` — Gemini visual vectors (`visual_chunks_gemini`), top 5.
3. `query_bm25` — BM25L over text chunks and table text, top 5.

Each source is guarded by `_safe_query` so a failure in one source does not
stop the others. The three ranked lists are merged with Reciprocal Rank Fusion
(`_rrf_fuse`, `rrf_k=60`) and then grouped by `chunk_id` (`_merge_by_chunk`),
where the RRF scores of all records in the same chunk are summed. The result
is capped at `FINAL_TOP_K` (5) chunks. If the representative record does not
own the `original_content` context, it is resolved through
`vector_store.get_chunk_context(chunk_id)`.

`generate_final_answer` builds the multimodal prompt. `ChatOpenAI` is
constructed with `model=config.ANSWER_MODEL` (`openrouter/free`) and
`base_url=config.OPENROUTER_BASE_URL` (OpenRouter), `temperature=0`. For each
retrieved chunk it:

- appends the raw text (bounded by `MAX_PROMPT_CHARS`),
- appends table HTML (each table capped at `TABLE_CONTEXT_MAX_CHARS`),
- collects base64 images, deduplicated by hash, capped at
  `MAX_IMAGES_IN_CONTEXT` (3),
- attaches images as `data:<mime>;base64,...` image_url parts.

If no chunk remains after deduplication, it returns "I don't have enough
information to answer the question." Any exception during generation is
logged and a friendly error string is returned.

`extract_sources` deserializes each chunk's `original_content` and returns a
plain dict (`text`, `tables`, `images`, `image_mime_types`, `source`, `page`)
for the Streamlit source panel. It deduplicates by `chunk_id`.
