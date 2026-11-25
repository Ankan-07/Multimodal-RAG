# config.py

This file is the single source of truth for every configurable value in the
system. Nothing outside it should hardcode a model name, path, or numeric
hyperparameter.

Environment: `.env` is loaded via `python-dotenv`. `OPENAI_API_KEY` holds the
OpenRouter key (named to satisfy LangChain's default `ChatOpenAI`
expectation); `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) holds the Gemini key. A
placeholder value starting with `your_` is treated as missing.

Models:

- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` — local text embeddings (384-d),
  used through `langchain-huggingface` `HuggingFaceEmbeddings`.
- `GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"` — image/table embeddings
  (768-d), used by `GeminiEmbedder`.
- `ANSWER_MODEL = "openrouter/free"` — answer generation via OpenRouter's free
  model router (vision-capable).

Storage: `VECTOR_STORE_DIR = "dbv3/chroma_db"`, with collections
`text_chunks_minilm` and `visual_chunks_gemini`. `CONTENT_DIR = "content"`.

Chunking: `CHUNK_MAX_CHARACTERS = 3000`, `CHUNK_NEW_AFTER_N_CHARS = 2500`,
`CHUNK_COMBINE_UNDER_N_CHARS = 500`.

Retrieval: `TEXT_TOP_K = VISUAL_TOP_K = BM25_TOP_K = 5`, `FINAL_TOP_K = 5`,
`RRF_K = 60`, `BM25_K1 = 1.5`, `BM25_B = 0.75`.

Prompt/context caps: `MAX_IMAGES_IN_CONTEXT = 3`,
`TABLE_EMBED_MAX_CHARS = 7000`, `TABLE_CONTEXT_MAX_CHARS = 4000`,
`MAX_PROMPT_CHARS = 24000`.

Images: `MIN_IMAGE_WIDTH = MIN_IMAGE_HEIGHT = 100`.

Gemini retry: `GEMINI_REQUEST_RETRIES = 3` (408/429/5xx and network errors,
exponential backoff 1s → 16s).

Startup: the repo-local `.env` also sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` (see `.env.example`) to skip HuggingFace online
checks and load the cached MiniLM model quickly.
