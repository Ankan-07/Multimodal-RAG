import os
from dotenv import load_dotenv

load_dotenv()

# Existing OpenRouter configuration. The answer model and the optional
# summarizer still use this key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Legacy summarizer setting. The new ingestion path does not call it.
SUMMARIZER_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
ANSWER_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"

# NEW: Text embeddings stay on the local Sentence Transformer model.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# NEW: Gemini Embedding 2 embeds images and table text into one vector space.
# The key is read from .env. GOOGLE_API_KEY is accepted as a fallback.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Treat the template value as missing so the app gives a clear configuration
# error instead of sending a fake key to Google.
if GEMINI_API_KEY and GEMINI_API_KEY.strip().lower().startswith("your_"):
    GEMINI_API_KEY = None
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_DIM = 768

# NEW: A separate database path prevents old MiniLM-only records from mixing
# with the new two-collection design.
VECTOR_STORE_DIR = "dbv3/chroma_db"
TEXT_COLLECTION = "text_chunks_minilm"
VISUAL_COLLECTION = "visual_chunks_gemini"
CONTENT_DIR = "content"

CHUNK_MAX_CHARACTERS = 3000
CHUNK_NEW_AFTER_N_CHARS = 2500
CHUNK_COMBINE_UNDER_N_CHARS = 500

# NEW: Retrieval settings. Each source returns a candidate list. RRF merges
# the lists. The final list is capped before it goes to the answer model.
TEXT_TOP_K = 5
VISUAL_TOP_K = 5
BM25_TOP_K = 5
FINAL_TOP_K = 5
RRF_K = 60

# NEW: BM25 parameters and an image context cap. Images consume many tokens.
BM25_K1 = 1.5
BM25_B = 0.75
MAX_IMAGES_IN_CONTEXT = 3

# NEW: Ignore very small PDF images such as icons and rules.
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# NEW: Table text can be longer than the embedding limit. Store the full text
# for BM25 and answer generation, but embed only this prefix.
TABLE_EMBED_MAX_CHARS = 7000

# NEW: Gemini request retry count for temporary API or network failures.
GEMINI_REQUEST_RETRIES = 3
