# Multimodal RAG: Hybrid Text, Image, and Table Retrieval

A multimodal Retrieval-Augmented Generation (RAG) system for PDFs. Text chunks
use a local Sentence Transformer. Images and tables use Gemini Embedding 2.
BM25 adds keyword and number matching. Results are merged with Reciprocal Rank
Fusion (RRF) and sent to a vision-capable answer model.

## Key Features

- **Two vector spaces**: MiniLM text vectors and Gemini visual vectors stay in
  separate Chroma collections.
- **Direct image retrieval**: Images are embedded as images, not as generated
  summaries.
- **Direct table retrieval**: Table text is embedded with Gemini and indexed by
  BM25 for exact names and numbers.
- **Hybrid ranking**: Text vectors, visual vectors, and BM25 results are fused
  with RRF.
- **No reranker by default**: The final context is capped at five chunks and
  three images.
- **Source attribution**: The UI shows source text, tables, and images.
- **Streamlit UI and CLI**: Use the chat interface or the command-line pipeline.

## Tech Stack

- **PDF processing**: Unstructured (`hi_res`, table and image extraction)
- **Text embeddings**: Sentence Transformers via `langchain-huggingface`
  (`all-MiniLM-L6-v2`, 384-d)
- **Visual embeddings**: Google Gemini Embedding 2 (`gemini-embedding-2`, 768-d)
- **Keyword search**: BM25 (`rank-bm25`, BM25L)
- **Vector database**: ChromaDB, two cosine collections
- **Fusion**: Reciprocal Rank Fusion (RRF)
- **Answer model**: Vision-capable model through OpenRouter
  (`openrouter/free`)
- **Frontend**: Streamlit
- **Package manager**: uv

## Installation

1. Clone the repository and enter the project directory.

   Unstructured `hi_res` PDF parsing needs **Poppler** and **Tesseract** on
   your `PATH`. Install them before the first ingestion.

2. Create `.env` from the template:

   ```env
   OPENAI_API_KEY=your_openrouter_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

   `OPENAI_API_KEY` is used for the OpenRouter answer model.
   `GEMINI_API_KEY` is used for image and table embeddings. `GOOGLE_API_KEY`
   is accepted as a fallback. A placeholder value that starts with `your_` is
   treated as missing.

   A text-only store can open without a Gemini key. Image and table ingestion
   and visual queries need the key.

3. Install dependencies:

   ```bash
   uv sync
   ```

## Usage

### Streamlit UI

```bash
uv run streamlit run app.py
```

Upload PDFs in the sidebar, process them, and ask questions in the chat.

### CLI

```bash
uv run python pipeline.py --pdf "./content/attention.pdf" --query "What is the transformer architecture?"
```

## How It Works

1. **Partition**: Unstructured extracts text, tables, and images from the PDF.
2. **Chunk**: Text is grouped by title.
3. **Text records**: Raw text chunks are embedded with `all-MiniLM-L6-v2` and
   stored in `text_chunks_minilm`.
4. **Visual records**: Each image is embedded with Gemini Embedding 2. Each
   table is converted to text, embedded with Gemini, and stored in
   `visual_chunks_gemini`.
5. **BM25**: Text chunks and table text are indexed for keyword and number
   matching.
6. **Retrieve**: The query is embedded by both models. Each source returns its
   top candidates. BM25 returns its top keyword candidates.
7. **Fuse**: RRF merges the three ranked lists. Results are merged by parent
   chunk and capped.
8. **Answer**: The answer model receives raw text, table HTML, and up to three
   images.

## Project Structure

```text
├── app.py                  # Streamlit UI
├── pipeline.py             # CLI pipeline
├── ingestion.py            # Build text and visual records
├── records.py              # Record dataclasses
├── document_processor.py   # PDF partition and content separation
├── gemini_embeddings.py    # Gemini Embedding 2 client
├── bm25_store.py           # BM25 index
├── vector_store.py         # Two Chroma collections + BM25
├── retriever.py            # Hybrid retrieval, RRF, and answer generation
├── tests/                  # Offline unit tests
├── content/                # Uploaded PDFs (ignored by Git)
├── dbv3/                   # Chroma persistence (ignored by Git)
├── pyproject.toml
└── README.md
```

## Notes

- The old `dbv2/chroma_db` store is not used. The new store is
  `dbv3/chroma_db`.
- Re-ingest documents because the collection layout and embedding models
  changed.
- Re-uploading a PDF replaces the old records for the same stored source
  before it adds the new ones. The app adds a short content hash to the stored
  file name, so two PDFs with the same name do not collide.
- Very small images (under 100×100) and duplicate images inside one chunk are
  skipped. A duplicate image in a different chunk is kept.
- The answer prompt caps each table at 4000 characters and the whole prompt at
  24000 characters. Full table text stays in the store for retrieval.
- A reranker is not part of the default pipeline. Add one only if retrieval
  evaluation shows good Recall@10 but poor Precision@3.
- The notebook `pipeline.ipynb` contains the old summary-based flow. Use
  `pipeline.py` and `app.py` for the current architecture.

## License

MIT License.
