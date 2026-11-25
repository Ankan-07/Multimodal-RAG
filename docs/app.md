# app.py

This is the Streamlit front-end — the file a user interacts with. It wires
the pipeline modules into a conversational interface.

Layout: a sidebar for document management and a main area for chat, with a
dark GitHub-inspired theme applied via inline CSS.

Store lifecycle: `_open_store()` is decorated with `@st.cache_resource`, so
the persisted hybrid store (`dbv3/chroma_db`) is opened once per server
process. A fresh browser session resumes the persisted index without
re-uploading PDFs. On any store-open failure, the error is captured in
`st.session_state.store_error` and shown in the sidebar.

Sidebar: users upload one or more PDFs. On "Process Documents", for each new
file (skipping names already in `st.session_state.processed_files`):

1. The file bytes are read and hashed (SHA-1, 8 hex chars); the file is saved
   as `content/<stem>_<hash>.pdf` so same-named PDFs do not collide and a
   crafted name cannot escape `content/`.
2. `build_ingestion_records(path)` builds the text/image/table batch.
3. Old records for the same stored source are removed with
   `db.delete_source(stored_name)`.
4. `db.add_batches([...])` embeds and upserts all records, then rebuilds the
   BM25 index once.

Visual failures are counted and reported as a warning; a total visual failure
shows the processing error. The status widget ends at "✅ All documents
processed!" or "❌ Processing failed".

Chat: the main area shows message history. On a submitted question, the app
calls `retrieve_documents` → `generate_final_answer` → `extract_sources`, and
renders the answer plus a "View Sources" expander. `render_sources` shows the
source file/page, text, sanitized tables (active tags/event attributes are
stripped for HTML safety), and images (base64 decoded via `st.image`).
Assistant messages store their `sources` in session state so source panels
survive rerenders.
