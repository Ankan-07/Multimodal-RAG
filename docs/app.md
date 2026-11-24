# app.py

This is the Streamlit front-end. It's the only file a user directly interacts with, and its job is to wire the pipeline modules together into a conversational interface.

The UI is split into two regions: a sidebar for document management and a main area for the chat.

In the sidebar, users upload one or more PDFs. When they click "Process Documents", we iterate over the uploaded files, skip any that have already been processed (tracked in `st.session_state.processed_files`), save each to a temp path under the `content/` directory, and run the full document processing pipeline on it. All the resulting documents are collected into a single list before calling `create_vector_store` once at the end. This is intentional — a single vector store across all uploaded files means a query can retrieve relevant chunks from any document rather than being siloed per file.

The `render_sources` helper is extracted as a function because the exact same source display logic is needed in two places: when replaying chat history on page load, and when rendering a fresh assistant response. Extracting it removes the duplication and keeps the main render loop readable.

Session state manages three things: the chat message history, the active vector store, and the list of processed file names. The message history includes the `sources` list on assistant messages so that source panels are preserved when the page rerenders or the user scrolls up through the conversation.

The custom CSS at the top applies a dark GitHub-inspired color scheme. This is purely cosmetic but matters for hosting — a polished UI makes the demo more credible when sharing a Streamlit Cloud URL. The theme is defined inline rather than via a `config.toml` so the file remains self-contained with no extra configuration needed at the host level.
