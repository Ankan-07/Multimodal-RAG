# pipeline.py

This module is the CLI orchestrator. It composes the ingestion and retrieval
modules into a single command-line workflow.

`ingest_pdf` calls `build_ingestion_records(pdf_path)` (from `ingestion.py`).
That function partitions the PDF with `unstructured` `hi_res`, chunks the
elements by title, and returns an `IngestionBatch` of text records (MiniLM)
and visual records (images and tables, Gemini). No vision-LLM summarization
happens.

`run_ingestion` creates the hybrid store with `create_vector_store()`,
deletes existing records for the same source file with `delete_source()`, and
adds the new batch with `add_batch()`. Deleting first means a re-ingested,
changed PDF does not leave stale text/image/table records behind.

`run_query` calls `retrieve_documents(store, query)` (hybrid RRF retrieval)
and `generate_final_answer(chunks, query)`, returning both the answer string
and the retrieved documents so callers can display sources.

The `__main__` block exposes a CLI: `--pdf` is required; `--query` is
optional. Without a query, the script only ingests. Useful for batch
processing, testing, or CI without the Streamlit UI.
