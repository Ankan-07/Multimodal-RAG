# pipeline.py

This module is the top-level orchestrator. Its job is to compose the lower-level modules into complete workflows so that calling code — whether it's the Streamlit app or a CLI script — doesn't need to know the internal sequence of steps.

`ingest_pdf` runs the three-stage ingestion sequence: partition the PDF into structured elements, chunk those elements by title, then generate AI-enhanced summaries. It returns a list of LangChain `Document` objects. Separating this from `run_ingestion` was a deliberate choice — the Streamlit app needs to collect documents from multiple files before building a single shared vector store. If we fused document processing and vector store creation into one function, multi-file ingestion would create one store per file and throw everything except the last one away.

`run_ingestion` is the convenience function for single-file workflows or the CLI case. It calls `ingest_pdf` then immediately creates a vector store from the result.

`run_query` takes an already-built vector store and a question, then returns both the answer string and the retrieved documents. Returning the documents alongside the answer lets the caller display sources or do further processing without needing to re-run retrieval.

The `__main__` block exposes a CLI interface using `argparse`. This is useful for batch processing, testing, or running the pipeline from a CI job without the full Streamlit UI. You pass `--pdf` for the file path and `--query` for an optional question to ask immediately after ingestion.
