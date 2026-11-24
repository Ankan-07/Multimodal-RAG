"""NEW: End-to-end ingestion and query pipeline for the hybrid store."""

import argparse
import logging
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from ingestion import build_ingestion_records
from records import IngestionBatch
from retriever import generate_final_answer, retrieve_documents
from vector_store import HybridVectorStore, create_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: str) -> IngestionBatch:
    """Build text and visual records from one PDF. No vision LLM call."""
    logger.info("Ingesting: %s", pdf_path)
    return build_ingestion_records(pdf_path)


def run_ingestion(
    pdf_path: str, persist_directory: Optional[str] = None
) -> HybridVectorStore:
    """Create the hybrid store and add one document batch."""
    batch = ingest_pdf(pdf_path)
    db = create_vector_store(persist_directory)
    # CHANGED: Replace old records for the same source before adding. A changed
    # PDF must not leave stale text, image, or table records behind.
    db.delete_source(os.path.basename(pdf_path))
    db.add_batch(batch)
    logger.info("Ingestion complete")
    return db


def run_query(
    vector_store: HybridVectorStore, query: str
) -> Tuple[str, List[Document]]:
    """Retrieve fused candidates and generate the final answer."""
    docs = retrieve_documents(vector_store, query)
    answer = generate_final_answer(docs, query)
    return answer, docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    args = parser.parse_args()

    db = run_ingestion(args.pdf)

    if args.query:
        answer, _ = run_query(db, args.query)
        print("\n--- ANSWER ---")
        print(answer)
        print("--------------")
