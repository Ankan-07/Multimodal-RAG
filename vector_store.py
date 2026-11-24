"""NEW: Hybrid vector store for MiniLM text vectors and Gemini visual vectors.

The store keeps two Chroma collections because the two embedding models use
different vector spaces and different dimensions. It also builds a BM25 index
over text chunks and table text for keyword and number matching.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings

import config
from bm25_store import BM25Index
from gemini_embeddings import GeminiEmbedder
from records import IngestionBatch, TextRecord, VisualRecord

logger = logging.getLogger(__name__)


class HybridVectorStore:
    """Two vector collections plus one BM25 index."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        text_embedder: Optional[Any] = None,
        visual_embedder: Optional[Any] = None,
    ) -> None:
        self.persist_directory = persist_directory or config.VECTOR_STORE_DIR
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # NEW: Each collection has one vector space. Do not mix 384-d MiniLM
        # vectors with 768-d Gemini vectors in one collection.
        self.text_collection = self.client.get_or_create_collection(
            name=config.TEXT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self.visual_collection = self.client.get_or_create_collection(
            name=config.VISUAL_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # NEW: The text model stays local. The visual model is lazy; it does not
        # call Google until the first image or visual query arrives.
        self.text_embedder = text_embedder or SentenceTransformerEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        self.visual_embedder = visual_embedder or GeminiEmbedder()

        # NEW: BM25 is rebuilt from persisted Chroma records. This keeps the
        # keyword index in sync after an application restart.
        self.bm25 = BM25Index([])
        self.rebuild_bm25()

    @property
    def text_count(self) -> int:
        return self.text_collection.count()

    @property
    def visual_count(self) -> int:
        return self.visual_collection.count()

    def add_text_records(self, records: List[TextRecord]) -> None:
        """Embed and upsert text chunks with MiniLM."""
        if not records:
            return

        texts = [record.text for record in records]
        embeddings = self.text_embedder.embed_documents(texts)
        self.text_collection.upsert(
            ids=[record.id for record in records],
            embeddings=embeddings,
            documents=texts,
            metadatas=[record.metadata for record in records],
        )
        logger.info("Upserted %s text records", len(records))

    def add_visual_records(self, records: List[VisualRecord]) -> None:
        """Embed and upsert images and tables with Gemini Embedding 2."""
        if not records:
            return

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        failures = 0

        for record in records:
            try:
                if record.kind == "image":
                    if record.image_bytes is None:
                        raise ValueError("Image record has no image bytes")
                    vector = self.visual_embedder.embed_image_bytes(
                        record.image_bytes,
                        record.mime_type or "image/png",
                    )
                else:
                    vector = self.visual_embedder.embed_text(
                        record.embedding_text or record.document
                    )
            except Exception as exc:
                failures += 1
                logger.warning("Skipping visual record %s: %s", record.id, exc)
                continue

            ids.append(record.id)
            embeddings.append(vector)
            documents.append(record.document)
            metadatas.append(record.metadata)

        if ids:
            self.visual_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Upserted %s visual records", len(ids))

        # A missing key or a total API failure must not look like a successful
        # visual ingestion with zero records.
        if failures and not ids:
            raise RuntimeError(
                "Gemini visual embedding failed for every record. "
                "Check GEMINI_API_KEY and the API error logs."
            )

    def add_batch(self, batch: IngestionBatch) -> None:
        """Add one document batch and refresh the BM25 index."""
        self.add_text_records(batch.text_records)
        self.add_visual_records(batch.visual_records)
        self.rebuild_bm25()

    def _query_collection(
        self,
        collection: Any,
        query_vector: List[float],
        top_k: int,
        retrieval_source: str,
    ) -> List[Dict[str, Any]]:
        """Run one vector query and normalize the Chroma response."""
        if top_k <= 0 or collection.count() == 0:
            return []

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for index, record_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            hits.append(
                {
                    "id": record_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    # Chroma cosine distance is 1 - cosine similarity.
                    "score": 1.0 - float(distance) if distance is not None else 0.0,
                    "source": retrieval_source,
                }
            )
        return hits

    def query_text(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query the MiniLM text collection."""
        vector = self.text_embedder.embed_query(query)
        return self._query_collection(
            self.text_collection, vector, top_k, "text_vector"
        )

    def query_visual(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query the Gemini image and table collection with a text query."""
        vector = self.visual_embedder.embed_text(query)
        return self._query_collection(
            self.visual_collection, vector, top_k, "visual_vector"
        )

    def query_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query the BM25 index over text chunks and table text."""
        return self.bm25.search(query, top_k)

    def rebuild_bm25(self) -> None:
        """Collect text and table records from Chroma and rebuild BM25.

        Images are excluded. Their labels do not contain useful keyword data.
        """
        documents: List[Dict[str, Any]] = []

        text_data = self.text_collection.get(include=["documents", "metadatas"])
        for index, record_id in enumerate(text_data.get("ids", [])):
            documents.append(
                {
                    "id": record_id,
                    "document": text_data["documents"][index] or "",
                    "metadata": text_data["metadatas"][index] or {},
                    "source": "text",
                }
            )

        table_data = self.visual_collection.get(
            where={"kind": "table"},
            include=["documents", "metadatas"],
        )
        for index, record_id in enumerate(table_data.get("ids", [])):
            documents.append(
                {
                    "id": record_id,
                    "document": table_data["documents"][index] or "",
                    "metadata": table_data["metadatas"][index] or {},
                    "source": "table",
                }
            )

        self.bm25 = BM25Index(documents)
        logger.info("BM25 index rebuilt with %s documents", self.bm25.size)

    def delete_source(self, source: str) -> None:
        """Delete all records for one source before re-ingestion."""
        self.text_collection.delete(where={"source": source})
        self.visual_collection.delete(where={"source": source})
        self.rebuild_bm25()


def create_vector_store(
    persist_directory: Optional[str] = None,
    text_embedder: Optional[Any] = None,
    visual_embedder: Optional[Any] = None,
) -> HybridVectorStore:
    """Create or open the hybrid store.

    This keeps the old function name used by the application, but it now
    returns a HybridVectorStore instead of a single Chroma object.
    """
    return HybridVectorStore(
        persist_directory=persist_directory,
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
    )


def load_vector_store(
    persist_directory: Optional[str] = None,
    text_embedder: Optional[Any] = None,
    visual_embedder: Optional[Any] = None,
) -> HybridVectorStore:
    """Open an existing hybrid store and rebuild its BM25 index."""
    return HybridVectorStore(
        persist_directory=persist_directory,
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
    )
