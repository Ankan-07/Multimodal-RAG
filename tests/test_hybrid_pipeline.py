"""NEW: Offline tests for BM25, RRF, and the hybrid store.

These tests use fake embedders. They do not call Gemini or OpenRouter.
"""

import base64
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image as PILImage

import ingestion
from bm25_store import BM25Index
from records import IngestionBatch, TextRecord, VisualRecord
from retriever import _merge_by_chunk, _rrf_fuse
from vector_store import HybridVectorStore


class FakeTextEmbedder:
    """Small deterministic stand-in for MiniLM."""

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0, 0.0]


class FakeVisualEmbedder:
    """Small deterministic stand-in for Gemini Embedding 2."""

    def embed_text(self, text):
        return [float(len(text)), 1.0, 0.0, 0.0, 0.0, 0.0]

    def embed_image_bytes(self, image_bytes, mime_type="image/png"):
        return [float(len(image_bytes)), 1.0, 0.0, 0.0, 0.0, 0.0]


def _png_base64(width=200, height=200):
    image = PILImage.new("RGB", (width, height), color="white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


# The names must match the unstructured class names because
# separate_content_types checks type(element).__name__.
class Table:
    def __init__(self):
        self.text = "Model BLEU 28.4"
        self.metadata = type(
            "Metadata",
            (),
            {"text_as_html": "<table><tr><td>Model</td><td>BLEU 28.4</td></tr></table>"},
        )()


class Image:
    def __init__(self):
        self.metadata = type(
            "Metadata",
            (),
            {
                "image_base64": _png_base64(),
                "image_mime_type": "image/png",
                "page_number": 1,
            },
        )()


class FakeChunk:
    def __init__(self):
        self.text = "The transformer architecture is shown in Figure 1."
        self.metadata = type(
            "Metadata",
            (),
            {
                "page_number": 1,
                "orig_elements": [Table(), Image()],
            },
        )()


class IngestionTests(unittest.TestCase):
    def test_build_records_separates_text_image_and_table(self):
        with patch.object(ingestion, "partition_document", return_value=[FakeChunk()]):
            with patch.object(ingestion, "create_chunks_by_title", return_value=[FakeChunk()]):
                batch = ingestion.build_ingestion_records("paper.pdf")

        self.assertEqual(len(batch.text_records), 1)
        self.assertEqual(len(batch.visual_records), 2)
        kinds = {record.kind for record in batch.visual_records}
        self.assertEqual(kinds, {"image", "table"})
        self.assertIn("BLEU 28.4", batch.visual_records[1].document)

        original = json.loads(batch.text_records[0].metadata["original_content"])
        self.assertEqual(original["image_mime_types"], ["image/png"])
        self.assertTrue(original["image_base64"])


class BM25Tests(unittest.TestCase):
    def test_number_and_keyword_search(self):
        index = BM25Index(
            [
                {
                    "id": "table-1",
                    "document": "Model BLEU score 28.4 on WMT 2014",
                    "metadata": {"kind": "table"},
                    "source": "table",
                },
                {
                    "id": "text-1",
                    "document": "The encoder has six layers",
                    "metadata": {"kind": "text"},
                    "source": "text",
                },
            ]
        )

        results = index.search("28.4 BLEU", 5)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "table-1")

    def test_empty_index(self):
        self.assertEqual(BM25Index([]).search("anything", 5), [])


class FusionTests(unittest.TestCase):
    def test_rrf_prefers_items_in_both_lists(self):
        text_hits = [
            {"id": "chunk-a", "metadata": {}, "document": "a", "source": "text_vector"},
            {"id": "chunk-b", "metadata": {}, "document": "b", "source": "text_vector"},
        ]
        bm25_hits = [
            {"id": "chunk-b", "metadata": {}, "document": "b", "source": "bm25"},
            {"id": "chunk-c", "metadata": {}, "document": "c", "source": "bm25"},
        ]

        fused = _rrf_fuse({"text_vector": text_hits, "bm25": bm25_hits}, rrf_k=60)
        self.assertEqual(fused[0]["id"], "chunk-b")
        self.assertIn("bm25", fused[0]["retrieval_sources"])

    def test_merge_keeps_one_record_per_chunk(self):
        candidates = [
            {
                "id": "text-1",
                "metadata": {"chunk_id": "chunk-1"},
                "document": "text",
                "rrf_score": 0.03,
                "retrieval_sources": ["text_vector"],
            },
            {
                "id": "image-1",
                "metadata": {"chunk_id": "chunk-1"},
                "document": "image",
                "rrf_score": 0.04,
                "retrieval_sources": ["visual_vector"],
            },
        ]

        merged = _merge_by_chunk(candidates)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "image-1")
        self.assertEqual(
            merged[0]["retrieval_sources"], ["text_vector", "visual_vector"]
        )
        # CHANGED: The chunk keeps the sum of both RRF scores, not only the
        # highest one, so multi-source evidence is not lost.
        self.assertAlmostEqual(merged[0]["rrf_score"], 0.07)


class HybridStoreTests(unittest.TestCase):
    def test_store_round_trip(self):
        # Chroma keeps the HNSW file open on Windows. Ignore cleanup errors so
        # the test process can exit without a file-lock failure.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store = HybridVectorStore(
                persist_directory=directory,
                text_embedder=FakeTextEmbedder(),
                visual_embedder=FakeVisualEmbedder(),
            )

            original_content = json.dumps(
                {
                    "raw_text": "Model BLEU score 28.4",
                    "tables_html": ["<table><tr><td>28.4</td></tr></table>"],
                    "image_base64": [],
                    "image_mime_types": [],
                    "source": "paper.pdf",
                    "page": 1,
                }
            )
            batch = IngestionBatch(
                text_records=[
                    TextRecord(
                        id="text:1",
                        text="Model BLEU score 28.4",
                        metadata={
                            "source": "paper.pdf",
                            "page": 1,
                            "chunk_id": "chunk-1",
                            "kind": "text",
                            "original_content": original_content,
                        },
                    )
                ],
                visual_records=[
                    VisualRecord(
                        id="table:1",
                        document="Model BLEU score 28.4",
                        metadata={
                            "source": "paper.pdf",
                            "page": 1,
                            "chunk_id": "chunk-1",
                            "kind": "table",
                            "original_content": original_content,
                        },
                        kind="table",
                    )
                ],
            )

            store.add_batch(batch)
            self.assertEqual(store.text_count, 1)
            self.assertEqual(store.visual_count, 1)
            self.assertTrue(store.query_text("BLEU", 3))
            self.assertTrue(store.query_visual("BLEU", 3))
            self.assertTrue(store.query_bm25("28.4", 3))

    def test_context_is_stored_once_and_resolved(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store = HybridVectorStore(
                persist_directory=directory,
                text_embedder=FakeTextEmbedder(),
                visual_embedder=FakeVisualEmbedder(),
            )
            context = json.dumps(
                {
                    "raw_text": "Model BLEU score 28.4",
                    "tables_html": [],
                    "image_base64": [],
                    "image_mime_types": [],
                    "source": "paper.pdf",
                    "page": 1,
                }
            )
            # The text record owns the context. The image record has only the
            # chunk_id and must resolve the context through the store.
            batch = IngestionBatch(
                text_records=[
                    TextRecord(
                        id="text:2",
                        text="Model BLEU score 28.4",
                        metadata={
                            "source": "paper.pdf",
                            "page": 1,
                            "chunk_id": "chunk-2",
                            "kind": "text",
                            "original_content": context,
                        },
                    )
                ],
                visual_records=[
                    VisualRecord(
                        id="image:2",
                        document="Image from paper.pdf",
                        metadata={
                            "source": "paper.pdf",
                            "page": 1,
                            "chunk_id": "chunk-2",
                            "kind": "image",
                        },
                        kind="image",
                        image_bytes=b"image-bytes",
                        mime_type="image/png",
                    )
                ],
            )

            store.add_batch(batch)
            self.assertEqual(store.get_chunk_context("chunk-2"), context)
            store.delete_source("paper.pdf")
            self.assertEqual(store.text_count, 0)
            self.assertEqual(store.visual_count, 0)
            self.assertIsNone(store.get_chunk_context("chunk-2"))


if __name__ == "__main__":
    unittest.main()
