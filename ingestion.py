"""NEW: Build text and visual records from a PDF.

This module replaces the old summarizer-based ingestion path. Text chunks keep
their raw text. Images and tables become separate visual records so each one
can be retrieved on its own.
"""

import base64
import hashlib
import html
import io
import json
import logging
import os
import re
from typing import List, Optional, Tuple

from PIL import Image

import config
from document_processor import (
    create_chunks_by_title,
    partition_document,
    separate_content_types,
)
from records import IngestionBatch, TextRecord, VisualRecord

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}


def _sha1(value: bytes) -> str:
    """Return a stable short ID for a byte value or text value."""
    return hashlib.sha1(value).hexdigest()[:16]


def _chunk_id(source: str, page: int, text: str, index: int) -> str:
    """Build a deterministic chunk ID. Re-ingestion can upsert the same IDs."""
    raw = f"{source}|{page}|{index}|{text}".encode("utf-8", errors="ignore")
    return _sha1(raw)


def _table_html_to_text(table_html: str) -> str:
    """Convert a simple table HTML string to text for BM25 and Gemini.

    This is deliberately lightweight. It keeps cell and row boundaries without
    adding BeautifulSoup as a runtime dependency.
    """
    if not table_html:
        return ""

    text = re.sub(r"(?i)</t[dh]>", " | ", table_html)
    text = re.sub(r"(?i)</tr>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip(" |")
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _prepare_image(
    image_base64: str, mime_type: str
) -> Optional[Tuple[str, str, bytes]]:
    """Decode, validate, and normalize one image.

    Gemini Embedding 2 accepts PNG and JPEG. Other image formats are converted
    to PNG. Very small images are dropped because they are usually icons.
    """
    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        logger.warning("Skipping image with invalid Base64 data")
        return None

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
                logger.info("Skipping small image: %sx%s", width, height)
                return None

            # CHANGED: Trust the format that Pillow detects, not the metadata
            # MIME type. A mislabeled BMP must not be sent to Gemini as PNG.
            detected_format = (image.format or "").upper()
            detected_mime = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
            }.get(detected_format)

            if detected_mime in SUPPORTED_IMAGE_MIME_TYPES:
                return image_base64, detected_mime, image_bytes

            # Convert unsupported or unknown formats to PNG for Gemini and the
            # answer model.
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
            converted = output.getvalue()
            return base64.b64encode(converted).decode("ascii"), "image/png", converted
    except Exception as exc:
        logger.warning("Skipping unreadable image: %s", exc)
        return None


def _build_original_content(
    text: str,
    tables: List[str],
    images: List[str],
    image_mime_types: List[str],
    source: str,
    page: int,
) -> str:
    """Serialize the answer-time context into one Chroma metadata string."""
    return json.dumps(
        {
            "raw_text": text,
            "tables_html": tables,
            "image_base64": images,
            "image_mime_types": image_mime_types,
            "source": source,
            "page": page,
        }
    )


def build_ingestion_records(pdf_path: str) -> IngestionBatch:
    """Partition one PDF and return text and visual records.

    No vision LLM is called here. Images are embedded directly later by the
    Gemini embedder. Tables are converted to text for Gemini and BM25.
    """
    source = os.path.basename(pdf_path)
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)

    batch = IngestionBatch()

    for index, chunk in enumerate(chunks):
        content = separate_content_types(chunk)
        page = int(content.get("page_number") or 0)
        chunk_id = _chunk_id(source, page, content["text"], index)

        # CHANGED: Deduplicate images inside one chunk, not across the whole
        # document. A duplicate image must stay with every chunk that contains
        # it, otherwise it is lost when the first chunk is not retrieved.
        valid_images: List[Tuple[str, str, bytes]] = []
        seen_images: set[str] = set()
        for image_base64, mime_type in zip(
            content["images"], content["image_mime_types"]
        ):
            prepared = _prepare_image(image_base64, mime_type)
            if prepared is None:
                continue
            normalized_base64, normalized_mime, image_bytes = prepared
            image_hash = _sha1(image_bytes)
            if image_hash in seen_images:
                continue
            seen_images.add(image_hash)
            valid_images.append((normalized_base64, normalized_mime, image_bytes))

        image_base64_values = [item[0] for item in valid_images]
        image_mime_values = [item[1] for item in valid_images]

        original_content = _build_original_content(
            text=content["text"],
            tables=content["tables"],
            images=image_base64_values,
            image_mime_types=image_mime_values,
            source=source,
            page=page,
        )

        # CHANGED: Store the answer-time context once per chunk. The first
        # record for the chunk owns it; later records store only chunk_id. This
        # stops every image and table record from duplicating all Base64 image
        # data in its metadata.
        context_written = False

        # Text stays on MiniLM. Skip an empty text record when the chunk
        # contains only a table or image.
        if content["text"].strip():
            metadata = {
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "kind": "text",
            }
            if not context_written:
                metadata["original_content"] = original_content
                context_written = True
            batch.text_records.append(
                TextRecord(
                    id=f"text:{chunk_id}",
                    text=content["text"],
                    metadata=metadata,
                )
            )

        # One visual record for each image. Do not aggregate several images
        # into one Gemini vector.
        for image_index, (image_base64, mime_type, image_bytes) in enumerate(
            valid_images
        ):
            image_hash = _sha1(image_bytes)
            metadata = {
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "kind": "image",
                "image_index": image_index,
                "image_hash": image_hash,
            }
            if not context_written:
                metadata["original_content"] = original_content
                context_written = True
            batch.visual_records.append(
                VisualRecord(
                    id=f"image:{chunk_id}:{image_hash}",
                    document=f"Image from {source}, page {page}",
                    kind="image",
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    metadata=metadata,
                )
            )

        # One visual record for each table. The full text is kept for BM25 and
        # answer generation. The embedding input may be shorter.
        for table_index, table_html in enumerate(content["tables"]):
            table_text = _table_html_to_text(table_html)
            if not table_text:
                continue
            metadata = {
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "kind": "table",
                "table_index": table_index,
            }
            if not context_written:
                metadata["original_content"] = original_content
                context_written = True
            batch.visual_records.append(
                VisualRecord(
                    id=f"table:{chunk_id}:{table_index}",
                    document=table_text,
                    kind="table",
                    embedding_text=table_text[: config.TABLE_EMBED_MAX_CHARS],
                    metadata=metadata,
                )
            )

    logger.info(
        "Built %s text records and %s visual records from %s",
        len(batch.text_records),
        len(batch.visual_records),
        source,
    )
    return batch
