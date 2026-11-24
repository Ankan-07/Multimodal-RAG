from typing import Any, Dict, List

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

import config


def partition_document(filepath: str) -> List[Any]:
    elements = partition_pdf(
        filename=filepath,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    return elements


def create_chunks_by_title(elements: List[Any]) -> List[Any]:
    return chunk_by_title(
        elements,
        max_characters=config.CHUNK_MAX_CHARACTERS,
        new_after_n_chars=config.CHUNK_NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER_N_CHARS,
    )


def separate_content_types(chunk: Any) -> Dict[str, Any]:
    # CHANGED: Keep image MIME types and page numbers. The visual embedder must
    # send the correct MIME type to Gemini, and source metadata needs the page.
    content_data: Dict[str, Any] = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "image_mime_types": [],
        "types": ["text"],
        "page_number": getattr(chunk.metadata, "page_number", 0) or 0,
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                # CHANGED: text_as_html can exist but be None. Fall back to the
                # element text so the table is not silently dropped.
                table_html = (
                    getattr(element.metadata, "text_as_html", None)
                    or element.text
                    or ""
                )
                if table_html:
                    content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    # CHANGED: Store the MIME type next to each Base64 image.
                    # Fall back to JPEG because unstructured often emits JPEG.
                    mime_type = getattr(
                        element.metadata, "image_mime_type", "image/jpeg"
                    ) or "image/jpeg"
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)
                    content_data["image_mime_types"].append(mime_type)

    content_data["types"] = list(set(content_data["types"]))
    return content_data
