"""NEW: Hybrid retrieval with Reciprocal Rank Fusion.

The retriever queries three candidate sources:
1. MiniLM text vectors
2. Gemini visual vectors
3. BM25 over text chunks and table text

The three ranked lists are fused with RRF. No reranker is used. The final list
is merged by chunk ID and capped before it goes to the answer model.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config

logger = logging.getLogger(__name__)


def _safe_query(
    function: Any, query: str, top_k: int, source_name: str
) -> List[Dict[str, Any]]:
    """Run one retrieval source. A failure must not stop the other sources."""
    try:
        return function(query, top_k)
    except Exception as exc:
        logger.warning("%s retrieval failed: %s", source_name, exc)
        return []


def _rrf_fuse(
    result_lists: Dict[str, List[Dict[str, Any]]],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Raw cosine scores from MiniLM and Gemini are not comparable. RRF uses only
    rank, so it is safe across different embedding models.
    """
    fused: Dict[str, Dict[str, Any]] = {}

    for source_name, results in result_lists.items():
        for rank, item in enumerate(results, start=1):
            record_id = item.get("id")
            if not record_id:
                continue

            if record_id not in fused:
                fused[record_id] = {
                    "item": item,
                    "rrf_score": 0.0,
                    "retrieval_sources": set(),
                }

            fused[record_id]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[record_id]["retrieval_sources"].add(source_name)

    ranked = []
    for entry in fused.values():
        item = dict(entry["item"])
        item["rrf_score"] = entry["rrf_score"]
        item["retrieval_sources"] = sorted(entry["retrieval_sources"])
        ranked.append(item)

    ranked.sort(key=lambda item: item["rrf_score"], reverse=True)
    return ranked


def _merge_by_chunk(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one representative record per parent chunk.

    A text hit and an image hit can point to the same chunk. They share the
    same parent chunk, so one representative is enough for the answer model.

    CHANGED: Add the RRF scores of every record in the chunk. A chunk found by
    all three sources must rank above a chunk found once, which is the point of
    RRF. The representative is still the record with the highest single score.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        group_id = metadata.get("chunk_id") or candidate.get("id")
        candidate_score = float(candidate.get("rrf_score", 0.0))

        if group_id not in groups:
            groups[group_id] = {
                "representative": dict(candidate),
                "rrf_score": candidate_score,
                "retrieval_sources": set(candidate.get("retrieval_sources", [])),
            }
            continue

        group = groups[group_id]
        group["rrf_score"] += candidate_score
        group["retrieval_sources"].update(candidate.get("retrieval_sources", []))
        if candidate_score > float(
            group["representative"].get("rrf_score", 0.0)
        ):
            group["representative"] = dict(candidate)

    merged: List[Dict[str, Any]] = []
    for group in groups.values():
        item = dict(group["representative"])
        item["rrf_score"] = group["rrf_score"]
        item["retrieval_sources"] = sorted(group["retrieval_sources"])
        merged.append(item)

    merged.sort(key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    return merged


def retrieve_documents(vector_store: Any, query: str) -> List[Document]:
    """Retrieve and fuse text, visual, and BM25 candidates."""
    text_hits = _safe_query(
        vector_store.query_text, query, config.TEXT_TOP_K, "text_vector"
    )
    visual_hits = _safe_query(
        vector_store.query_visual, query, config.VISUAL_TOP_K, "visual_vector"
    )
    bm25_hits = _safe_query(
        vector_store.query_bm25, query, config.BM25_TOP_K, "bm25"
    )

    fused = _rrf_fuse(
        {
            "text_vector": text_hits,
            "visual_vector": visual_hits,
            "bm25": bm25_hits,
        },
        rrf_k=config.RRF_K,
    )
    merged = _merge_by_chunk(fused)[: config.FINAL_TOP_K]

    documents: List[Document] = []
    for item in merged:
        metadata = dict(item.get("metadata") or {})

        # CHANGED: The context is stored once per chunk. If this record is not
        # the owner, fetch the context from the store by chunk_id.
        if not metadata.get("original_content"):
            context = vector_store.get_chunk_context(
                str(metadata.get("chunk_id", ""))
            )
            if context:
                metadata["original_content"] = context

        # Keep retrieval diagnostics in metadata. This helps tune RRF weights
        # and inspect why a chunk was selected.
        metadata["rrf_score"] = float(item.get("rrf_score", 0.0))
        metadata["retrieval_sources"] = ",".join(
            item.get("retrieval_sources", [])
        )
        documents.append(
            Document(
                page_content=item.get("document", ""),
                metadata=metadata,
            )
        )
    return documents


def _parse_original_content(chunk: Document) -> Dict[str, Any]:
    """Read the JSON context string that every record stores."""
    try:
        return json.loads(chunk.metadata.get("original_content", "{}"))
    except (TypeError, json.JSONDecodeError):
        logger.warning("Could not parse original_content for %s", chunk.metadata)
        return {}


def _image_hash(image_base64: str) -> str:
    """Deduplicate images that appear in more than one retrieved chunk."""
    return hashlib.sha1(image_base64.encode("ascii", errors="ignore")).hexdigest()


def generate_final_answer(chunks: List[Document], query: str) -> str:
    """Generate the final answer from raw text, tables, and images."""
    try:
        llm = ChatOpenAI(
            model=config.ANSWER_MODEL,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=0,
        )

        prompt_text = (
            f"Based on the following documents, answer the question: {query}\n"
            "CONTENT TO ANALYZE:\n"
        )

        seen_chunks: set[str] = set()
        images: List[tuple[str, str]] = []
        seen_images: set[str] = set()

        for i, chunk in enumerate(chunks):
            original_data = _parse_original_content(chunk)
            chunk_key = (
                chunk.metadata.get("chunk_id")
                or original_data.get("source", "")
                + str(original_data.get("page", ""))
            )
            if chunk_key in seen_chunks:
                continue
            seen_chunks.add(str(chunk_key))

            prompt_text += f"-----Document {i + 1}-----\n"
            raw_text = original_data.get("raw_text", "")
            if raw_text:
                # CHANGED: Keep the prompt inside a fixed character budget.
                remaining = config.MAX_PROMPT_CHARS - len(prompt_text)
                if remaining > 0:
                    prompt_text += f"TEXT:\n{raw_text[:remaining]}\n\n"

            tables_html = original_data.get("tables_html", [])
            if tables_html:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(tables_html):
                    remaining = config.MAX_PROMPT_CHARS - len(prompt_text)
                    if remaining <= 0:
                        break
                    # CHANGED: Cap each table and the total prompt so a large
                    # table cannot truncate the question or the instructions.
                    table_text = str(table)[: config.TABLE_CONTEXT_MAX_CHARS]
                    prompt_text += f"table {j + 1}: {table_text[:remaining]}\n"
            prompt_text += "\n"

            mime_types = original_data.get("image_mime_types", [])
            for image_index, image_base64 in enumerate(
                original_data.get("image_base64", [])
            ):
                image_key = _image_hash(image_base64)
                if image_key in seen_images:
                    continue
                seen_images.add(image_key)
                mime_type = (
                    mime_types[image_index]
                    if image_index < len(mime_types)
                    else "image/jpeg"
                )
                images.append((image_base64, mime_type))

        if not seen_chunks:
            return "I don't have enough information to answer the question."

        # Cap image count. Images consume many context tokens.
        images = images[: config.MAX_IMAGES_IN_CONTEXT]
        prompt_text += (
            "Please provide a clear, comprehensive answer using the text, "
            "tables, and images above. If the documents do not contain a "
            'sufficient answer, say "I don\'t have enough information to '
            'answer the question."\n\nANSWER:\n'
        )

        message_content: List[dict] = [{"type": "text", "text": prompt_text}]
        for image_base64, mime_type in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                }
            )

        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content

    except Exception:
        logger.exception("Answer generation failed")
        return (
            "Sorry, I encountered an error while generating the answer. "
            "Please check the logs for details."
        )


def extract_sources(chunks: List[Document]) -> List[dict]:
    """Return unique source contexts for the UI."""
    sources: List[dict] = []
    seen_chunks: set[str] = set()

    for doc in chunks:
        original_data = _parse_original_content(doc)
        chunk_key = doc.metadata.get("chunk_id") or (
            original_data.get("source", "")
            + str(original_data.get("page", ""))
        )
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(str(chunk_key))

        sources.append(
            {
                "text": original_data.get("raw_text", ""),
                "tables": original_data.get("tables_html", []),
                "images": original_data.get("image_base64", []),
                "image_mime_types": original_data.get("image_mime_types", []),
                "source": original_data.get("source", ""),
                "page": original_data.get("page", 0),
            }
        )
    return sources
