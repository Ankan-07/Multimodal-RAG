"""Legacy summary-based ingestion.

The current pipeline does not import this module. Images and tables are now
embedded directly with Gemini Embedding 2. Keep this file only as a reference
for the old searchable-summary approach.
"""

import json
import logging
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config
from document_processor import separate_content_types

logger = logging.getLogger(__name__)


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    llm = ChatOpenAI(
        model=config.SUMMARIZER_MODEL,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0.4,
    )

    prompt_text = (
        f"You are creating a searchable description for document content retrieval.\n\n"
        f"Content to analyze:\ntext content: {text}\n"
    )

    if tables:
        prompt_text += "tables\n"
        for i, table in enumerate(tables):
            prompt_text += f"Table {i+1}:\n{table}\n\n"

    prompt_text += """
Your task:
Generate a comprehensive, searchable description that covers:
1. key facts, numbers and data point from the tables.
2. main topics and concepts discussed.
3. questions this content could answer.
4. visual content analysis [charts, diagrams, images]
5. alternative search terms users might use

Make it searchable and detailed. Prioritize findability over brevity.

SEARCHABLE DESCRIPTION:
"""

    message_content: List[dict] = [{"type": "text", "text": prompt_text}]
    for image_base64 in images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        })

    response = llm.invoke([HumanMessage(content=message_content)])
    return response.content if response.content else text


def summarize_chunks(chunks: List[Any]) -> List[Document]:
    langchain_documents = []

    for chunk in chunks:
        content_data = separate_content_types(chunk)

        if content_data["tables"] or content_data["images"]:
            try:
                enhanced_summary = create_ai_enhanced_summary(
                    content_data["text"],
                    content_data["tables"],
                    content_data["images"],
                )
                enhanced_summary = enhanced_summary or content_data["text"]
            except Exception as e:
                logger.warning(f"AI summary failed, using raw text: {e}")
                enhanced_summary = content_data["text"]
        else:
            enhanced_summary = content_data["text"]

        doc = Document(
            page_content=enhanced_summary,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    "image_base64": content_data["images"],
                })
            },
        )
        langchain_documents.append(doc)

    return langchain_documents
