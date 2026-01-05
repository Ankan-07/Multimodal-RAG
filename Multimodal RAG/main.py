import os
import json
import logging
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def partition_document(filepath: str) -> List[Any]:
    """Partitions a PDF document into elements."""
    logger.info(f"Partitioning document: {filepath}")
    try:
        elements = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        logger.info(f"Extracted {len(elements)} elements")
        return elements
    except Exception as e:
        logger.error(f"Error partitioning document: {e}")
        raise

def create_chunks_by_title(elements: List[Any]) -> List[Any]:
    """Chunks elements by title."""
    logger.info("Creating smart chunks")
    try:
        chunks = chunk_by_title(
            elements,
            max_characters=3000,
            new_after_n_chars=2500,
            combine_text_under_n_chars=500
        )
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking elements: {e}")
        raise

def separate_content_types(chunk: Any) -> Dict[str, Any]:
    """Extracts text, tables, and images from a chunk."""
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"]
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Generates an AI-enhanced summary for a chunk."""
    try:
        llm = ChatOpenAI(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            base_url="https://openrouter.ai/api/v1",
            temperature=0.4
        )

        prompt_text = f"You are creating a searchable description for document content retrieval.\n\nContent to analyze:\ntext content: {text}\n"

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

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content if response.content else text

    except Exception as e:
        logger.warning(f"AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [contains {len(tables)} table(s)]"
        if images:
            summary += f" [contains {len(images)} image(s)]"
        return summary

def summarize_chunks(chunks: List[Any]) -> List[Document]:
    """Processes chunks and generates AI summaries."""
    logger.info("Processing chunks with AI summaries...")
    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{total_chunks}")
        content_data = separate_content_types(chunk)

        if content_data["tables"] or content_data["images"]:
            enhanced_summary = create_ai_enhanced_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"]
            )
        else:
            enhanced_summary = content_data["text"]

        doc = Document(
            page_content=enhanced_summary,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data["text"],
                    "tables_html": content_data["tables"],
                    "image_base64": content_data["images"]
                })
            }
        )
        langchain_documents.append(doc)

    logger.info(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents

def create_vector_store(documents: List[Document], persist_directory: str = "dbv2/chroma_db") -> Chroma:
    """Creates a vector store from documents."""
    logger.info(f"Creating vector store at {persist_directory}")
    try:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.info("Finished creating vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def export_chunks_to_json(chunks: List[Document], filename: str = "chunks_export.json"):
    """Exports retrieved chunks to a JSON file."""
    export_data = []
    for i, doc in enumerate(chunks):
        export_data.append({
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        })
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Exported {len(export_data)} chunks to {filename}")

def generate_final_answer(chunks: List[Document], query: str) -> str:
    """Generates a final answer using the retrieved chunks."""
    try:
        llm = ChatOpenAI(
            model="google/gemma-3-12b-it:free",
            base_url="https://openrouter.ai/api/v1",
            temperature=0
        )

        prompt_text = f"Based on the following documents, answer the question: {query}\nCONTENT TO ANALYZE:\n"

        for i, chunk in enumerate(chunks):
            prompt_text += f"-----Document {i+1}-----\n"
            original_data = json.loads(chunk.metadata.get('original_content', '{}'))
            
            raw_text = original_data.get("raw_text", "")
            if raw_text:
                prompt_text += f"TEXT:\n{raw_text}\n\n"

            tables_html = original_data.get("tables_html", [])
            if tables_html:
                prompt_text += "TABLES: "
                for j, table in enumerate(tables_html):
                    prompt_text += f"table {j+1}: {table}\n"
            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, images above. If the document doesn't contain sufficient answer to the question, 
just say "I don't have enough information to answer the question."

ANSWER:
"""
        message_content = [{"type": "text", "text": prompt_text}]

        for chunk in chunks:
            original_data = json.loads(chunk.metadata.get("original_content", "{}"))
            images_base64 = original_data.get("image_base64", [])
            for image_base64 in images_base64:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception:
        logger.exception("Answer generation failed")
        return "Sorry, I encountered an error while generating the answer. Please check the logs for details."

def save_qna(query: str, answer: str, chunks: List[Document], filename: str = "qna.json"):
    """Saves the Q&A pair and its sources to a JSON file."""
    sources = []
    for doc in chunks:
        original_data = json.loads(doc.metadata.get("original_content", "{}"))
        sources.append({
            "text": original_data.get("raw_text", ""),
            "tables": original_data.get("tables_html", []),
            "images": [f"Image logic: base64 present ({len(img)} chars)" for img in original_data.get("image_base64", [])]
        })
    
    qna_data = {
        "question": query,
        "answer": answer,
        "sources": sources
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qna_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Q&A to {filename}")

def run_pipeline(pdf_path: str, query: str = None):
    """Runs the full RAG pipeline."""
    logger.info("Starting RAG pipeline")
    
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarized_chunks = summarize_chunks(chunks)
    db = create_vector_store(summarized_chunks)

    if query:
        logger.info(f"Processing query: {query}")
        retriever = db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)
        
        export_chunks_to_json(retrieved_docs, "rag_result.json")
        answer = generate_final_answer(retrieved_docs, query)
        save_qna(query, answer, retrieved_docs)
        
        print("\n--- FINAL ANSWER ---")
        print(answer)
        print("--------------------")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--query", type=str, help="Question to ask the RAG system")
    args = parser.parse_args()

    run_pipeline(args.pdf, args.query)
