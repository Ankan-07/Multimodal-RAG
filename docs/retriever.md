# retriever.py

This module handles everything on the query side of the system: finding the right chunks, generating an answer from them, and extracting the source material for display.

`retrieve_documents` is a thin wrapper around ChromaDB's retriever interface. We use LangChain's `.as_retriever()` with `k=3` — meaning we fetch the top 3 most semantically similar chunks. Three was chosen as a balance: enough context for the LLM to synthesize a complete answer, few enough that we don't blow past the model's context window or dilute the relevant signal with loosely related chunks.

`generate_final_answer` is where the system becomes truly multimodal at query time. The answer prompt is built in two passes over the retrieved chunks. First we construct the text portion: we unpack each chunk's `original_content` metadata (which holds the raw text and table HTML, separate from the AI-enhanced summary we stored in `page_content`) and build a structured prompt. We deliberately use the original raw text here rather than the enhanced summary — the summary was optimized for retrieval, but the raw content is what the LLM should reason over for factual accuracy. Then in a second pass, we attach any base64 images as vision blocks. This means the answer model sees the full original text, the structured table data, and the actual images — everything it needs to give a grounded, accurate response.

`extract_sources` is a utility for the UI layer. It deserializes the same `original_content` metadata and returns the raw text, tables, and images in a plain dict structure so the Streamlit app can render them without needing to know anything about the internal metadata format.
