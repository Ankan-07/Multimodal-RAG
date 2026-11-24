import base64
import hashlib
import os
import re

import streamlit as st

import config
from ingestion import build_ingestion_records
from retriever import extract_sources, generate_final_answer, retrieve_documents
from vector_store import create_vector_store

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stChatMessage { background-color: #161b22; border-radius: 10px; padding: 15px; margin-bottom: 10px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .source-box { background-color: #0d1117; border: 1px solid #30363d; border-radius: 5px; padding: 10px; margin-top: 10px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)


# CHANGED: Remove active content and event handlers from table HTML before it
# is rendered. Uploaded PDFs are untrusted input.
_ACTIVE_TAG_RE = re.compile(
    r"(?is)<(script|style|iframe|object|embed)[^>]*>.*?</\1>"
)
_EVENT_ATTR_RE = re.compile(r"(?i)\son\w+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)")


def _safe_table_html(table_html: str) -> str:
    cleaned = _ACTIVE_TAG_RE.sub("", str(table_html))
    return _EVENT_ATTR_RE.sub("", cleaned)


def display_base64_image(base64_str: str):
    # CHANGED: A corrupt image must not stop the whole source panel.
    try:
        st.image(base64.b64decode(base64_str))
    except Exception as exc:
        st.warning(f"Could not display image: {exc}")


def render_sources(sources: list):
    for idx, source in enumerate(sources):
        # CHANGED: Show the source file and page for easier verification.
        location = source.get("source", "unknown")
        page = source.get("page", 0)
        st.markdown(f"**Source {idx + 1} ({location}, page {page}):**")
        st.text(source["text"][:500] + "...")
        if source["tables"]:
            st.markdown("**Tables Found:**")
            for table in source["tables"]:
                st.markdown(_safe_table_html(table), unsafe_allow_html=True)
        if source["images"]:
            st.markdown("**Images Found:**")
            for img in source["images"]:
                display_base64_image(img)


@st.cache_resource(show_spinner=False)
def _open_store():
    """Open the persisted store once per server process.

    The visual embedder is lazy, so this works without a Gemini key until an
    image or table is actually queried.
    """
    return create_vector_store()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "store_error" not in st.session_state:
    st.session_state.store_error = ""
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# CHANGED: Reopen the persisted store on a fresh session. Without this, a
# restart ignored dbv3/chroma_db and asked the user to upload the PDFs again.
if st.session_state.vector_db is None and not st.session_state.store_error:
    try:
        st.session_state.vector_db = _open_store()
    except Exception as exc:
        st.session_state.store_error = str(exc)

with st.sidebar:
    st.title(" Document Hub")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents") and uploaded_files:
        with st.status(" Processing Documents...", expanded=True) as status:
            # CHANGED: Build one ingestion batch per file. The batch contains
            # MiniLM text records and Gemini image/table records.
            new_batches = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.processed_files:
                    continue

                st.write(f"Processing {uploaded_file.name}...")

                # CHANGED: Use only the base name and add a content hash. A
                # crafted name cannot escape content/, and two PDFs with the
                # same name cannot overwrite each other.
                safe_name = os.path.basename(uploaded_file.name)
                if not safe_name.lower().endswith(".pdf"):
                    st.error(f"Skipping {uploaded_file.name}: not a PDF")
                    continue

                try:
                    file_bytes = uploaded_file.read()
                    digest = hashlib.sha1(file_bytes).hexdigest()[:8]
                    stem, _ = os.path.splitext(safe_name)
                    stored_name = f"{stem}_{digest}.pdf"
                    os.makedirs(config.CONTENT_DIR, exist_ok=True)
                    temp_path = os.path.join(config.CONTENT_DIR, stored_name)
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)

                    batch = build_ingestion_records(temp_path)
                    new_batches.append((uploaded_file.name, stored_name, batch))
                    st.write(
                        f"Built {len(batch.text_records)} text records and "
                        f"{len(batch.visual_records)} visual records."
                    )
                except Exception as exc:
                    st.error(f"Could not process {uploaded_file.name}: {exc}")

            if new_batches:
                # CHANGED: Open the existing hybrid store once, then add every
                # new batch. The store embeds text with MiniLM and visuals with
                # Gemini Embedding 2.
                try:
                    db = st.session_state.vector_db or _open_store()

                    # CHANGED: Replace old records for the same source before
                    # adding the new ones. Then rebuild BM25 once for the whole
                    # upload instead of once per file.
                    for _, stored_name, _ in new_batches:
                        db.delete_source(stored_name)
                    failures = db.add_batches([batch for _, _, batch in new_batches])

                    for display_name, _, _ in new_batches:
                        st.session_state.processed_files.append(display_name)
                    st.session_state.vector_db = db
                    st.session_state.store_error = ""
                    if failures:
                        st.warning(
                            f"{failures} visual records could not be embedded. "
                            "Check GEMINI_API_KEY and the logs."
                        )
                    status.update(
                        label="✅ All documents processed!",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    st.error(f"Could not build the vector store: {exc}")
                    status.update(
                        label="❌ Processing failed",
                        state="error",
                        expanded=False,
                    )
            else:
                status.update(
                    label="ℹ️ No new documents to process",
                    state="complete",
                    expanded=False,
                )

    st.divider()
    st.markdown("### Status")
    if st.session_state.vector_db:
        # CHANGED: Show both collections so text and visual ingestion are visible.
        db = st.session_state.vector_db
        if db.text_count or db.visual_count:
            st.success(
                f"Database active with {db.text_count} text and "
                f"{db.visual_count} visual records"
            )
        else:
            st.info("Upload and process files to begin")
    elif st.session_state.store_error:
        st.error(
            "Could not open the vector store: "
            f"{st.session_state.store_error}"
        )
    else:
        st.info("Upload and process files to begin")

st.title("🧠 Multimodal RAG Chatbot")
st.markdown("Ask anything about your uploaded documents. I can see text, tables, and images!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander(" View Sources"):
                render_sources(message["sources"])

if query := st.chat_input("What is the transformer architecture?"):
    if not st.session_state.vector_db:
        st.error("Please process some documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Reflecting on documents..."):
                retrieved_docs = retrieve_documents(st.session_state.vector_db, query)
                answer = generate_final_answer(retrieved_docs, query)
                sources = extract_sources(retrieved_docs)

                st.markdown(answer)
                with st.expander(" View Sources"):
                    render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
