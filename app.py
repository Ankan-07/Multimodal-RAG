import base64
import os

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


def display_base64_image(base64_str: str):
    st.image(base64.b64decode(base64_str))


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
                st.markdown(table, unsafe_allow_html=True)
        if source["images"]:
            st.markdown("**Images Found:**")
            for img in source["images"]:
                display_base64_image(img)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

with st.sidebar:
    st.title(" Document Hub")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents") and uploaded_files:
        with st.status(" Processing Documents...", expanded=True) as status:
            # CHANGED: Build one ingestion batch per file. The batch contains
            # MiniLM text records and Gemini image/table records.
            new_batches = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    st.write(f"Processing {uploaded_file.name}...")
                    os.makedirs(config.CONTENT_DIR, exist_ok=True)
                    temp_path = os.path.join(config.CONTENT_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    try:
                        batch = build_ingestion_records(temp_path)
                        new_batches.append((uploaded_file.name, batch))
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
                    db = st.session_state.vector_db or create_vector_store()
                    for filename, batch in new_batches:
                        db.add_batch(batch)
                        st.session_state.processed_files.append(filename)
                    st.session_state.vector_db = db
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
        st.success(
            f"Database active with {len(st.session_state.processed_files)} files "
            f"({db.text_count} text, {db.visual_count} visual records)"
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
