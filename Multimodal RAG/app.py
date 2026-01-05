import streamlit as st
import os
import json
import base64
from main import run_pipeline, partition_document, create_chunks_by_title, summarize_chunks, create_vector_store, generate_final_answer, save_qna
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Page Config
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Minimalist Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: #161b22;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #30363d;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Source box styling */
    .source-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to display images from base64
def display_base64_image(base64_str):
    st.image(base64.b64decode(base64_str))

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Sidebar
with st.sidebar:
    st.title("📂 Document Hub")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.status("🚀 Processing Documents...", expanded=True) as status:
            all_docs = []
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    st.write(f"Processing {uploaded_file.name}...")
                    
                    # Save file temporarily
                    temp_path = os.path.join("content", uploaded_file.name)
                    os.makedirs("content", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # RAG Pipeline
                    elements = partition_document(temp_path)
                    chunks = create_chunks_by_title(elements)
                    summarized_chunks = summarize_chunks(chunks)
                    all_docs.extend(summarized_chunks)
                    
                    st.session_state.processed_files.append(uploaded_file.name)
            
            if all_docs:
                st.session_state.vector_db = create_vector_store(all_docs)
                status.update(label="✅ All documents processed!", state="complete", expanded=False)
            else:
                status.update(label="ℹ️ No new documents to process", state="complete", expanded=False)

    st.divider()
    st.markdown("### Status")
    if st.session_state.vector_db:
        st.success(f"Database active with {len(st.session_state.processed_files)} files")
    else:
        st.info("Upload and process files to begin")

# Main Chat Interface
st.title("🧠 Multimodal RAG Chatbot")
st.markdown("Ask anything about your uploaded documents. I can see text, tables, and images!")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 View Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1} (Text):**")
                    st.text(source["text"][:500] + "...")
                    
                    if source["tables"]:
                        st.markdown("**Tables Found:**")
                        for table in source["tables"]:
                            st.markdown(table, unsafe_allow_html=True)
                    
                    if source["images"]:
                        st.markdown("**Images Found:**")
                        for img in source["images"]:
                            display_base64_image(img)

# Chat Input
if query := st.chat_input("What is the transformer architecture?"):
    if not st.session_state.vector_db:
        st.error("Please process some documents first!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Reflecting on documents..."):
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(query)
                answer = generate_final_answer(retrieved_docs, query)
                
                # Format sources for display
                sources = []
                for doc in retrieved_docs:
                    original_data = json.loads(doc.metadata.get("original_content", "{}"))
                    sources.append({
                        "text": original_data.get("raw_text", ""),
                        "tables": original_data.get("tables_html", []),
                        "images": original_data.get("image_base64", [])
                    })
                
                st.markdown(answer)
                
                with st.expander("🔍 View Sources"):
                    for idx, source in enumerate(sources):
                        st.markdown(f"**Source {idx+1} (Text):**")
                        st.text(source["text"][:500] + "...")
                        
                        if source["tables"]:
                            st.markdown("**Tables Found:**")
                            for table in source["tables"]:
                                st.markdown(table, unsafe_allow_html=True)
                        
                        if source["images"]:
                            st.markdown("**Images Found:**")
                            for img in source["images"]:
                                display_base64_image(img)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Save Q&A to file (background)
                save_qna(query, answer, retrieved_docs)
