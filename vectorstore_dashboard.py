import streamlit as st
import os
import shutil
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
import config

def build_vectorstore_streamlit(doc_path, db_path, collection_name):
    st.write(f"📁 Indexing documents from: `{doc_path}`")
    st.write(f"📁 Saving to ChromaDB at: `{db_path}`")

    # Clear old DB
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        st.warning(f"Cleared existing ChromaDB at `{db_path}`")

    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDINGS_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    # Load PDFs
    loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    st.success(f"Loaded `{len(docs)}` PDF pages.")

    # Add progress for splitting
    for i, doc in enumerate(docs):
        doc.metadata["page"] = doc.metadata.get("page", i + 1)

    split_docs = splitter.split_documents(docs)
    st.info(f"Split into `{len(split_docs)}` chunks.")

    # Progress bar for indexing
    progress_bar = st.progress(0)
    step = 1 / max(1, len(split_docs))
    for i in range(len(split_docs)):
        progress_bar.progress(min(1.0, step * (i + 1)))

    Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=db_path, collection_name=collection_name)
    st.success(f"Indexing complete for `{collection_name}`!")

# Streamlit UI
st.title("📚 VectorStore Builder Dashboard")

option = st.selectbox("Choose a document set to index:", ["4007", "4100",  "Requirements", "Existing Test Cases", "All",])

if st.button("Start Indexing"):
    if option == "4007":
        build_vectorstore_streamlit(config.DOCS_4007_PATH, config.CHROMA_DB_4007_PATH, "4007")
    elif option == "4100":
        build_vectorstore_streamlit(config.DOCS_4100_PATH, config.CHROMA_DB_4100_PATH, "4100")
    elif option == "All":
        with st.spinner("Indexing 4007..."):
            build_vectorstore_streamlit(config.DOCS_4007_PATH, config.CHROMA_DB_4007_PATH, "4007")
        with st.spinner("Indexing 4100..."):
            build_vectorstore_streamlit(config.DOCS_4100_PATH, config.CHROMA_DB_4100_PATH, "4100")
    elif option == "Requirements":
      build_vectorstore_streamlit(
        config.REQUIREMENTS_DOCS_PATH,
        config.CHROMA_DB_REQUIREMENTS_PATH,
        "requirements",
    )
    elif option == "Existing Test Cases":
     build_vectorstore_streamlit(
        config.TESTCASES_DOCS_PATH,
        config.CHROMA_DB_TESTCASES_PATH,
        "testcases",
    )
