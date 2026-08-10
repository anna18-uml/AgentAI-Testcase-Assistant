import os
import shutil

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

import config

def load_documents(folder):
    documents = []
    os.makedirs(folder, exist_ok=True)

    for root, _, filenames in os.walk(folder):
        for filename in sorted(filenames):
            path = os.path.join(root, filename)
            lower = filename.lower()

            try:
                if lower.endswith(".pdf"):
                    docs = PyPDFLoader(path).load()
                elif lower.endswith(".txt"):
                    docs = TextLoader(path, encoding="utf-8").load()
                elif lower.endswith(".docx"):
                    docs = Docx2txtLoader(path).load()
                else:
                    continue

                for doc in docs:
                    doc.metadata["source"] = path
                documents.extend(docs)

            except Exception as exc:
                print(f"Skipping {path}: {exc}")

    return documents

def build_vectorstore():
    docs = load_documents(config.DOCS_PATH)

    if not docs:
        print("No PDF, TXT, or DOCX files found in knowledge_docs.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    if os.path.exists(config.CHROMA_DB_PATH):
        shutil.rmtree(config.CHROMA_DB_PATH)

    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDINGS_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DB_PATH,
        collection_name=config.COLLECTION_NAME,
    )

    print(f"Indexed {len(chunks)} chunks into one shared vector database.")

if __name__ == "__main__":
    build_vectorstore()
