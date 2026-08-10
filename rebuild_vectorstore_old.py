import os
import shutil
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
import config

def build_vectorstore(doc_path, db_path, collection_name):
    print(f"Indexing documents from: {doc_path}")
    print(f"Saving to ChromaDB at: {db_path}")

    # clear old DB if exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Cleared existing ChromaDB at: {db_path}")

    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDINGS_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} PDF pages.")

    for i, doc in enumerate(docs):
        doc.metadata["page"] = doc.metadata.get("page", i + 1)

    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")

    Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=db_path, collection_name=collection_name)
    print("Done.")


build_vectorstore(
    config.REQUIREMENTS_DOCS_PATH,
    config.CHROMA_DB_REQUIREMENTS_PATH,
    "requirements",
)

build_vectorstore(
    config.TESTCASES_DOCS_PATH,
    config.CHROMA_DB_TESTCASES_PATH,
    "testcases",
)

if __name__ == "__main__":
    #build_vectorstore(config.DOCS_PATH, config.CHROMA_DB_PATH, "docs")
    build_vectorstore(config.DOCS_4007_PATH, config.CHROMA_DB_4007_PATH, "4007")
    build_vectorstore(config.DOCS_4100_PATH, config.CHROMA_DB_4100_PATH, "4100")
    build_vectorstore(
        config.REQUIREMENTS_DOCS_PATH,
        config.CHROMA_DB_REQUIREMENTS_PATH,
        "requirements",
    )

    build_vectorstore(
        config.TESTCASES_DOCS_PATH,
        config.CHROMA_DB_TESTCASES_PATH,
        "testcases",
    )
    #build_vectorstore(config.DOCS_4100_REG_PATH, config.CHROMA_DB_4100_REG_PATH, "4100_reg")