from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import config

embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDINGS_MODEL)

vectorstore = Chroma(
    collection_name=config.COLLECTION_NAME,
    persist_directory=config.CHROMA_DB_PATH,
    embedding_function=embeddings,
)

def search_knowledge(query: str, k: int = 6):
    if not query or not query.strip():
        raise ValueError("search_knowledge requires a non-empty query.")
    return vectorstore.similarity_search(query, k=k)

def documents_as_text(documents):
    parts = []
    for i, doc in enumerate(documents, start=1):
        parts.append(
            f"[SOURCE {i}]\n"
            f"File: {doc.metadata.get('source', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)
