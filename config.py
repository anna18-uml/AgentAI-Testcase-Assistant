import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "knowledge_docs")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "knowledge_chroma_db")
COLLECTION_NAME = "knowledge"

OLLAMA_EMBEDDINGS_MODEL = "nomic-embed-text"
OLLAMA_GENERATION_MODEL = "mistral"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
