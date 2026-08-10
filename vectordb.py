from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import *
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
import os, uuid, shutil


def ensure_soffice_on_path():
    """Ensure LibreOffice binary is discoverable in this process."""
    if shutil.which("soffice") or shutil.which("soffice.exe"):
        return True

    candidates = [
        r"C:\Program Files\LibreOffice\program",
        r"C:\Program Files (x86)\LibreOffice\program",
    ]

    for folder in candidates:
        exe_path = os.path.join(folder, "soffice.exe")
        if os.path.exists(exe_path):
            os.environ["PATH"] = f"{folder};{os.environ.get('PATH', '')}"
            return True

    return False


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)      
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
HAS_SOFFICE = ensure_soffice_on_path()

vectorstore = Chroma(collection_name="docs", persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

vectorstore_4007 = Chroma(collection_name="4007", persist_directory=CHROMA_DB_4007_PATH, embedding_function=embeddings)
vectorstore_4100 = Chroma(collection_name="4100", persist_directory=CHROMA_DB_4100_PATH, embedding_function=embeddings)
vectorstore_4100_reg = Chroma(collection_name="4100_reg", persist_directory=CHROMA_DB_4100_REG_PATH, embedding_function=embeddings)

namespace = f"chroma_db/docs_index"
record_manager = SQLRecordManager(
    namespace, db_url="sqlite:///record_manager_cache.sql"
)

namespace_4007 = f"4007_chroma_db/docs_index"
record_manager_4007 = SQLRecordManager(
    namespace_4007, db_url="sqlite:///4007_record_manager_cache.sql"
)

vectorstore_requirements = Chroma(
    collection_name="requirements",
    persist_directory=CHROMA_DB_REQUIREMENTS_PATH,
    embedding_function=embeddings,
)

vectorstore_testcases = Chroma(
    collection_name="testcases",
    persist_directory=CHROMA_DB_TESTCASES_PATH,
    embedding_function=embeddings,
)

namespace_4100 = f"4100_chroma_db/docs_index"
record_manager_4100 = SQLRecordManager(
    namespace_4100, db_url="sqlite:///4100_record_manager_cache.sql"
)

namespace_4100_reg = f"400_reg_chroma_db/docs_index"
record_manager_4100_reg = SQLRecordManager(
    namespace_4100_reg, db_url="sqlite:///4100_reg_record_manager_cache.sql"
)

class MyCustomLoader(BaseLoader):
    def __init__(self, path):
        super().__init__()  # Initialize the base class
        self.path = path

    def lazy_load(self):
        all_doc_chunks = []
        for root, _, files in os.walk(self.path):
            for file in sorted(files):
                # Skip Office lock/temp files (e.g. ~$filename.docx)
                if file.startswith("~$"):
                    continue

                full_path = os.path.join(root, file)
                file_lower = file.lower()

                # Load supported document types
                if file_lower.endswith(".txt"):
                    loader = TextLoader(full_path)
                elif file_lower.endswith(".pdf"):
                    loader = PyPDFLoader(full_path)
                elif file_lower.endswith(".docx"):
                    loader = Docx2txtLoader(full_path)
                elif file_lower.endswith(".doc"):
                    # Legacy .doc parsing requires LibreOffice (soffice) on PATH.
                    if not HAS_SOFFICE:
                        print(f"Skipping legacy .doc file (install LibreOffice to index): {full_path}")
                        continue
                    loader = UnstructuredWordDocumentLoader(full_path)
                else:
                    continue

                try:
                    docs = loader.load()
                except Exception as e:
                    print(f"Failed to load {full_path}: {e}")
                    continue

                chunks = text_splitter.split_documents(docs)
                for chunk in chunks:
                    # Add a unique ID to the metadata
                    chunk.metadata["unique_id"] = str(uuid.uuid4())  # Add a UUID for uniqueness
                all_doc_chunks.extend(chunks)
 
        yield from all_doc_chunks

    def load(self):
        return list(self.lazy_load())
    
record_manager.create_schema()
record_manager_4007.create_schema()
record_manager_4100.create_schema()
record_manager_4100_reg.create_schema()

loader = MyCustomLoader(DOCS_PATH)
loader_4007 = MyCustomLoader(DOCS_4007_PATH)
loader_4100 = MyCustomLoader(DOCS_4100_PATH)
loader_4100_reg = MyCustomLoader(DOCS_4100_REG_PATH)
loader_requirements = MyCustomLoader(REQUIREMENTS_DOCS_PATH)
loader_testcases = MyCustomLoader(TESTCASES_DOCS_PATH)

requirements_record_manager = SQLRecordManager(
    "requirements_chroma_db/docs_index",
    db_url="sqlite:///requirements_record_manager_cache.sql",
)

testcases_record_manager = SQLRecordManager(
    "testcases_chroma_db/docs_index",
    db_url="sqlite:///testcases_record_manager_cache.sql",
)

requirements_record_manager.create_schema()
testcases_record_manager.create_schema()

index(
    loader_requirements,
    requirements_record_manager,
    vectorstore_requirements,
    cleanup="full",
    source_id_key="source",
)

index(
    loader_testcases,
    testcases_record_manager,
    vectorstore_testcases,
    cleanup="full",
    source_id_key="source",
)

index(loader, record_manager, vectorstore, cleanup="full", source_id_key="source")
index(loader_4007, record_manager_4007, vectorstore_4007, cleanup="full", source_id_key="source")
index(loader_4100, record_manager_4100, vectorstore_4100, cleanup="full", source_id_key="source")
index(loader_4100_reg, record_manager_4100_reg, vectorstore_4100_reg, cleanup="full", source_id_key="source")

