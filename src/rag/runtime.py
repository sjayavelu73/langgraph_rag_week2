# src/rag/runtime.py
from .pdf_processor import load_pdf_documents
from .vector_store import setup_vector_store
from .rag_graph import create_rag_graph
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'config'))
from config import PDF_FILE_PATHS

print("Initializing RAG chatbot...")

# Load PDFs from file paths if specified, otherwise use default directory
if PDF_FILE_PATHS:
    print(f"Loading PDFs from specified file paths: {PDF_FILE_PATHS}")
    all_docs = load_pdf_documents(file_paths=PDF_FILE_PATHS)
else:
    print("Loading PDFs from default data directory...")
    all_docs = load_pdf_documents()

db, ensemble_retriever, semantic_retriever, bm25_retriever = setup_vector_store(all_docs if all_docs else None)
app_graph = create_rag_graph(ensemble_retriever)
