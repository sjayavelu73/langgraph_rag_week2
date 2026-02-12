"""Vector store management for the RAG system."""

import os
import time
import shutil
from langchain_chroma import Chroma
from langchain_classic.embeddings import OpenAIEmbeddings
from chromadb.api.client import SharedSystemClient
from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'config'))
from config import PERSIST_DIRECTORY, COLLECTION_NAME, RETRIEVAL_K


def setup_vector_store(documents=None):
    """Set up or load the vector store database with ensemble retriever."""
    embeddings = OpenAIEmbeddings()
    
    # Clear ChromaDB cache
    SharedSystemClient.clear_system_cache()
    
    # Convert Path to string if needed
    persist_dir = str(PERSIST_DIRECTORY)
    
    # Remove directory safely only if we have new documents to load
    if documents and len(documents) > 0:
        if os.path.exists(persist_dir):
            print("Removing existing database to rebuild with new documents...")
            try:
                shutil.rmtree(persist_dir)
            except PermissionError:
                # On some filesystems permission errors happen, try again after a delay
                time.sleep(1)
                shutil.rmtree(persist_dir)
            # Give the filesystem some time to flush deletes
            time.sleep(2)
        
        # Create database with documents
        print(f"Creating database with {len(documents)} documents...")
        db = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=persist_dir,
            collection_name=COLLECTION_NAME
        )
        print("Database created successfully!")
    else:
        # If no documents, try to load existing database or create empty one
        if os.path.exists(persist_dir):
            print("No new documents found. Loading existing database...")
            db = Chroma(
                persist_directory=persist_dir,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
            print("Existing database loaded.")
        else:
            print("WARNING: No documents found and no existing database! Creating empty database.")
            db = Chroma(
                persist_directory=persist_dir,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
    
    # Create semantic retriever
    semantic_retriever = db.as_retriever(
        search_type='similarity',
        search_kwargs={'k': RETRIEVAL_K}
    )
    
    # Create BM25 retriever
    if documents and len(documents) > 0:
        print("Setting up BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = RETRIEVAL_K
    else:
        # If no documents, create empty BM25 retriever
        print("WARNING: Creating empty BM25 retriever (no documents available)")
        bm25_retriever = BM25Retriever.from_documents([])
        bm25_retriever.k = RETRIEVAL_K
    
    # Create ensemble retriever with equal weights (0.5 each)
    print("Creating ensemble retriever with equal weights (0.5 semantic, 0.5 BM25)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Equal weights
    )
    
    return db, ensemble_retriever, semantic_retriever, bm25_retriever
