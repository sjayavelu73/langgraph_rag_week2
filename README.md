# RAG LangGraph Implementation

This directory contains a LangGraph-based RAG (Retrieval-Augmented Generation) chatbot implementation with:
- Query rewrite functionality
- Ensemble retriever (BM25 + Semantic) with equal weights
- ReAct agent pattern
- TypedDict state management

## Directory Structure

```
rag_langgraph/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── pdf_processor.py   # PDF processing utilities
│       ├── vector_store.py    # Vector store and ensemble retriever setup
│       ├── rag_graph.py       # LangGraph implementation with ReAct agent
│       └── runtime.py         # Runtime initialization
├── data/                      # PDF files directory
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration (loads from .env)
├── .env                       # Environment variables (create from .env.example)
├── .env.example               # Example environment file
├── example_usage.py           # Interactive chatbot script
├── run_chatbot.py            # Single query script
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
cd /Users/terisasaravanan/Cursor/cursor_rag_chatbot_industry/rag_langgraph
pip install -r requirements.txt
```

**Note:** For OCR functionality, you may also need to install Tesseract:
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt-get install tesseract-ocr`
- **Windows:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Configure Environment

Copy the example environment file and add your API key:

```bash
cd /Users/terisasaravanan/Cursor/cursor_rag_chatbot_industry/rag_langgraph
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Add PDF Files

Place your PDF files in the `data/` directory, or set `PDF_FILE_PATHS` in your `.env` file:

```bash
PDF_FILE_PATHS=/path/to/file1.pdf,/path/to/file2.pdf
```

## Usage

### Method 1: Interactive Chatbot

Run an interactive session:

```bash
cd /Users/terisasaravanan/Cursor/cursor_rag_chatbot_industry/rag_langgraph
python example_usage.py
```

### Method 2: Single Query

Run a single query:

```bash
cd /Users/terisasaravanan/Cursor/cursor_rag_chatbot_industry/rag_langgraph
python run_chatbot.py "Your question here"
```

### Method 3: Use in Your Code

```python
import sys
from pathlib import Path

# Add project to path
project_root = Path('/Users/terisasaravanan/Cursor/cursor_rag_chatbot_industry/rag_langgraph')
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from rag.runtime import app_graph
from rag.rag_graph import RagState
from langchain_core.messages import HumanMessage

# Initialize state
state: RagState = {
    "messages": [HumanMessage(content="Your question here")],
    "original_query": "Your question here",
    "rewritten_query": "",
    "retrieved_docs": [],
    "context": "",
    "output": ""
}

# Invoke the graph
config = {"configurable": {"thread_id": "session-1"}}
final_state = app_graph.invoke(state, config=config)

# Get the output
print(final_state["output"])
```

## Configuration

All configuration is done through the `.env` file. See `.env.example` for all available options:

- `OPENAI_API_KEY` (required): Your OpenAI API key
- `OPENAI_MODEL` (optional): Model to use (default: gpt-4o)
- `OPENAI_TEMPERATURE` (optional): Temperature setting (default: 0)
- `PDF_FILE_PATHS` (optional): Comma-separated paths to PDF files
- `CHUNK_SIZE` (optional): Text chunk size (default: 800)
- `CHUNK_OVERLAP` (optional): Chunk overlap (default: 200)
- `COLLECTION_NAME` (optional): Vector DB collection name (default: RAG5)
- `RETRIEVAL_K` (optional): Number of documents to retrieve (default: 10)

## Features

- **Query Rewrite**: Automatically rewrites queries for better retrieval
- **Ensemble Retriever**: Combines semantic (Chroma) and BM25 retrievers with equal weights
- **ReAct Agent**: Uses LangGraph's ReAct pattern with tool calling
- **Memory**: Conversation history is maintained using LangGraph's MemorySaver
- **Environment-based Config**: All settings managed through `.env` file
