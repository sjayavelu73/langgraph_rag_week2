"""Configuration settings for the RAG chatbot - loads from .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Find the project root (2 levels up from this file: config -> rag_langgraph -> root)
project_root = Path(__file__).parent.parent
env_file = project_root / '.env'

# Try loading from root .env file
if env_file.exists():
    load_dotenv(env_file)
    print(f"Loaded .env from: {env_file}")
else:
    # Also try loading from config/.env
    config_env = Path(__file__).parent / '.env'
    if config_env.exists():
        load_dotenv(config_env)
        print(f"Loaded .env from: {config_env}")
    else:
        print("WARNING: No .env file found. Please create one in the project root or config/ directory.")

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')
OPENAI_TEMPERATURE = float(os.environ.get('OPENAI_TEMPERATURE', '0'))

# PDF Processing Configuration
# Data directory is at project root level
DATA_DIR = project_root / 'data'
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))

# PDF File Paths (optional - if set, will use these instead of DATA_DIR)
# Can be a single path or comma-separated paths
PDF_FILE_PATHS = os.environ.get('PDF_FILE_PATHS')
if PDF_FILE_PATHS:
    # Split by comma and strip whitespace
    PDF_FILE_PATHS = [path.strip() for path in PDF_FILE_PATHS.split(',')]
else:
    PDF_FILE_PATHS = None

# Vector Database Configuration
COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'RAG5')
PERSIST_DIRECTORY = project_root / 'RAG5'

# Retrieval Configuration
RETRIEVAL_K = int(os.environ.get('RETRIEVAL_K', '10'))

# Validate required configuration
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in your .env file.\n"
        f"Create a .env file at: {env_file}\n"
        "With content: OPENAI_API_KEY=your-api-key-here"
    )
