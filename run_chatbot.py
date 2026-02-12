#!/usr/bin/env python3
"""Simple script to run a single query through the RAG chatbot."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import rag
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from rag.runtime import app_graph
from rag.rag_graph import RagState
from langchain_core.messages import HumanMessage


def run_query(query: str, session_id: str = "default"):
    """Run a single query through the RAG system."""
    state: RagState = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "rewritten_query": "",
        "retrieved_docs": [],
        "context": "",
        "output": ""
    }
    
    config = {"configurable": {"thread_id": session_id}}
    final_state = app_graph.invoke(state, config=config)
    
    # Extract output
    output = final_state.get("output", "")
    if not output and final_state.get("messages"):
        for msg in reversed(final_state["messages"]):
            if hasattr(msg, 'content') and msg.content:
                output = msg.content
                break
    
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_chatbot.py 'Your question here'")
        print("Example: python run_chatbot.py 'What is the main topic of the document?'")
        sys.exit(1)
    
    query = sys.argv[1]
    print(f"Query: {query}\n")
    print("Processing...\n")
    
    try:
        result = run_query(query)
        print(f"Answer: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
