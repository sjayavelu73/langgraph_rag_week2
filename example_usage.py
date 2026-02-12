"""Example usage of the RAG LangGraph implementation."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import rag
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from rag.runtime import app_graph
from rag.rag_graph import RagState
from langchain_core.messages import HumanMessage


def main():
    """Run the RAG chatbot interactively."""
    print("=" * 60)
    print("RAG LangGraph Chatbot")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the conversation\n")
    
    session_id = "default-session"
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Initialize state with the user's message
            state: RagState = {
                "messages": [HumanMessage(content=user_input)],
                "original_query": user_input,
                "rewritten_query": "",
                "retrieved_docs": [],
                "context": "",
                "output": ""
            }
            
            # Invoke the graph with checkpointing
            config = {"configurable": {"thread_id": session_id}}
            print("\nProcessing...")
            
            final_state = app_graph.invoke(
                state,
                config=config
            )
            
            # Extract the final output
            output = final_state.get("output", "")
            if not output and final_state.get("messages"):
                # Get the last AI message content
                for msg in reversed(final_state["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        output = msg.content
                        break
            
            print(f"\nAssistant: {output}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
