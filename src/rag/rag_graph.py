"""RAG graph implementation using LangGraph with query rewrite."""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, chain
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from functools import partial
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'config'))
from config import OPENAI_MODEL, OPENAI_TEMPERATURE


class RagState(TypedDict):
    """State for the RAG graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    rewritten_query: str
    retrieved_docs: list[Document]
    context: str
    output: str


# Initialize the model
model = ChatOpenAI(temperature=OPENAI_TEMPERATURE, model=OPENAI_MODEL)

# Memory storage for checkpointing
memory = MemorySaver()


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# Query rewrite chain setup
sessions_memory_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query rewriting assistant. "
     "Rewrite the user's question to be fully self-contained. "
     "Resolve pronouns using the conversation history. "
     "If no rewrite is needed, return the original question."),
    MessagesPlaceholder(variable_name="ai_human_session_conversation"),
    ("human", "{human_question}")
])

_rewrite_chain = sessions_memory_rewrite_prompt | model | StrOutputParser()


def query_rewrite_node(state: RagState) -> RagState:
    """Rewrite the query based on conversation history."""
    # Get the last human message (the latest question)
    last_question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_question = msg.content
            break
    
    if not last_question:
        last_question = state.get("original_query", "")
    
    # Get conversation history (all messages except the last human message)
    # This will be used for context in rewriting
    conversation_history = state["messages"]
    
    # Rewrite the query using the new prompt structure
    rewritten = _rewrite_chain.invoke({
        "human_question": last_question,
        "ai_human_session_conversation": conversation_history
    })
    
    print(f"Original query: {last_question}")
    print(f"Rewritten query: {rewritten}")
    
    return {
        "original_query": last_question,
        "rewritten_query": rewritten,
        "messages": state["messages"]
    }


def retrieve_docs_node(state: RagState, ensemble_retriever=None) -> RagState:
    """Retrieve documents using the rewritten query."""
    if ensemble_retriever is None:
        raise ValueError("ensemble_retriever must be provided")
    
    query = state.get("rewritten_query") or state.get("original_query", "")
    
    if not query:
        # Fallback to last human message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
    
    retrieved_docs = ensemble_retriever.invoke(query)
    print(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
    
    # Format documents into context
    context = format_docs(retrieved_docs)
    print(f"Formatted context length: {len(context)} characters")
    if not context or len(context.strip()) < 10:
        print("WARNING: Context is empty or very short! Retrieval may have failed.")
    
    return {
        "retrieved_docs": retrieved_docs,
        "context": context
    }


def agent_node(state: RagState) -> RagState:
    """Agent node that generates response based on retrieved context."""
    # Get the pre-retrieved context
    context = state.get("context", "")
    
    # Use ChatPromptTemplate with proper variable substitution to avoid brace issues
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. 
Always refer to yourself as 'assistant' and address the user as 'Sir'.

You have been provided with the following context from the knowledge base:
{context}

When answering questions:
1. Use the provided context above to answer the question
2. Analyze the retrieved documents carefully
3. Provide accurate answers based on the retrieved context
4. If the retrieved documents don't contain the answer, say so clearly
5. Cite specific information from the documents when possible

Think step by step and provide a clear, helpful answer."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Create the agent chain (no tools, just model)
    agent_chain = agent_prompt | model | StrOutputParser()
    
    # Invoke the agent with context as a variable
    response = agent_chain.invoke({
        "context": context,
        "messages": state["messages"]
    })
    
    # Return just the new message - add_messages will automatically merge it
    return {
        "messages": [AIMessage(content=response)],
        "output": response
    }


def create_rag_graph(ensemble_retriever):
    """Create and compile the RAG graph."""
    
    # Use functools.partial to bind ensemble_retriever parameter
    retrieve_docs_bound = partial(retrieve_docs_node, ensemble_retriever=ensemble_retriever)
    
    # Create the graph
    graph = StateGraph(RagState)
    
    # Add nodes
    graph.add_node("QUERY_REWRITE", query_rewrite_node)
    graph.add_node("RETRIEVE_DOCS", retrieve_docs_bound)
    graph.add_node("AGENT", agent_node)
    
    # Set entry point
    graph.set_entry_point("QUERY_REWRITE")
    
    # Add edges - simple linear flow
    graph.add_edge("QUERY_REWRITE", "RETRIEVE_DOCS")
    graph.add_edge("RETRIEVE_DOCS", "AGENT")
    graph.add_edge("AGENT", END)
    
    # Compile the graph with memory
    return graph.compile(checkpointer=memory)
