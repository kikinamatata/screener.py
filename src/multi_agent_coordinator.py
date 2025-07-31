"""
Multi-agent financial RAG system coordinator using LangGraph.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from src.models.schemas import RAGState
from src.agents.classifier_agent import classify_query
from src.agents.retriever_agent import retrieve_and_process_sync
from src.agents.rag_agent import generate_financial_answer
from src.agents.document_sufficiency import check_context_sufficiency

logger = logging.getLogger(__name__)

def create_financial_rag_graph():
    """
    Create and compile the financial RAG multi-agent graph.
    
    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the state graph
    workflow = StateGraph(RAGState)
    
    # Add nodes for each agent
    workflow.add_node("classifier", classify_query)
    workflow.add_node("retriever", retrieve_and_process_sync)
    workflow.add_node("rag_processor", generate_financial_answer)
    workflow.add_node("context_sufficiency", check_context_sufficiency)
    
    # Define the workflow edges
    workflow.add_edge(START, "context_sufficiency")


    
    # The agents will use Command objects to route themselves
    # No need to add conditional edges since agents handle their own routing
    
    # Compile the graph with checkpointer for server persistence
    memory = InMemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    logger.info("Financial RAG graph compiled successfully")
    return graph

# Export the compiled graph for the LangGraph server
graph = create_financial_rag_graph()

def query_financial_data(query: str) -> dict:
    """
    Query financial data using the multi-agent system.
    This function is kept for backward compatibility but is not used by the server.
    The server will directly invoke the graph.
    
    Args:
        query: User's financial question
    
    Returns:
        Dictionary with the final state including answer or error
    """
    logger.info(f"Processing financial query: {query}")

    # Initial state
    graph_instance = create_financial_rag_graph()
    initial_state = {
        "query": query,
        "messages": [{
            "role": "user",
            "content": query
        }],
        "error": None,
        "classification": None,
        "documents_used": [],
        "price_data": {},  # Dictionary for multi-company price data
        "vector_store_updated": False,
        "use_existing_data": False,
        "final_answer": None,
        "chat_history": None  # Added missing field
    }
    
    try:
        # Run the graph
        config = {"configurable": {"thread_id": "legacy_call"}}
        result = graph_instance.invoke(initial_state, config)
        return result
    
    except Exception as e:
        error_msg = f"Error in financial query processing: {str(e)}"
        logger.error(error_msg)
        
        return {
            **initial_state,
            "error": error_msg
        }

# Export the compiled graph for the LangGraph server
graph = create_financial_rag_graph()
