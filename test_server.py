#!/usr/bin/env python3
"""
Test script for the LangGraph Financial RAG Server.
"""

import asyncio
import json
from langgraph_sdk import get_client

async def test_financial_rag_server():
    """Test the financial RAG server with sample queries."""
    
    # Connect to local development server
    client = get_client(url="http://localhost:2024")
    
    # Test queries
    test_queries = [
        "What was Apple's stock price last month?",
        "Show me Microsoft's financial performance in Q3 2023",
        "Compare Google and Amazon's revenue growth"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {query}")
        print('='*50)
        
        try:
            # Create a thread for this conversation
            thread = await client.threads.create()
            thread_id = thread["thread_id"]
            print(f"Created thread: {thread_id}")
            
            # Send the query and stream responses
            print("\nStreaming response:")
            async for chunk in client.runs.stream(
                thread_id,
                "agent",  # Assistant ID from langgraph.json
                input={
                    "query": query,
                    "messages": [{"role": "user", "content": query}],
                    "documents_used": [],
                    "price_data": {},
                    "vector_store_updated": False,
                    "use_existing_data": False,
                    "chat_history": None
                },
                stream_mode="updates"
            ):
                print(f"Event: {chunk.event}")
                if chunk.data:
                    print(f"Data: {json.dumps(chunk.data, indent=2, default=str)}")
                print("-" * 30)
                
        except Exception as e:
            print(f"Error testing query '{query}': {e}")
            
        # Add delay between tests
        await asyncio.sleep(2)

def test_sync():
    """Synchronous version for easier testing."""
    from langgraph_sdk import get_sync_client
    
    client = get_sync_client(url="http://localhost:2024")
    
    # Simple test
    query = "What is Apple's current stock price?"
    
    print(f"Testing query: {query}")
    
    try:
        # Create thread
        thread = client.threads.create()
        thread_id = thread["thread_id"]
        
        # Run and wait for completion
        result = client.runs.wait(
            thread_id,
            "agent",
            input={
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "documents_used": [],
                "price_data": {},
                "vector_store_updated": False,
                "use_existing_data": False,
                "chat_history": None
            }
        )
        
        print("Final result:")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        test_sync()
    else:
        asyncio.run(test_financial_rag_server())
