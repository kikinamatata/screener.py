#!/bin/bash

# Test script for LangGraph Financial RAG Server using cURL
# Make sure the server is running on http://localhost:2024

SERVER_URL="http://localhost:2024"

echo "Testing LangGraph Financial RAG Server"
echo "======================================"

# Test 1: Check server health
echo "1. Checking server health..."
curl -s "$SERVER_URL/ok" | jq '.' || echo "Health check failed"
echo ""

# Test 2: Create a thread
echo "2. Creating a new thread..."
THREAD_RESPONSE=$(curl -s -X POST "$SERVER_URL/threads" -H "Content-Type: application/json" -d '{}')
THREAD_ID=$(echo $THREAD_RESPONSE | jq -r '.thread_id')
echo "Thread ID: $THREAD_ID"
echo ""

# Test 3: Send a query and stream results
echo "3. Sending a financial query..."
curl -s -X POST \
  "$SERVER_URL/threads/$THREAD_ID/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "query": "What is Apple'\''s current stock price?",
      "messages": [{"role": "user", "content": "What is Apple'\''s current stock price?"}],
      "documents_used": [],
      "price_data": {},
      "vector_store_updated": false,
      "use_existing_data": false,
      "chat_history": null
    },
    "stream_mode": "updates"
  }'

echo ""
echo ""

# Test 4: Wait for completion version
echo "4. Testing wait for completion..."
curl -s -X POST \
  "$SERVER_URL/threads/$THREAD_ID/runs/wait" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "query": "Show me Microsoft'\''s financial data",
      "messages": [{"role": "user", "content": "Show me Microsoft'\''s financial data"}],
      "documents_used": [],
      "price_data": {},
      "vector_store_updated": false,
      "use_existing_data": false,
      "chat_history": null
    }
  }' | jq '.'

echo ""
echo "Testing complete!"
