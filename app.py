#!/usr/bin/env python3
"""
Simple LLM Chat Interface using Streamlit
Direct integration with Financial RAG Graph
"""

import streamlit as st
import sys
import uuid
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the graph creation function
from src.multi_agent_coordinator import create_financial_rag_graph

# Page configuration
st.set_page_config(
    page_title="Financial RAG Chat",
    page_icon="💬",
    layout="centered"
)

# Simple CSS for chat styling
st.markdown("""
<style>
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 5px 15px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        color: #333;
        padding: 15px 20px;
        border-radius: 15px 15px 15px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .supporting-data {
        background-color: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    
    .data-source {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 6px;
        padding: 8px;
        margin: 5px 0;
        font-size: 0.85em;
        color: #856404;
    }
    
    .timestamp {
        font-size: 0.7em;
        color: #666;
        margin-top: 5px;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 6px;
        margin: 5px 0;
    }
    
    /* Align send button with input field */
    .stButton > button {
        height: 38px;
        margin-top: 25px;
    }
    
    /* Align form submit button with input field */
    .stFormSubmitButton > button {
        height: 38px;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"chat_{str(uuid.uuid4())}"

@st.cache_resource
def initialize_graph():
    """Initialize the Financial RAG graph with checkpointing"""
    try:
        # Create graph
        graph = create_financial_rag_graph()
        
        return graph
    except Exception as e:
        st.error(f"Failed to initialize graph: {str(e)}")
        return None

def format_financial_response(response: Any) -> str:
    """Format the financial response for better presentation"""
    if isinstance(response, dict):
        # Handle structured response
        if 'final_answer' in response and response['final_answer']:
            final_answer = response['final_answer']
            if isinstance(final_answer, dict):
                return format_structured_answer(final_answer, response)
            else:
                return clean_response_text(str(final_answer))
        elif 'messages' in response and response['messages']:
            last_message = response['messages'][-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                return clean_response_text(last_message['content'])
            else:
                return clean_response_text(str(last_message))
        else:
            return clean_response_text(str(response))
    else:
        return clean_response_text(str(response))

def format_structured_answer(answer: Dict[str, Any], full_response: Dict[str, Any]) -> str:
    """Format a structured FinancialAnswer object"""
    html_parts = []
    
    # Main answer
    if 'answer' in answer:
        html_parts.append(f"**{answer['answer']}**\n\n")
    
    # Supporting data
    if 'supporting_data' in answer and answer['supporting_data']:
        html_parts.append("📈 **Supporting Data:**\n")
        supporting_data = answer['supporting_data']
        if isinstance(supporting_data, dict):
            for key, value in supporting_data.items():
                html_parts.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")
        else:
            html_parts.append(f"- {supporting_data}\n")
        html_parts.append("\n")
    
    # Sources
    if 'sources' in answer and answer['sources']:
        html_parts.append("📚 **Sources:**\n")
        for source in answer['sources']:
            html_parts.append(f"- {source}\n")
        html_parts.append("\n")
    
    # Confidence score
    if 'confidence' in answer:
        confidence = float(answer['confidence'])
        html_parts.append(f"🎯 **Confidence:** {confidence:.1%}\n")
    
    return "".join(html_parts)

def clean_response_text(text: str) -> str:
    """Clean up response text for better presentation"""
    # Remove agent metadata
    text = re.sub(r"'role':\s*'assistant',?\s*'content':\s*'([^']*)'", r'\1', text)
    text = re.sub(r"'agent':\s*'[^']*'", '', text)
    
    # Remove excessive quotes and brackets
    text = re.sub(r'^["\'\[\{]+|["\'\]\}]+$', '', text.strip())
    
    # Clean up common formatting issues but keep it as markdown, not HTML
    text = text.replace('\\n', '\n')
    
    return text

def send_query_to_graph(user_input):
    """Send query to graph and extract response from state"""
    try:
        # Configuration with thread ID for state persistence
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Invoke the graph with proper state structure
        result = st.session_state.graph.invoke(
            {
                "messages": st.session_state.messages, 
                "query": user_input,
                "error": None,
                "classification": None,
                "price_data": {},
                "use_existing_data": False,
                "use_vector_base": False
            },
            config=config
        )
        
        # Format the response for better presentation
        formatted_response = format_financial_response(result)

        return True, formatted_response, result
        
    except Exception as e:
        error_html = f"""
        <div class='error-message'>
            <strong>❌ Error:</strong> {str(e)}
        </div>
        """
        return False, error_html, None

# Main interface
st.title("💬 Financial RAG Chat")
st.write("Ask questions about financial data, stocks, and market information")

# Initialize graph if not done
if st.session_state.graph is None:
    with st.spinner("🔄 Initializing Financial RAG System..."):
        st.session_state.graph = initialize_graph()
    
    if st.session_state.graph is None:
        st.error("❌ Failed to initialize the graph. Please check your configuration.")
        st.stop()

# Display all chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about stocks, financial reports, market data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("🤔 Processing your financial query..."):
        success, formatted_response, raw_result = send_query_to_graph(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Display the response
        st.markdown(formatted_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": formatted_response,
        "raw_result": raw_result  # Store for debugging if needed
    })

# Sidebar with controls
with st.sidebar:
    st.header("💼 Chat Controls")
    
    # Session info
    st.write(f"**Session ID:** `{st.session_state.thread_id[:12]}...`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        # Start new session for fresh state
        st.session_state.thread_id = f"chat_{str(uuid.uuid4())}"
        st.rerun()
    
    # Debug section (expandable)
    with st.expander("🔧 Debug Info"):
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if "raw_result" in last_message and last_message["raw_result"]:
                st.write("**Last Raw Result:**")
                st.json(last_message["raw_result"], expanded=False)
        
        st.write("**Graph Status:**")
        st.write(f"Initialized: {'✅' if st.session_state.graph else '❌'}")
