import sys
from typing import List, Dict, Literal
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from src.models.schemas import RAGState
from langgraph.types import Command
from src.utils.llm_config import get_chatbot_llm
from src.utils.vector_store import vector_store


DOCUMENT_SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial document analysis expert that determines whether existing documents contain sufficient information to answer a user's query.

Your task is to analyze:
1. The user's current query
2. Chat history for context
3. List of existing documents available
4. Document metadata and summaries

Make a decision: SUFFICIENT or RETRIEVE_NEW

DECISION CRITERIA:

📚 SUFFICIENT (use existing documents) when:
- Existing documents contain the specific financial data requested
- Document types match what's needed for the query (e.g., annual reports for financial metrics)
- Time periods in documents align with the query timeframe
- Company/companies mentioned are covered in existing documents
- Chat history shows similar questions were answered with current documents
- Available documents have sufficient detail and granularity for the query

🔍 RETRIEVE_NEW (need new documents) when:
- Query asks about companies not covered in existing documents
- Time periods requested are outside the range of existing documents
- Document types available don't match query requirements (e.g., need call transcripts but only have annual reports)
- Existing documents lack the specific metrics or data points requested
- Query requires more recent data than what's available
- User explicitly asks for updated or current information
- Chat history shows previous answers were incomplete due to limited document scope

ANALYSIS FRAMEWORK:
1. **Company Coverage**: Are all mentioned companies represented?
2. **Time Alignment**: Do document dates match the query timeframe?
3. **Document Type Match**: Do available document types support the query?
4. **Data Completeness**: Do documents contain the specific information needed?
5. **Recency Requirements**: Is the data fresh enough for the query?

OUTPUT FORMAT:
You must respond with a structured JSON containing:
- "decision": Either "SUFFICIENT" or "RETRIEVE_NEW"
- "reasoning": Brief explanation (1-2 sentences) for your decision
- "enhanced_query": Enhanced query that incorporates context from chat history

ENHANCED QUERY GUIDELINES:
- Analyze chat history to understand implicit context and references
- If the current query uses pronouns (it, that, them), replace with specific entities from chat history
- If the query is comparative ("compare with", "versus", "difference"), include both entities being compared
- If the query refers to previous discussions, make the reference explicit
- Always include the company name even if not mentioned in current query
- Include specific years, time periods, or metrics mentioned in previous context

EXAMPLES:
Chat: "What is TATA Motors net profit in 2022?" → "₹15,000 crores in 2022"
Current: "Compare with 2023" → Enhanced: "Compare TATA Motors net profit between 2022 and 2023"

Chat: "Show me Reliance revenue growth" → "Revenue grew 12% to ₹8,00,000 crores"  
Current: "What about profit margins?" → Enhanced: "What are Reliance Industries profit margins for the same period"

Chat: "Infosys Q1 2025 earnings" → "Q1 2025 revenue was ₹38,318 crores"
Current: "How does it compare to previous quarter?" → Enhanced: "Compare Infosys Q1 2025 earnings with Q4 2024 earnings"

Be concise and decisive in your analysis."""),

    ("user", """CURRENT QUERY: {query}

CHAT HISTORY:
{chat_history}

EXISTING DOCUMENTS METADATA:
{existing_documents}

Please analyze whether the existing documents are sufficient to answer the query or if new documents need to be retrieved.""")
])

def format_chat_history_with_agents(messages: List[Dict[str, str]]) -> str:
    """
    Format chat history with agent annotations for better context understanding.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        Formatted string with agent annotations
    """
    formatted_messages = []
    
    for msg in messages:
        role = msg.get("role", msg.get("type", ""))
        content = msg.get("content", "")
        agent_info = msg.get("agent", "")
        # Determine the display name based on role and agent info
        if role == "user" or role == "human":
            display_name = "👤 USER"
        elif role == "assistant":
            if "rag" in agent_info.lower():
                display_name = "🤖 RAG AGENT"
            else:
                continue
        # Format the message with clear separation
        formatted_msg = f"{display_name}:\n{content}\n"
        formatted_messages.append(formatted_msg)
    
    return "\n".join(formatted_messages[-5:]) if len(formatted_messages) > 5 else "\n".join(formatted_messages)


class DocumentSufficiencyDecision(BaseModel):
    """Structured output for document sufficiency decision."""
    decision: Literal["SUFFICIENT", "RETRIEVE_NEW"] = Field(
        description="Whether existing documents are sufficient or new documents need to be retrieved"
    )
    reasoning: str = Field(
        description="Brief explanation for the decision"
    )
    enhanced_query: str = Field(
        description="Enhanced query that incorporates context from chat history for better understanding"
    )


def check_context_sufficiency(state: RAGState) -> Command:
    if state.get("error"):
        print(f"\nError: {state['error']}")
        sys.exit(1)
    
    llm = get_chatbot_llm()
    llm_with_structured_output = llm.with_structured_output(DocumentSufficiencyDecision)
    chain = DOCUMENT_SUFFICIENCY_PROMPT | llm_with_structured_output
    query = state["messages"][-1]["content"]

    response = chain.invoke({
        "query": query,
        "chat_history": format_chat_history_with_agents(state["messages"]),
        "existing_documents": _format_existing_documents(state),
    })

    # Use enhanced query for better context understanding
    enhanced_query = response.enhanced_query
    state["messages"][-1]["content"] = enhanced_query

    if response.decision == "SUFFICIENT":
        return Command(goto="rag_processor", update={
            "query": enhanced_query,  # Use enhanced query instead of original
            "use_vector_base": True,
            "messages": state["messages"]
        })
    elif response.decision == "RETRIEVE_NEW":
        return Command(goto="classifier", update={
            "query": enhanced_query,  # Use enhanced query instead of original
            "use_vector_base": False,
            "messages": state["messages"]
        })


def _format_existing_documents(state: RAGState) -> str:
    """
    Safely format existing documents metadata for the prompt.
    
    Args:
        state: Current RAG state
    
    Returns:
        Formatted string describing existing documents
    """
    documents_info = []
    
    # Handle documents_used
    documents_used = vector_store.get_all_documents_metadata()
    if documents_used and isinstance(documents_used, list):
        for doc in documents_used:
            if isinstance(doc, dict):
                company = doc.get("company", "Unknown Company")
                doc_type = doc.get("document_type", "Unknown Document")
                year = doc.get("year", "Year")
                month = doc.get("month", "")
                documents_info.append(f"- {company} {doc_type} ({year},{month})")
            else:
                documents_info.append(f"- {doc}")
    
    # Handle price_data
    price_data = state.get("price_data")
    if price_data:
        if isinstance(price_data, dict):
            # Handle dictionary format (new format)
            if len(price_data) > 0:
                for company_symbol, price_info in price_data.items():
                    if price_info and price_info.strip() and price_info != "No price data available.":
                        # Extract basic info from price data
                        if "Company:" in price_info:
                            lines = price_info.split('\n')[:3]
                            for line in lines:
                                if "Company:" in line:
                                    documents_info.append(f"- {company_symbol} Price Data: {line.strip()}")
                                    break
                        else:
                            documents_info.append(f"- {company_symbol} Price Data: Available")
        elif isinstance(price_data, str) and price_data.strip():
            # Handle legacy string format
            if "Company:" in price_data:
                # Try to extract company name from price data
                lines = price_data.split('\n')[:3]  # First few lines usually have metadata
                for line in lines:
                    if "Company:" in line:
                        documents_info.append(f"- Price Data: {line.strip()}")
                        break
                else:
                    documents_info.append("- Price Data: Available")
            else:
                documents_info.append("- Price Data: Available")
    
    if not documents_info:
        return "No documents or price data available from previous queries."
    
    return "\n".join(documents_info)
