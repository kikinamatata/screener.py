import logging
from typing import Literal, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.types import Command
from langgraph.graph import END
from src.models.schemas import RAGState, FinancialAnswer
from src.utils.llm_config import get_rag_llm
from src.utils.vector_store import vector_store

logger = logging.getLogger(__name__)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analysis expert. Use the provided financial documents, price data, and chat history to answer the user's question accurately and comprehensively.

CRITICAL GUIDELINES:
- Answer EXACTLY what the user asked.
- Use chat history to understand context and avoid repeating information already provided
- If the user asks about profits but the company had losses, clearly state: "The company reported a net loss of [amount], not a profit"
- Base your answer strictly on the provided documents and price data
- If the documents don't contain the specific information requested, say so clearly
- Provide specific numbers, metrics, and data points when available
- Be precise with financial terminology - distinguish between profit, loss, revenue, earnings, etc.
- When referencing information, mention the document type and section if available

CHAT HISTORY USAGE:
- Review previous conversations to understand context
- Avoid repeating information already shared in recent messages
- Build upon previous answers when relevant
- Reference previous discussions when appropriate (e.g., "As mentioned earlier...")

PRICE DATA ANALYSIS GUIDELINES:
- When price data is available, provide current price, recent trends, and percentage changes
- Calculate and mention price performance over the requested time period
- Highlight significant price movements or patterns
- Compare current price to moving averages if available in the data
- Always include the currency symbol (₹ for Indian stocks, $ for US stocks)

Always structure your response with:
1. Direct answer to the specific question asked
2. Supporting data and evidence from documents/price data
3. Context or explanation if needed
4. Sources used (company name and data type)

If the specific information requested is not available in the documents or price data, clearly state that and suggest what type of additional information would be needed."""),
    ("user", """Question: {question}

Chat History:
{chat_history}

Financial Documents:
{context}

Price Data (if available):
{price_data}

Please provide a comprehensive answer based on the available financial data and chat history. Pay careful attention to answering the EXACT question asked while considering the conversation context.""")
])


def generate_financial_answer(state: RAGState) -> Command:
    """
    Generate financial answer using similarity search on vector store.
    
    Args:
        state: Current RAG state with processed data in vector store
    
    Returns:
        Command object with final answer
    """
    # Check if we have any data to work with
    use_vector_base = state.get("use_vector_base", False)
    price_data_dict = state.get("price_data", {})
    has_price_data = price_data_dict is not None and len(price_data_dict) > 0
    
    # We need either newly processed data, existing data in vector store, or price data
    if not use_vector_base and not has_price_data:
        error_msg = "No financial data available in vector store"
        logger.error(error_msg)
        return Command(
            goto=END,
            update={
                "error": error_msg,
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": "Error: No financial data was processed and stored for analysis.",
                    "agent": "rag_agent"
                }]
            }
        )
    
    try:
        # Determine data source and initialize variables
        classification = state.get("classification")
        documents_used = []  # Initialize as empty list if None
        
        # Perform similarity search on vector store for both new and existing data
        relevant_docs = []
        # Use the query (which is now enhanced by the classifier)
        logger.info(f"Using query for analysis: {state['query']}")
        
        if use_vector_base:
            relevant_docs = vector_store.similarity_search(
                query=state["query"],
                k=8  # Get top 8 most relevant chunks
            )
            
            # Extract comprehensive metadata from retrieved chunks
            for doc in relevant_docs:
                metadata = doc.metadata
                
                # Create a comprehensive document metadata entry
                doc_metadata = {
                    "company": metadata.get("company", "Unknown"),
                    "company_symbol": metadata.get("company_symbol", "Unknown"), 
                    "document_type": metadata.get("processed_type", metadata.get("document_type", "Unknown")),
                    "year": metadata.get("year"),
                    "month": metadata.get("month"),
                }
                
                # Add to documents_used if not already present (avoid duplicates)
                if doc_metadata not in documents_used:
                    documents_used.append(doc_metadata)
            
            logger.info(f"Found {len(relevant_docs)} relevant document chunks from vector store")
            
            if documents_used:
                unique_companies = set([doc.get('company', 'Unknown') for doc in documents_used])
                logger.info(f"Using documents from: {', '.join(unique_companies)}")
        
        if has_price_data:
            logger.info(f"Using price data for query: {state['query']}")

        
        # Generate answer using retrieved documents and/or price data
        answer = _generate_answer(state["query"], relevant_docs, price_data_dict, state.get("messages", []))
        
        if answer:
            logger.info("Successfully generated financial answer")
            
            return Command(
                goto=END,
                update={
                    "messages": state["messages"] + [{
                        "role": "assistant",
                        "content": answer.answer,
                        "agent": "rag_agent"
                    }]
                }
            )
        else:
            error_msg = "Failed to generate answer"
            logger.error(error_msg)
            
            return Command(
                goto=END,
                update={
                    "error": error_msg,
                    "messages": state["messages"] + [{
                        "role": "assistant",
                        "content": "Sorry, I was unable to generate an answer based on the available financial data.",
                        "agent": "rag_agent"
                    }]
                }
            )
    
    except Exception as e:
        error_msg = f"Error in RAG answer generation: {str(e)}"
        logger.error(error_msg)
        
        return Command(
            goto=END,
            update={
                "error": error_msg,
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error while generating the answer: {error_msg}",
                    "agent": "rag_agent"
                }]
            }
        )


def _format_chat_history(messages: List) -> str:
    """
    Format chat history for context in RAG prompt.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Formatted chat history string
    """
    if not messages:
        return "No previous conversation history."
    
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        agent_info = msg.get("agent", "")
        
        if role == "user":
            formatted_messages.append(f"User: {content}")
        elif role == "assistant":
            if agent_info == "rag_agent":
                formatted_messages.append(f"Assistant: {content}")
            # Skip non-RAG agent messages to keep history clean
    
    return "\n\n".join(formatted_messages[-5:])  # Keep last 5 exchanges


def _generate_answer(query: str, documents: List[Document], price_data_dict: Dict[str, str] = None, messages: List = None) -> FinancialAnswer:
    """
    Generate a financial answer using RAG with documents and/or price data.
    
    Args:
        query: User's financial question
        documents: Retrieved relevant documents
        price_data_dict: Dictionary of price data by company symbol (optional)
        messages: Chat history for context
    
    Returns:
        FinancialAnswer object
    """
    try:
        # Prepare context from documents without document numbers
        context_parts = []
        for doc in documents:
            company = doc.metadata.get('company', 'Unknown Company')
            doc_type = doc.metadata.get('processed_type', 'financial document')
            chunk_id = doc.metadata.get('chunk_id', '')
            
            # Extract section information from chunk_id if available
            section_info = ""
            if chunk_id and '_chunk_' in chunk_id:
                base_id = chunk_id.split('_chunk_')[0]
                section_info = f" ({base_id})"
            
            context_header = f"From {company} {doc_type}{section_info}:"
            context_parts.append(f"{context_header}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        if not context.strip():
            context = "No relevant financial documents found."
        
        # Prepare price data - format dictionary into readable text
        formatted_price_data = "No price data available."
        if price_data_dict and len(price_data_dict) > 0:
            price_sections = []
            for company_symbol, price_info in price_data_dict.items():
                price_sections.append(f"=== {company_symbol} Price Data ===\n{price_info}")
            formatted_price_data = "\n\n".join(price_sections)
        
        # Format chat history
        formatted_chat_history = _format_chat_history(messages)
        
        # Get LLM and generate answer
        llm = get_rag_llm()
        chain = RAG_PROMPT | llm
        
        response = chain.invoke({
            "question": query,
            "chat_history": formatted_chat_history,
            "context": context,
            "price_data": formatted_price_data
        })
        
        # Extract sources with proper company names
        sources = []
        companies = []
        doc_types = []
        
        # Add price data as sources if available
        if price_data_dict and len(price_data_dict) > 0:
            for company_symbol, price_info in price_data_dict.items():
                if price_info != "No price data available.":
                    sources.append(f"Screener.in - {company_symbol} Price Data")
        
        for doc in documents:
            company = doc.metadata.get('company', 'Unknown')
            doc_type = doc.metadata.get('processed_type', 'Unknown')
            
            # Only add if we have valid company name
            if company and company != 'Unknown':
                source_str = f"{company} - {doc_type}"
                if source_str not in sources:
                    sources.append(source_str)
            
            companies.append(company)
            doc_types.append(doc_type)
        
        # If no valid sources found, try to extract from metadata differently
        if not sources or all('Unknown' in source for source in sources):
            # Try to get company from document_id or other metadata
            for doc in documents:
                alt_company = (doc.metadata.get('document_id', '').split('_')[0] if 
                             doc.metadata.get('document_id') else None)
                if alt_company and alt_company not in ['Unknown', '']:
                    doc_type = doc.metadata.get('processed_type', 'financial document')
                    source_str = f"{alt_company} - {doc_type}"
                    if source_str not in sources:
                        sources.append(source_str)
        
        # Calculate confidence based on document relevance, price data, and content quality
        confidence = _calculate_confidence(query, documents, response.content, price_data_dict)
        
        return FinancialAnswer(
            answer=response.content,
            sources=sources if sources else ["Unknown - financial document"],
            confidence=confidence,
            supporting_data={
                "num_documents": len(documents),
                "document_types": doc_types,
                "companies": companies
            }
        )
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return None


def _calculate_confidence(query: str, documents: List[Document], answer: str, price_data_dict: Dict[str, str] = None) -> float:
    """
    Calculate confidence score for the generated answer.
    
    Args:
        query: Original query
        documents: Retrieved documents
        answer: Generated answer
        price_data_dict: Available price data as dictionary
    
    Returns:
        Confidence score between 0 and 1
    """
    base_confidence = 0.5
    
    # Increase confidence if we have documents
    if documents:
        base_confidence += 0.2
    
    # Increase confidence if we have price data
    if price_data_dict and len(price_data_dict) > 0:
        # Check if any company has meaningful price data
        has_meaningful_data = any(
            val != "No price data available." for val in price_data_dict.values()
        )
        if has_meaningful_data:
            base_confidence += 0.2
    
    # Increase confidence based on number of relevant documents
    doc_bonus = min(len(documents) * 0.05, 0.15)
    base_confidence += doc_bonus
    
    # Increase confidence if answer is substantive
    if len(answer) > 100:
        base_confidence += 0.1
    
    # Decrease confidence if answer indicates missing information
    if any(phrase in answer.lower() for phrase in [
        "not available", "not found", "insufficient information", 
        "cannot determine", "not specified"
    ]):
        base_confidence -= 0.2
    
    return max(0.0, min(1.0, base_confidence))
