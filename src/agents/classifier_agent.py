"""
Classifier agent for determining document types needed for financial queries.
"""

import json
import logging
import os
from typing import Literal, Optional, List
from difflib import get_close_matches
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from langgraph.graph import END
from src.models.schemas import RAGState, DocumentClassification
from src.utils.llm_config import get_classifier_llm

logger = logging.getLogger(__name__)


class DocumentClassificationInput(BaseModel):
    """Model for LLM classification input without company symbol (to be looked up)."""
    document_type: Literal["price_data", "annual_report", "call_transcript"] = Field(
        description="Type of document needed"
    )
    confidence: float = Field(description="Confidence score (0-1)")
    company: str = Field(description="Company name extracted from query")
    year: Optional[int] = Field(default=None, description="Specific year mentioned in the query")
    month: Optional[str] = Field(default=None, description="Specific month mentioned in the query (3-letter abbreviation)")
    days_range: Optional[str] = Field(default=None, description="Time range for price data (only for price_data type)")


class ClassificationResponse(BaseModel):
    """Model for LLM classification response containing multiple document classifications."""
    classifications: List[DocumentClassificationInput] = Field(
        description="List of document classifications needed to answer the query"
    )
    enhanced_query: str = Field(
        description="Enhanced version of the query with specific temporal context if missing (e.g., 'latest' → '2025', adding year/time range)"
    )
    reasoning: str = Field(
        description="Brief explanation of why these documents are needed and what enhancements were made to the query"
    )

# Load company symbol mapping
def load_company_mapping():
    """Load the company name to symbol mapping from JSON file."""
    try:
        mapping_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'company_symbol_mapping.json')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading company mapping: {e}")
        return {}

COMPANY_MAPPING = load_company_mapping()

def find_company_symbol(company_name: str) -> Optional[str]:
    """
    Find the stock symbol for a company name using exact and fuzzy matching.
    
    Args:
        company_name: Name of the company
        
    Returns:
        Stock symbol if found, None otherwise
    """
    if not company_name or not COMPANY_MAPPING:
        return None
    
    # Normalize the company name
    normalized_name = company_name.strip()
    
    # Try exact match first
    if normalized_name in COMPANY_MAPPING:
        return COMPANY_MAPPING[normalized_name]
    
    # Try case-insensitive exact match
    for key in COMPANY_MAPPING.keys():
        if key.lower() == normalized_name.lower():
            return COMPANY_MAPPING[key]
    
    # Try fuzzy matching with high cutoff
    matches = get_close_matches(normalized_name, COMPANY_MAPPING.keys(), n=1, cutoff=0.8)
    if matches:
        return COMPANY_MAPPING[matches[0]]
    
    # Try partial matching for common cases
    normalized_lower = normalized_name.lower()
    for key in COMPANY_MAPPING.keys():
        key_lower = key.lower()
        # Check if the user input is a substring of a company name
        if normalized_lower in key_lower or key_lower in normalized_lower:
            # Additional check to avoid false matches
            if len(normalized_lower) > 3:  # Avoid very short matches
                return COMPANY_MAPPING[key]
    
    logger.warning(f"No symbol found for company: {company_name}")
    return None


CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial data classifier. Your job is to analyze financial queries and determine what type of documents are needed to answer them comprehensively.

Document types:
1. "price_data" - For queries about stock prices, market cap, P/E ratios, trading volumes, price movements, technical analysis
2. "annual_report" - For queries about revenue, profit, debt, cash flow, balance sheet items, income statement items, business performance metrics
3. "call_transcript" - For queries about management commentary, future outlook, strategy, guidance, qualitative insights from executives

QUERY ENHANCEMENT: When temporal context is vague or missing, enhance the query with specific dates/years:

VAGUE TERMS TO ENHANCE:
- "latest", "recent", "current", "now" → Add "2025" as current year
- "this year" → Add "2025"
- "last year" → Add "2024"  
- "historical", "past trends" → Add specific historical range like "2020-2025"
- "quarterly" without year → Add "Q4 2024" or "Q1 2025" based on context

EXAMPLES OF QUERY ENHANCEMENT:
- "What is Reliance current stock price?" → "What is Reliance Industries Ltd. current stock price as of 2025?"
- "Show me TCS latest revenue" → "Show me Tata Consultancy Services Ltd. latest revenue for 2025"
- "Wipro recent earnings call" → "Wipro Ltd. recent earnings call from Q4 2024 or Q1 2025"
- "Historical performance of Infosys" → "Historical performance of Infosys Ltd. from 2020-2025"

IMPORTANT: Return a list of document classifications. Most queries need only ONE document type, but some complex queries may need multiple:

SINGLE DOCUMENT scenarios (most common):
- Stock price queries → [price_data only]
- Financial statements, revenue, profit, debt levels, or financial performance, queries → [annual_report only]
- Management outlook, strategy, or qualitative commentary → [call_transcript only]

MULTIPLE DOCUMENT scenarios (less common):
- "Compare TCS revenue with stock performance" → [annual_report + price_data]
- "Show Reliance financials and recent management outlook" → [annual_report + call_transcript]
- Comprehensive company analysis → [annual_report + call_transcript + price_data]

Guidelines:
- Always extract the company name from the query as accurately as possible
- If a specific year is mentioned in the query (e.g., "2022 annual report", "2023 revenue"), extract the year
- If terms like "latest", "recent", "current" are used without specific year, assume 2025 as the latest year
- If a specific month is mentioned in the query (e.g., "January 2023", "March call"), extract it as 3-letter abbreviation
- For quarters, use the common mapping: Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct (typically when earnings calls happen)
- Focus on identifying the correct company name - the system will handle symbol lookup automatically
- For price_data queries, determine appropriate time range based on context:
  * Recent/current/today/latest = 30 days
  * Short term/this month/few weeks = 180 days  
  * Default/general price queries = 365 days
  * Historical/long term/yearly trends = 1825 days (5 years)
  * Maximum historical/all time = 10000 days
- Provide a confidence score for each classification
- Include reasoning for why these specific documents are needed AND what enhancements were made

RESPONSE FORMAT: Return classifications as a list, even for single document queries. Also return enhanced_query with temporal context added.

Examples:
- "What is Reliance current stock price?" → 
  enhanced_query: "What is Reliance Industries Ltd. current stock price as of 2025?"
  classifications: [price_data: company="Reliance Industries Ltd.", year=null, month=null, days_range="30"]
  reasoning: "Stock price query requires current price data. Enhanced query with company full name and current year 2025."

- "Show me Reliance Industries profit" → 
  enhanced_query: "Show me Reliance Industries Ltd. profit for 2025"
  classifications: [annual_report: company="Reliance Industries Ltd.", year=2025, month=null]
  reasoning: "Profit information requires annual report financial data. Enhanced with current year 2025 for latest data."

- "Tata Motors 2022 annual report profits" → 
  enhanced_query: "Tata Motors Ltd. 2022 annual report profits"
  classifications: [annual_report: company="Tata Motors Ltd.", year=2022, month=null]
  reasoning: "Specific year annual report requested for profit information. Enhanced with full company name."

- "Compare TCS revenue with its stock performance this year" →
  enhanced_query: "Compare Tata Consultancy Services Ltd. revenue with its stock performance for 2025"
  classifications: [
    annual_report: company="Tata Consultancy Services Ltd.", year=2025, month=null,
    price_data: company="Tata Consultancy Services Ltd.", year=null, month=null, days_range="365"
  ]
  reasoning: "Comparison requires both financial data from annual report and stock price performance data. Enhanced with full company name and current year 2025."

- "Wipro January 2023 earnings call" → 
  enhanced_query: "Wipro Ltd. January 2023 earnings call transcript"
  classifications: [call_transcript: company="Wipro Ltd.", year=2023, month="Jan"]
  reasoning: "Earnings call transcript requested for specific month and year. Enhanced with full company name and clarified document type."

"""),
    ("user", "Query: {query}")
])


def classify_query(state: RAGState) -> Command:
    """
    Classify the user query to determine what types of financial documents are needed.
    
    Args:
        state: Current classifier state
    
    Returns:
        Command object with classification results as a list
    """
    state["query"] = state["messages"][-1]["content"]
    logger.info(f"Classifying query: {state['query']}")
    
    try:
        # Get LLM with structured output
        llm = get_classifier_llm()
        structured_llm = llm.with_structured_output(ClassificationResponse)
        
        # Create classification chain
        classification_chain = CLASSIFICATION_PROMPT | structured_llm
        
        # Run classification
        result = classification_chain.invoke({"query": state["query"]})
        
        # Process each classification and validate company symbols
        valid_classifications = []
        
        for classification_input in result.classifications:
            # Look up company symbol using our mapping
            company_symbol = find_company_symbol(classification_input.company)
            
            if not company_symbol:
                error_msg = f"Could not find stock symbol for company: {classification_input.company}. Please check the company name or ensure it's listed on Indian stock exchanges."
                logger.warning(error_msg)
                
                return Command(
                    goto=END,
                    update={
                        "error": error_msg
                    }
                )
            
            # Create validated classification with symbol
            validated_classification = DocumentClassification(
                document_type=classification_input.document_type,
                confidence=classification_input.confidence,
                company=classification_input.company,
                company_symbol=company_symbol,
                year=classification_input.year,
                month=classification_input.month,
                days_range=classification_input.days_range
            )
            
            valid_classifications.append(validated_classification)
        
        # Log the classification results
        logger.info(f"Original query: {state['query']}")
        logger.info(f"Enhanced query: {result.enhanced_query}")
        logger.info(f"Classification reasoning: {result.reasoning}")
        for i, classification in enumerate(valid_classifications):
            year_info = f" for year {classification.year}" if classification.year else ""
            month_info = f" {classification.month}" if classification.month else ""
            logger.info(f"Classification {i+1}: {classification.document_type} for {classification.company} ({classification.company_symbol}){month_info}{year_info}")
        
        # Update state and route to retriever - replace query with enhanced version
        return Command(
            goto="retriever",
            update={
                "classification": valid_classifications,
                "query": result.enhanced_query,  # Replace original query with enhanced version
                "price_data": {}  # Initialize empty price data dict
            }
        )
    
    except Exception as e:
        error_msg = f"Error in classification: {str(e)}"
        logger.error(error_msg)
        
        return Command(
            goto=END,
            update={
                "error": error_msg
            }
        )