"""
Data models and schemas for the financial multi-agent RAG system.
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# Document type classification
DocumentType = Literal["price_data", "annual_report", "call_transcript"]


class DocumentClassification(BaseModel):
    """Model for document type classification results."""
    document_type: DocumentType = Field(description="Type of document needed")
    confidence: float = Field(description="Confidence score (0-1)")
    company: str = Field(description="Company name extracted from query")
    company_symbol: str = Field(description="Stock ticker/symbol for the company")
    year: Optional[int] = Field(default=None, description="Specific year mentioned in the query")
    month: Optional[str] = Field(default=None, description="Specific month mentioned in the query (3-letter abbreviation like 'Jan', 'Feb', etc.)")
    days_range: Optional[str] = Field(default="365", description="Time range for price data (30, 180, 365, 1095, 1825, 3652, 10000, or 1M, 6M, 1Yr, 3Yr, 5Yr, 10Yr, Max)")


class DownloadedDocument(BaseModel):
    """Model for downloaded financial documents."""
    document_type: DocumentType
    company: str
    url: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    download_timestamp: str


class ProcessedDocument(BaseModel):
    """Model for processed documents ready for RAG."""
    document_id: str
    content_chunks: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinancialAnswer(BaseModel):
    """Model for final financial answers."""
    answer: str
    sources: List[str]
    confidence: float
    supporting_data: Optional[Dict[str, Any]] = None


# LangGraph State Models
class RAGState(TypedDict):
    """Complete state for the financial RAG multi-agent system."""
    query: str
    messages: List[Dict[str, str]]
    error: Optional[str]
    classification: Optional[List[DocumentClassification]]  # Made Optional
    price_data: Optional[Dict[str, str]]  # Dictionary of price data by company symbol {company_symbol: formatted_price_data}
    use_vector_base: Optional[bool]  # Flag indicating using existing data from vector store


