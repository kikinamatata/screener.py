"""
Document processors for preparing financial data for RAG.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models.schemas import DownloadedDocument, ProcessedDocument

logger = logging.getLogger(__name__)


class FinancialDocumentProcessor:
    """Processor for financial documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_price_data(self, doc: DownloadedDocument) -> ProcessedDocument:
        """Process price data document."""
        try:
            # Check if content is JSON or plain text
            try:
                # Try to parse as JSON first
                price_data = json.loads(doc.content)
                is_json = True
            except (json.JSONDecodeError, TypeError):
                # Content is plain text
                is_json = False
            
            if is_json:
                # Process JSON data (structured price data)
                text_content = f"Price Data for {doc.company}:\n\n"
                
                if price_data.get("current_price"):
                    text_content += f"Current Price: {price_data['current_price']}\n"
                
                if price_data.get("market_cap"):
                    text_content += f"Market Cap: {price_data['market_cap']}\n"
                
                if price_data.get("pe_ratio"):
                    text_content += f"P/E Ratio: {price_data['pe_ratio']}\n"
                
                if price_data.get("52_week_high"):
                    text_content += f"52 Week High: {price_data['52_week_high']}\n"
                
                if price_data.get("52_week_low"):
                    text_content += f"52 Week Low: {price_data['52_week_low']}\n"
                
                # Add analysis context
                text_content += f"\nThis price data shows key market metrics for {doc.company}. "
                text_content += "The current price indicates the market valuation, while the P/E ratio "
                text_content += "provides insight into the stock's valuation relative to earnings. "
                text_content += "The 52-week range shows the stock's price volatility over the past year."
            else:
                # Process plain text content (direct from screener.in)
                text_content = f"Price Data for {doc.company}:\n\n"
                
                # Clean and format the text content
                clean_content = doc.content.strip()
                if clean_content:
                    text_content += clean_content
                else:
                    text_content += "Price data could not be extracted or is empty."
                
                # Add analysis context for text content
                text_content += f"\n\nThis price information provides current market data for {doc.company}. "
                text_content += "Market data includes current stock price, trading metrics, and valuation ratios "
                text_content += "that help assess the company's market performance and investor sentiment."
            
            chunks = self.text_splitter.split_text(text_content)
            
            return ProcessedDocument(
                document_id=f"{doc.company}_price_data",
                content_chunks=chunks,
                metadata={
                    **doc.metadata,
                    "company": doc.company,
                    "processed_type": "price_data",
                    "content_type": "json" if is_json else "text",
                    "original_url": doc.url
                }
            )
        
        except Exception as e:
            logger.error(f"Error processing price data: {e}")
            return self._create_empty_processed_doc(doc, "price_data")
    
    def process_annual_report(self, doc: DownloadedDocument) -> ProcessedDocument:
        """Process annual report document."""
        try:
            # Check if content is JSON or plain text
            try:
                # Try to parse as JSON first
                financial_data = json.loads(doc.content)
                is_json = True
            except (json.JSONDecodeError, TypeError):
                # Content is plain text (from PDF extraction)
                is_json = False
            
            if is_json:
                # Process JSON data (structured financial data)
                text_content = f"Annual Report Data for {doc.company}:\n\n"
                
                # Process revenue data
                if financial_data.get("revenue"):
                    text_content += "Revenue Information:\n"
                    for metric, values in financial_data["revenue"].items():
                        text_content += f"- {metric}: {', '.join(values) if isinstance(values, list) else values}\n"
                    text_content += "\n"
                
                # Process profit data
                if financial_data.get("profit"):
                    text_content += "Profit Information:\n"
                    for metric, values in financial_data["profit"].items():
                        text_content += f"- {metric}: {', '.join(values) if isinstance(values, list) else values}\n"
                    text_content += "\n"
                
                # Process debt data
                if financial_data.get("debt"):
                    text_content += "Debt Information:\n"
                    for metric, values in financial_data["debt"].items():
                        text_content += f"- {metric}: {', '.join(values) if isinstance(values, list) else values}\n"
                    text_content += "\n"
                
                # Process cash data
                if financial_data.get("cash"):
                    text_content += "Cash Information:\n"
                    for metric, values in financial_data["cash"].items():
                        text_content += f"- {metric}: {', '.join(values) if isinstance(values, list) else values}\n"
                    text_content += "\n"
                
                # Add analysis context
                text_content += f"\nThis annual report data provides comprehensive financial information for {doc.company}. "
                text_content += "Revenue figures show the company's income generation capability. "
                text_content += "Profit metrics indicate operational efficiency and profitability. "
                text_content += "Debt levels show the company's leverage and financial risk profile. "
                text_content += "Cash positions indicate liquidity and financial stability."
            else:
                # Process plain text content (from PDF extraction)
                text_content = f"Annual Report for {doc.company}:\n\n"
                
                # Clean and format the text content
                clean_content = doc.content.strip()
                if clean_content:
                    text_content += clean_content
                else:
                    text_content += "Annual report content could not be extracted or is empty."
                
                # Add analysis context for text content
                text_content += f"\n\nThis annual report contains detailed financial and operational information for {doc.company}. "
                text_content += "It includes financial statements, business performance metrics, management discussion, "
                text_content += "and strategic insights that provide a comprehensive view of the company's operations and financial health."
            
            chunks = self.text_splitter.split_text(text_content)
            
            return ProcessedDocument(
                document_id=f"{doc.company}_annual_report_{doc.metadata.get('year', 'unknown')}",
                content_chunks=chunks,
                metadata={
                    **doc.metadata,
                    "company": doc.company,
                    "processed_type": "annual_report",
                    "content_type": "json" if is_json else "text",
                    "original_url": doc.url
                }
            )
        
        except Exception as e:
            logger.error(f"Error processing annual report: {e}")
            return self._create_empty_processed_doc(doc, "annual_report")
    
    def process_call_transcript(self, doc: DownloadedDocument) -> ProcessedDocument:
        """Process call transcript document."""
        try:
            # Check if content is JSON or plain text
            try:
                # Try to parse as JSON first
                transcript_data = json.loads(doc.content)
                is_json = True
            except (json.JSONDecodeError, TypeError):
                # Content is plain text (from PDF extraction or direct text)
                is_json = False
            
            if is_json:
                # Process JSON data (structured transcript data)
                text_content = f"Call Transcript for {doc.company}:\n\n"
                
                # Process structured transcript data if available
                if transcript_data.get("transcript"):
                    text_content += transcript_data["transcript"]
                elif transcript_data.get("content"):
                    text_content += transcript_data["content"]
                else:
                    # Convert JSON data to readable format
                    for key, value in transcript_data.items():
                        if isinstance(value, (str, int, float)):
                            text_content += f"{key}: {value}\n"
                        elif isinstance(value, list):
                            text_content += f"{key}: {', '.join(map(str, value))}\n"
                
                # Add analysis context for JSON
                text_content += f"\n\nThis call transcript data provides insights into {doc.company}'s "
                text_content += "management commentary, strategic direction, and Q&A discussions that "
                text_content += "reveal important information about the company's performance and future outlook."
            else:
                # Process plain text content (from PDF extraction or direct text)
                text_content = doc.content.strip()
                
                if text_content == "Call transcript not available from screener.in":
                    text_content = f"Call Transcript for {doc.company}:\n\n"
                    text_content += "Call transcripts are not available from screener.in. "
                    text_content += "For earnings call transcripts and management commentary, "
                    text_content += "please refer to the company's investor relations page or "
                    text_content += "financial news sources."
                elif text_content:
                    # Add context for actual transcripts
                    text_content = f"Call Transcript for {doc.company}:\n\n" + text_content
                    text_content += "\n\nThis transcript contains management commentary and Q&A "
                    text_content += "that provides insights into the company's strategy, performance, "
                    text_content += "and future outlook as discussed by executives."
                else:
                    # Handle empty content
                    text_content = f"Call Transcript for {doc.company}:\n\n"
                    text_content += "Call transcript content could not be extracted or is empty."
            
            chunks = self.text_splitter.split_text(text_content)
            
            return ProcessedDocument(
                document_id=f"{doc.company}_call_transcript_{doc.metadata.get('year', 'unknown')}_{doc.metadata.get('month', 'unknown')}",
                content_chunks=chunks,
                metadata={
                    **doc.metadata,
                    "company": doc.company,
                    "processed_type": "call_transcript",
                    "content_type": "json" if is_json else "text",
                    "original_url": doc.url
                }
            )
        
        except Exception as e:
            logger.error(f"Error processing call transcript: {e}")
            return self._create_empty_processed_doc(doc, "call_transcript")
    
    def process_document(self, doc: DownloadedDocument) -> ProcessedDocument:
        """Process a document based on its type."""
        if doc.document_type == "price_data":
            return self.process_price_data(doc)
        elif doc.document_type == "annual_report":
            return self.process_annual_report(doc)
        elif doc.document_type == "call_transcript":
            return self.process_call_transcript(doc)
        else:
            logger.error(f"Unknown document type: {doc.document_type}")
            return self._create_empty_processed_doc(doc, doc.document_type)
    
    def _create_empty_processed_doc(
        self, 
        doc: DownloadedDocument, 
        doc_type: str
    ) -> ProcessedDocument:
        """Create an empty processed document for error cases."""
        return ProcessedDocument(
            document_id=f"{doc.company}_{doc_type}_error",
            content_chunks=[f"Error processing {doc_type} for {doc.company}"],
            metadata={
                **doc.metadata,
                "company": doc.company,
                "processed_type": doc_type,
                "error": True
            }
        )
    
    def create_langchain_documents(
        self, 
        processed_docs: List[ProcessedDocument]
    ) -> List[Document]:
        """Convert processed documents to LangChain Document objects."""
        langchain_docs = []
        
        for proc_doc in processed_docs:
            for i, chunk in enumerate(proc_doc.content_chunks):
                langchain_doc = Document(
                    page_content=chunk,
                    metadata={
                        **proc_doc.metadata,
                        "chunk_id": f"{proc_doc.document_id}_chunk_{i}",
                        "document_id": proc_doc.document_id
                    }
                )
                langchain_docs.append(langchain_doc)
        
        return langchain_docs


# Global processor instance
document_processor = FinancialDocumentProcessor()
