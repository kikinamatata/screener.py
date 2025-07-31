"""
Vector store utilities for document storage and retrieval.
"""

import os
import logging
from typing import List, Optional, Dict, Set
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store manager for financial documents."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "financial_documents"
    ):
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", 
            "./data/chroma_db"
        )
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self._vectorstore = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Get or create the vector store instance."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        # Filter out complex metadata that ChromaDB can't handle
        filtered_docs = filter_complex_metadata(documents)
        return self.vectorstore.add_documents(filtered_docs)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """Search for similar documents."""
        # Handle ChromaDB filter format if needed
        processed_filter = self._process_filter(filter_dict) if filter_dict else None
        
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=processed_filter
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter_dict: Optional[dict] = None
    ) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        # Handle ChromaDB filter format if needed
        processed_filter = self._process_filter(filter_dict) if filter_dict else None
        
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=processed_filter
        )
    
    def _process_filter(self, filter_dict: dict) -> dict:
        """
        Process filter dictionary to ensure ChromaDB compatibility.
        
        ChromaDB requires explicit operators for filters. This method converts
        simple key-value filters to use explicit $eq operators and combines
        multiple conditions with $and.
        """
        if not filter_dict:
            return filter_dict
        
        # Check if the filter already has operators (contains $and, $or, etc.)
        if any(key.startswith('$') for key in filter_dict.keys()):
            return filter_dict
        
        # Convert simple key-value pairs to explicit $eq operators
        conditions = []
        for key, value in filter_dict.items():
            conditions.append({key: {"$eq": value}})
        
        # If only one condition, return it directly
        if len(conditions) == 1:
            return conditions[0]
        
        # Multiple conditions need $and operator
        return {"$and": conditions}
    
    def check_document_exists(
        self, 
        company_symbol: str, 
        document_type: str, 
        year: Optional[int] = None,
        month: Optional[str] = None
    ) -> bool:
        """
        Check if a document already exists in the vector store.
        
        Args:
            company_symbol: Company symbol (e.g., 'RELIANCE')
            document_type: Type of document ('annual_report', 'call_transcript', 'price_data')
            year: Optional year filter
            month: Optional month filter
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            # Build filter based on available criteria
            # ChromaDB requires explicit operators, so we use $and for multiple conditions
            conditions = []
            
            # Add company symbol condition
            conditions.append({"company_symbol": {"$eq": company_symbol}})
            
            # Add document type condition
            conditions.append({"processed_type": {"$eq": document_type}})
            
            # Add year filter if provided
            if year:
                conditions.append({"year": {"$eq": year}})
            
            # Add month filter if provided
            if month:
                conditions.append({"month": {"$eq": month}})
            
            # Build final filter with $and operator
            if len(conditions) == 1:
                filter_dict = conditions[0]
            else:
                filter_dict = {"$and": conditions}
            
            # Search for any documents matching the criteria (k=1 for efficiency)
            results = self.vectorstore.similarity_search(
                query=f"{company_symbol} {document_type}",
                k=1,
                filter=filter_dict
            )
            
            exists = len(results) > 0
            return exists
            
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False
    
    def delete_collection(self):
        """Delete the entire collection."""
        if self._vectorstore:
            self._vectorstore.delete_collection()
            self._vectorstore = None
    
    def get_all_documents_metadata(self) -> List[Dict]:
        """
        Get metadata of all documents stored in the vector store.
        
        Returns:
            List of dictionaries containing document metadata with unique documents
            based on company, document type, year, and month.
        """
        try:
            # Get the collection to access all documents
            collection = self.vectorstore._collection
            
            # Get all documents from the collection
            # Using peek() to get a large number or get() with limit
            all_data = collection.peek(limit=10000)  # Adjust limit as needed
            
            if not all_data or 'metadatas' not in all_data:
                return []
            
            # Process metadata to get unique documents
            unique_documents = {}
            
            for metadata in all_data['metadatas']:
                if not metadata:
                    continue
                
                # Extract key fields
                company = metadata.get('company', 'Unknown')
                company_symbol = metadata.get('company_symbol', 'Unknown')
                doc_type = metadata.get('processed_type', metadata.get('document_type', 'Unknown'))
                year = metadata.get('year')
                month = metadata.get('month')
                url = metadata.get('url', '')
                download_timestamp = metadata.get('download_timestamp', '')
                
                # Create a unique key for deduplication
                unique_key = f"{company_symbol}_{doc_type}_{year}_{month}"
                
                # Only keep one entry per unique document
                if unique_key not in unique_documents:
                    unique_documents[unique_key] = {
                        'company': company,
                        'company_symbol': company_symbol,
                        'document_type': doc_type,
                        'year': year,
                        'month': month,
                        'url': url,
                        'download_timestamp': download_timestamp
                    }
            
            # Convert to list and sort by company then year
            documents_list = list(unique_documents.values())
            documents_list.sort(key=lambda x: (x['company'], x['year'] or 0, x['month'] or ''))
            
            return documents_list
            
        except Exception as e:
            logger.error(f"Error getting all documents metadata: {e}")
            return []
    
    def get_documents_summary(self) -> Dict:
        """
        Get a summary of all documents in the vector store.
        
        Returns:
            Dictionary with summary statistics of stored documents.
        """
        try:
            documents = self.get_all_documents_metadata()
            
            if not documents:
                return {
                    'total_documents': 0,
                    'companies': [],
                    'document_types': {},
                    'years_covered': [],
                    'summary': 'No documents found in vector store.'
                }
            
            # Analyze the documents
            companies = set()
            doc_types = {}
            years = set()
            
            for doc in documents:
                companies.add(doc['company'])
                
                doc_type = doc['document_type']
                if doc_type in doc_types:
                    doc_types[doc_type] += 1
                else:
                    doc_types[doc_type] = 1
                
                if doc['year']:
                    years.add(doc['year'])
            
            return {
                'total_documents': len(documents),
                'companies': sorted(list(companies)),
                'document_types': doc_types,
                'years_covered': sorted(list(years)),
                'documents_detail': documents
            }
            
        except Exception as e:
            logger.error(f"Error getting documents summary: {e}")
            return {
                'total_documents': 0,
                'companies': [],
                'document_types': {},
                'years_covered': [],
                'summary': f'Error retrieving documents: {str(e)}'
            }
    
    def format_documents_for_display(self) -> str:
        """
        Format the list of all documents for human-readable display.
        
        Returns:
            Formatted string describing all stored documents.
        """
        try:
            summary = self.get_documents_summary()
            
            if summary['total_documents'] == 0:
                return "📂 No documents found in vector store."
            
            output = []
            output.append(f"📚 VECTOR STORE DOCUMENT INVENTORY")
            output.append(f"{'='*50}")
            output.append(f"Total Documents: {summary['total_documents']}")
            output.append(f"Companies: {len(summary['companies'])}")
            output.append(f"Years Covered: {min(summary['years_covered'])} - {max(summary['years_covered'])}")
            output.append("")
            
            # Group by company
            documents = summary['documents_detail']
            current_company = None
            
            for doc in documents:
                if doc['company'] != current_company:
                    current_company = doc['company']
                    output.append(f"🏢 {current_company} ({doc['company_symbol']})")
                
                # Format document entry
                year_str = f" ({doc['year']})" if doc['year'] else ""
                month_str = f" - {doc['month']}" if doc['month'] else ""
                
                doc_icon = {
                    'annual_report': '📊',
                    'call_transcript': '🎙️',
                    'price_data': '💹'
                }.get(doc['document_type'], '📄')
                
                output.append(f"  {doc_icon} {doc['document_type']}{year_str}{month_str}")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error formatting documents for display: {e}")
            return f"❌ Error retrieving documents: {str(e)}"
    
    def as_retriever(self, **kwargs):
        """Get a retriever interface to the vector store."""
        return self.vectorstore.as_retriever(**kwargs)


# Global vector store instance
vector_store = VectorStore()