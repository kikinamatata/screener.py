"""
Retriever agent for fetching, processing, and storing financial documents from screener.in.
This agent handles document retrieval, processing, and vector store operations.
"""

import asyncio
import logging
from langchain_core.documents import Document
from langgraph.types import Command
from langgraph.graph import END
from src.models.schemas import RAGState
from src.data.downloaders.screener_downloader import download_document
from src.data.downloaders.price_data_scraper import get_company_price_data
from src.data.processors.document_processor import document_processor
from src.utils.vector_store import vector_store

logger = logging.getLogger(__name__)


def retrieve_and_process_sync(state: RAGState) -> Command:
    """
    Synchronous wrapper for the async retrieve_and_process function.
    Required for LangGraph compatibility.
    """
    return asyncio.run(retrieve_and_process(state))


async def retrieve_and_process(state: RAGState) -> Command:
    """
    Retrieve financial documents/data, process them, and add to vector store.
    
    Args:
        state: Current retriever state with classification list
    
    Returns:
        Command object with processed data ready for RAG
    """
    if not state.get("classification"):
        error_msg = "No classification found in state"
        logger.error(error_msg)
        return Command(
            goto=END,
            update={
                "error": error_msg
            }
        )
    
    classifications = state["classification"]
    logger.info(f"Processing {len(classifications)} classification(s)")
    
    # Track processed results
    processed_count = 0
    price_data_results = {}  # Dictionary to store price data by company symbol
    use_vector_base = False
    errors = []
    
    # Process each classification in the list
    for i, classification in enumerate(classifications):
        year_info = f" for year {classification.year}" if classification.year else ""
        month_info = f" {classification.month}" if classification.month else ""
        days_info = f" ({classification.days_range} days)" if classification.days_range else ""
        logger.info(f"Processing classification {i+1}/{len(classifications)}: {classification.document_type} for {classification.company} ({classification.company_symbol}){month_info}{year_info}{days_info}")
        
        try:
            # Check if document already exists in vector store
            document_exists = False
            
            # For non-price data documents (annual reports, call transcripts), check if they exist
            if classification.document_type != "price_data":
                document_exists = vector_store.check_document_exists(
                    company_symbol=classification.company_symbol,
                    document_type=classification.document_type,
                    year=classification.year,
                    month=classification.month
                )
            
            if document_exists:
                logger.info(f"Document already exists in vector store for {classification.company} ({classification.company_symbol}) "
                        f"{classification.document_type}{month_info}{year_info}")
                processed_count += 1
                use_vector_base = True
                continue
            
            # Document doesn't exist, proceed with download and processing
            logger.info(f"Document not found in vector store, proceeding with download and processing")
            
            # Handle price data requests
            if classification.document_type == "price_data":
                logger.info(f"Fetching price data for {classification.company_symbol} with {classification.days_range} days range")
                
                # Get price data using the advanced scraper
                price_data = await get_company_price_data(
                    classification.company_symbol,
                    classification.days_range or "365"
                )
                
                # Check if we got valid price data (not an error message)
                if (price_data and 
                    not price_data.startswith("Could not retrieve") and 
                    not price_data.startswith("Error retrieving")):
                    logger.info(f"Successfully retrieved price data for {classification.company} ({classification.company_symbol})")
                    
                    # Process price data and add to vector store
                    try:
                        logger.info(f"Added price data for {classification.company}")
                        
                        # Store price data in dictionary with company symbol as key
                        price_data_results[classification.company_symbol] = price_data
                        processed_count += 1
                        
                    except Exception as e:
                        error_msg = f"Error adding price data to vector store: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                else:
                    error_msg = f"Failed to retrieve price data for {classification.company} ({classification.company_symbol}): {price_data}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Handle document-based requests (annual reports, call transcripts)
            else:
                # Download the document asynchronously
                downloaded_doc = await download_document(
                    classification.company, 
                    classification.document_type,
                    classification.company_symbol,
                    classification.year,
                    classification.month
                )
            
                if downloaded_doc:
                    logger.info(f"Successfully downloaded {classification.document_type} for {classification.company} ({classification.company_symbol})")
                    
                    # Process the document and add to vector store
                    try:
                        # Process document
                        processed_doc = document_processor.process_document(downloaded_doc)
                        logger.info(f"Successfully processed {classification.document_type} for {classification.company}")
                        
                        # Convert to LangChain documents
                        langchain_docs = document_processor.create_langchain_documents([processed_doc])
                        
                        # Add to vector store
                        vector_store.add_documents(langchain_docs)
                        logger.info(f"Added {len(langchain_docs)} document chunks to vector store for {classification.company}")
                        
                        processed_count += 1
                        use_vector_base = True
                        
                    except Exception as e:
                        error_msg = f"Error processing document: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                else:
                    error_msg = f"Failed to download {classification.document_type} for {classification.company} ({classification.company_symbol})"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        except Exception as e:
            error_msg = f"Error processing classification {i+1}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Determine final result based on processing outcomes
    logger.info(f"Processing complete: {processed_count}/{len(classifications)} classifications processed successfully")
    
    if processed_count == 0:
        # No classifications were processed successfully
        final_error = f"Failed to process any of the {len(classifications)} classifications. Errors: {'; '.join(errors)}"
        return Command(
            goto=END,
            update={
                "error": final_error
            }
        )
    
    # At least some classifications were processed successfully
    update_dict = {
        "use_vector_base": use_vector_base
    }
    
    # If we have price data, include the dictionary
    if price_data_results:
        update_dict["price_data"] = price_data_results
        logger.info(f"Collected price data for companies: {list(price_data_results.keys())}")
    
    # Include any errors as warnings if some processing succeeded
    if errors and processed_count < len(classifications):
        logger.warning(f"Some classifications failed: {'; '.join(errors)}")
    
    return Command(
        goto="rag_processor",
        update=update_dict
    )
