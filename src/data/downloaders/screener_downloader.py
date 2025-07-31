"""
Screener.in data downloader for financial information using web scraping.
"""

import os
import asyncio
import logging
import requests
from datetime import datetime
from typing import Optional, List
from bs4 import BeautifulSoup
from src.models.schemas import DownloadedDocument, DocumentType

logger = logging.getLogger(__name__)


class ScreenerDownloader:
    """Web scraper for screener.in financial data."""
    
    BASE_URL = "https://www.screener.in"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
    
    def search_company_symbol(self, company_name: str) -> Optional[str]:
        """
        Search for a company symbol on screener.in.
        Returns the company symbol/ID if found.
        """
        try:
            # Try common variations of company names
            search_terms = [
                company_name.lower(),
                company_name.lower().replace(" ", "-"),
                company_name.lower().replace(" ", ""),
                company_name.upper(),
                company_name.replace(" ", "").upper(),
            ]
            
            for term in search_terms:
                # Try direct access to company page
                test_url = f"{self.BASE_URL}/company/{term}/consolidated/"
                response = self.session.get(test_url)
                
                if response.status_code == 200:
                    # Verify it's actually a company page
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.find('title')
                    if title and company_name.lower() in title.text.lower():
                        logger.info(f"Found company symbol: {term}")
                        return term
            
            # If direct access fails, try search page scraping
            search_url = f"{self.BASE_URL}/screen/raw/"
            params = {"query": company_name}
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for company links in search results
                company_links = soup.find_all('a', href=True)
                for link in company_links:
                    href = link.get('href', '')
                    if '/company/' in href and company_name.lower() in link.text.lower():
                        # Extract symbol from URL
                        symbol = href.split('/company/')[-1].split('/')[0]
                        logger.info(f"Found company symbol from search: {symbol}")
                        return symbol
                        
        except Exception as e:
            logger.error(f"Error searching for company {company_name}: {e}")
        
        return None

    def extract_annual_report_links(self, soup: BeautifulSoup, year: int = None) -> List[str]:
        """Extract annual report PDF links from the documents section."""
        links = []
        try:
            doc_section = soup.find("section", {"id": "documents"})
            logger.debug(f"Document section found: {doc_section is not None}")
            
            if not doc_section:
                return links
                
            flex_row = doc_section.find("div", class_="flex-row flex-gap-small")
            logger.debug(f"Flex row found: {flex_row is not None}")
            
            if not flex_row:
                return links
                
            annual_div = flex_row.find("div", class_="documents annual-reports flex-column")
            logger.debug(f"Annual reports div found: {annual_div is not None}")
            
            if not annual_div:
                return links
                
            show_box = annual_div.find("div", class_="show-more-box")
            logger.debug(f"Show more box found: {show_box is not None}")
            
            if not show_box:
                return links
                
            ul_tag = show_box.find("ul", class_="list-links")
            logger.debug(f"UL tag found: {ul_tag is not None}")
            
            if not ul_tag:
                return links

            for li in ul_tag.find_all("li"):
                a_tag = li.find("a", href=True)
                if a_tag:
                    link_text = a_tag.get_text(strip=True)
                    if year is None or str(year) in link_text:
                        full_url = a_tag["href"]
                        if not full_url.startswith('http'):
                            full_url = self.BASE_URL + full_url
                        links.append(full_url)
                        logger.info(f"Found annual report link: {link_text}")
                        
        except Exception as e:
            logger.error(f"Failed to extract annual report links: {e}")
        
        return links

    def extract_concall_transcript_links_with_dates(self, soup: BeautifulSoup, year: int = None, month_abbr: str = None) -> List[tuple]:
        """Extract concall transcript PDF links with date information from the documents section."""
        links_with_dates = []
        try:
            doc_section = soup.find("section", {"id": "documents"})
            logger.debug(f"Document section found: {doc_section is not None}")
            
            if not doc_section:
                return links_with_dates
                
            flex_row = doc_section.find("div", class_="flex-row flex-gap-small")
            
            if not flex_row:
                return links_with_dates
                
            concall_div = flex_row.find("div", class_="documents concalls flex-column")
            
            if not concall_div:
                return links_with_dates
                
            show_box = concall_div.find("div", class_="show-more-box")
            
            if not show_box:
                return links_with_dates
                
            ul_tag = show_box.find("ul", class_="list-links")
            
            if not ul_tag:
                return links_with_dates

            for li in ul_tag.find_all("li", class_="flex flex-gap-8 flex-wrap"):
                date_div = li.find("div")
                if not date_div:
                    continue
                    
                date_text = date_div.get_text(strip=True).lower()
                
                # Filter by year and month if specified
                if year and str(year) not in date_text:
                    continue
                if month_abbr and month_abbr.lower() not in date_text:
                    continue
                    
                transcript_link = li.find("a", class_="concall-link", href=True)
                if transcript_link:
                    full_url = transcript_link["href"]
                    if not full_url.startswith('http'):
                        full_url = self.BASE_URL + full_url
                    
                    # Extract month from date_text (assume format like "Jan 2025" or similar)
                    month_extracted = self._extract_month_from_date(date_text)
                    
                    links_with_dates.append((full_url, date_text, month_extracted))
                    logger.info(f"Found concall transcript link: {date_text}")
                    
        except Exception as e:
            logger.error(f"Failed to extract concall transcript links with dates: {e}")
        
        return links_with_dates

    def _extract_month_from_date(self, date_text: str) -> str:
        """Extract month abbreviation from date text."""
        month_mapping = {
            'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
            'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
            'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec'
        }
        
        date_lower = date_text.lower()
        for abbr, full_abbr in month_mapping.items():
            if abbr in date_lower:
                return full_abbr
        
        # Fallback - try to extract first 3 letters that might be a month
        import re
        month_match = re.search(r'\b([a-z]{3})\b', date_lower)
        if month_match:
            month = month_match.group(1)
            return month_mapping.get(month, month.capitalize())
        
        return "Unknown"

    def extract_concall_transcript_links(self, soup: BeautifulSoup, year: int = None, month_abbr: str = None) -> List[str]:
        """Extract concall transcript PDF links from the documents section."""
        links = []
        try:
            doc_section = soup.find("section", {"id": "documents"})
            logger.debug(f"Document section found: {doc_section is not None}")
            
            if not doc_section:
                return links
                
            flex_row = doc_section.find("div", class_="flex-row flex-gap-small")
            
            if not flex_row:
                return links
                
            concall_div = flex_row.find("div", class_="documents concalls flex-column")
            
            if not concall_div:
                return links
                
            show_box = concall_div.find("div", class_="show-more-box")
            
            if not show_box:
                return links
                
            ul_tag = show_box.find("ul", class_="list-links")
            
            if not ul_tag:
                return links

            for li in ul_tag.find_all("li", class_="flex flex-gap-8 flex-wrap"):
                date_div = li.find("div")
                if not date_div:
                    continue
                    
                date_text = date_div.get_text(strip=True).lower()
                
                # Filter by year and month if specified
                if year and str(year) not in date_text:
                    continue
                if month_abbr and month_abbr.lower() not in date_text:
                    continue
                    
                transcript_link = li.find("a", class_="concall-link", href=True)
                if transcript_link:
                    full_url = transcript_link["href"]
                    if not full_url.startswith('http'):
                        full_url = self.BASE_URL + full_url
                    links.append(full_url)
                    logger.info(f"Found concall transcript link: {date_text}")
                    
        except Exception as e:
            logger.error(f"Failed to extract concall transcript links: {e}")
        
        return links

    def extract_price_data(self, soup: BeautifulSoup) -> str:
        """Extract current price and market data from the main company page."""
        try:
            # Look for price information in various possible locations
            price_data = {}
            
            # Try to find current price
            price_elements = soup.find_all(['span', 'div'], class_=['current-price', 'price', 'number'])
            for elem in price_elements:
                text = elem.get_text(strip=True)
                if '₹' in text or (text.replace(',', '').replace('.', '').isdigit() and len(text) > 2):
                    price_data['current_price'] = text
                    break
            
            # Look for market cap, P/E ratio, etc. in various table structures
            ratio_tables = soup.find_all('table')
            for table in ratio_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        metric = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        if 'market cap' in metric:
                            price_data['market_cap'] = value
                        elif 'p/e' in metric or 'pe ratio' in metric:
                            price_data['pe_ratio'] = value
                        elif 'book value' in metric:
                            price_data['book_value'] = value
                        elif 'dividend yield' in metric:
                            price_data['dividend_yield'] = value
                        elif 'eps' in metric and 'ttm' in metric:
                            price_data['eps_ttm'] = value
                        elif '52w high' in metric or '52 week high' in metric:
                            price_data['52_week_high'] = value
                        elif '52w low' in metric or '52 week low' in metric:
                            price_data['52_week_low'] = value
            
            # Also try to extract from divs with specific classes
            for div in soup.find_all('div', class_=['flex', 'company-info']):
                text = div.get_text(strip=True)
                if '₹' in text and 'current price' not in price_data:
                    # Extract price from text
                    import re
                    price_match = re.search(r'₹\s*([\d,]+\.?\d*)', text)
                    if price_match:
                        price_data['current_price'] = f"₹{price_match.group(1)}"
            
            # Format as readable text
            if price_data:
                result = "Price Data:\n"
                for key, value in price_data.items():
                    result += f"{key.replace('_', ' ').title()}: {value}\n"
                return result
            else:
                return "No price data found on the page."
            
        except Exception as e:
            logger.error(f"Error extracting price data: {e}")
            return f"Error extracting price data: {e}"

    def download_pdf_content(self, pdf_url: str, company_symbol: str, doc_type: str, year: int = None, month_abbr: str = None) -> Optional[str]:
        """Download and extract text content from PDF."""
        try:
            # Create downloads directory
            os.makedirs("downloads", exist_ok=True)
            
            # Generate filename based on document type
            if doc_type == "call_transcript" and year and month_abbr:
                filename = f"{company_symbol}_{doc_type}_{year}{month_abbr}.pdf"
            elif doc_type == "annual_report" and year:
                filename = f"{company_symbol}_{doc_type}_{year}.pdf"
            else:
                # Fallback to original URL-based naming
                filename = os.path.basename(pdf_url)
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            file_path = os.path.join("downloads", filename)
            
            response = self.session.get(pdf_url)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded PDF: {file_path}")
            
            # Try to extract text using PyPDF2
            try:
                import PyPDF2
                
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n\n"
                    
                    if text_content.strip():
                        logger.info(f"Successfully extracted text from PDF: {len(text_content)} characters")
                        return text_content
                    else:
                        logger.warning("PDF text extraction returned empty content")
                        return f"PDF downloaded successfully: {filename}\nFile path: {file_path}\nNote: Text extraction was empty - PDF might be image-based."
                        
            except ImportError:
                logger.warning("PyPDF2 not available for text extraction")
                return f"PDF downloaded successfully: {filename}\nFile path: {file_path}\nNote: Install PyPDF2 for automatic text extraction."
            except Exception as pdf_error:
                logger.error(f"Error extracting text from PDF: {pdf_error}")
                return f"PDF downloaded successfully: {filename}\nFile path: {file_path}\nNote: Text extraction failed - {pdf_error}"
            
        except Exception as e:
            logger.error(f"Error downloading PDF {pdf_url}: {e}")
            return None

    async def fetch_and_process(self, company_symbol: str, doc_type: str, year: int = None, month_abbr: str = None) -> Optional[DownloadedDocument]:
        """
        Main method to fetch and process documents from screener.in.
        """
        try:
            url = f"{self.BASE_URL}/company/{company_symbol}/consolidated/"
            response = self.session.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to load page for {company_symbol}. Status: {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract company name from page title
            title_elem = soup.find('title')
            company_name = company_symbol
            if title_elem:
                title_text = title_elem.text.strip()
                # Extract company name before " share price" or first "|" 
                if " share price" in title_text:
                    company_name = title_text.split(" share price")[0].strip()
                elif "|" in title_text:
                    company_name = title_text.split("|")[0].strip()
                elif " - " in title_text:
                    company_name = title_text.split(" - ")[0].strip()
                else:
                    company_name = company_symbol

            if doc_type == "price_data":
                content = self.extract_price_data(soup)
                
                return DownloadedDocument(
                    document_type="price_data",
                    company=company_name,
                    url=url,
                    content=content,
                    metadata={
                        "company_symbol": company_symbol,
                        "source": "screener.in",
                        "data_type": "current_price_data"
                    },
                    download_timestamp=datetime.now().isoformat()
                )
                
            elif doc_type == "annual_report":
                pdf_links = self.extract_annual_report_links(soup, year)
                
                if not pdf_links:
                    logger.warning(f"No annual report found for {company_symbol}")
                    return None
                
                # Download the first (most recent) annual report
                pdf_content = self.download_pdf_content(pdf_links[0], company_symbol, "annual_report", year)
                if not pdf_content:
                    return None
                
                return DownloadedDocument(
                    document_type="annual_report",
                    company=company_name,
                    url=pdf_links[0],
                    content=pdf_content,
                    metadata={
                        "company_symbol": company_symbol,
                        "source": "screener.in",
                        "data_type": "annual_report",
                        "pdf_links_count": len(pdf_links),
                        "primary_pdf_link": pdf_links[0] if pdf_links else "",
                        "year": year
                    },
                    download_timestamp=datetime.now().isoformat()
                )
                
            elif doc_type == "call_transcript":
                links_with_dates = self.extract_concall_transcript_links_with_dates(soup, year, month_abbr)
                
                if not links_with_dates:
                    logger.warning(f"No call transcript found for {company_symbol}")
                    return None
                
                # Get the first (most recent) transcript with its date info
                pdf_url, date_text, month_extracted = links_with_dates[0]
                
                # Download the transcript
                pdf_content = self.download_pdf_content(pdf_url, company_symbol, "call_transcript", year, month_extracted)
                if not pdf_content:
                    return None
                
                return DownloadedDocument(
                    document_type="call_transcript",
                    company=company_name,
                    url=pdf_url,
                    content=pdf_content,
                    metadata={
                        "company_symbol": company_symbol,
                        "source": "screener.in",
                        "data_type": "call_transcript",
                        "date_text": date_text,
                        "month_extracted": month_extracted,
                        "year": year,
                        "month": month_abbr if month_abbr else month_extracted,
                        "pdf_links_count": len(links_with_dates),
                        "primary_pdf_link": pdf_url
                    },
                    download_timestamp=datetime.now().isoformat()
                )
                
            else:
                logger.error(f"Unsupported doc_type: {doc_type}. Use 'price_data', 'annual_report', or 'call_transcript'.")
                return None
                
        except Exception as e:
            logger.error(f"Error in fetch_and_process: {e}")
            return None


async def download_document(
    company_name: str, 
    document_type: DocumentType,
    company_symbol: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[str] = None
) -> Optional[DownloadedDocument]:
    """Download a specific document type for a company using web scraping."""
    downloader = ScreenerDownloader()
    
    # Use provided symbol or search for it
    if company_symbol:
        logger.info(f"Using provided company symbol: {company_symbol}")
        symbol = company_symbol
    else:
        logger.info(f"Searching for company symbol for: {company_name}")
        symbol = downloader.search_company_symbol(company_name)
        if not symbol:
            logger.error(f"Could not find company symbol for: {company_name}")
            return None
    
    # Map document types
    doc_type_mapping = {
        "price_data": "price_data",
        "annual_report": "annual_report", 
        "call_transcript": "call_transcript"
    }
    
    doc_type = doc_type_mapping.get(document_type)
    if not doc_type:
        logger.error(f"Unknown document type: {document_type}")
        return None
    
    # Use provided year or default to current year for annual reports
    current_year = datetime.now().year
    target_year = year if year is not None else (current_year if doc_type != "price_data" else None)
    
    return await downloader.fetch_and_process(
        company_symbol=symbol,
        doc_type=doc_type,
        year=target_year,
        month_abbr=month
    )


# Additional utility function for testing
async def fetch_and_index(company_symbol: str, doc_type: str, year: int = None, month_abbr: str = None) -> str:
    """
    Utility function that mirrors the original function signature for compatibility.
    """
    downloader = ScreenerDownloader()
    result = await downloader.fetch_and_process(company_symbol, doc_type, year, month_abbr)
    
    if result:
        return f"Successfully processed {doc_type} for {company_symbol}"
    else:
        return f"Failed to process {doc_type} for {company_symbol}"
