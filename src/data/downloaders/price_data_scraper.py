"""
Price data scraper for screener.in using zendriver browser automation.
Implements network interception to capture chart API data.
"""

import asyncio
import re
import json
import logging
from typing import Optional, Literal, Dict, List, Any
from urllib.parse import urlparse, parse_qs
from zendriver import cdp, loop, start

logger = logging.getLogger(__name__)


class PriceDataScraper:
    """
    Advanced price data scraper using browser automation and network interception.
    Captures chart API data from screener.in for accurate price information.
    """
    
    def __init__(self):
        self.found_chart_data = None
        self.target_days = 365
        self.data_found_event = None
        self.chart_api_pattern = re.compile(
            r"https://www\.screener\.in/api/company/(.*?)/chart/\?q=.*"
        )
    
    async def _receive_handler(self, event: cdp.network.ResponseReceived, tab):
        """
        Handler for network response events.
        Intercepts chart API responses and extracts price data.
        """
        response_url = event.response.url
        url = urlparse(response_url)
        qsl = parse_qs(url.query)

        # Check if response matches chart API pattern and target days
        if (self.chart_api_pattern.match(response_url) and 
            str(self.target_days) in qsl.get("days", [])):
            
            logger.info(f"Intercepted chart API response: {response_url}")
            
            # Extract company ID from URL
            match = self.chart_api_pattern.match(response_url)
            company_id = match.group(1) if match else "Unknown"
            logger.info(f"Company ID detected: {company_id}")

            try:
                # Get response body
                body, base64encoded = await tab.send(
                    cdp.network.get_response_body(event.request_id)
                )

                if base64encoded:
                    import base64
                    body = base64.b64decode(body)

                # Parse JSON data
                json_data = json.loads(body)
                self.found_chart_data = json_data
                
                if self.data_found_event:
                    self.data_found_event.set()
                
                logger.info("Chart data successfully captured and parsed")

            except UnicodeDecodeError:
                logger.error(f"Failed to decode response body for {response_url}")
            except json.JSONDecodeError:
                logger.error(f"Response is not valid JSON: {response_url}")
            except Exception as e:
                logger.error(f"Error processing response: {e}")

    def _map_days_parameter(self, days_query: Any) -> int:
        """
        Map various day representations to numeric values.
        """
        literal_to_days = {
            "1M": 30, "30": 30,
            "6M": 180, "180": 180,
            "1Yr": 365, "365": 365,
            "3Yr": 1095, "1095": 1095,
            "5Yr": 1825, "1825": 1825,
            "10Yr": 3652, "3652": 3652,
            "Max": 10000, "10000": 10000,
        }
        return literal_to_days.get(str(days_query), 365)

    async def get_price_data(
        self,
        company_symbol: str,
        days_query: Optional[
            Literal[30, 180, 365, 1095, 1825, 3652, 10000] |
            Literal["1M", "6M", "1Yr", "3Yr", "5Yr", "10Yr", "Max"]
        ] = 365
    ) -> Optional[Dict[str, Any]]:
        """
        Extract price data for a company using browser automation.
        
        Args:
            company_symbol: Company symbol/ID on screener.in
            days_query: Time range for price data
            
        Returns:
            Dictionary containing price data or None if failed
        """
        self.target_days = self._map_days_parameter(days_query)
        self.found_chart_data = None
        self.data_found_event = asyncio.Event()
        
        browser = None
        try:
            logger.info(f"Starting price data extraction for {company_symbol}")
            browser = await start()
            tab = browser.main_tab

            # Navigate to company page
            company_url = f"https://www.screener.in/company/{company_symbol}/"
            logger.info(f"Navigating to {company_url}")
            
            await tab.get(company_url)
            logger.info("Page navigation completed")
            
            # Add network response handler
            tab.add_handler(cdp.network.ResponseReceived, self._receive_handler)
            logger.info("Network response handler added")
            
            # Wait for page to load with extended timeout
            try:
                await asyncio.wait_for(tab.wait_for_ready_state(), timeout=20)
                logger.info("Page ready state achieved")
            except asyncio.TimeoutError:
                logger.warning("Page ready state timeout, but continuing - some data may have loaded")
                # Continue anyway - we might have intercepted data during loading
            
            # Click the specific days button to trigger chart data request
            day_button_selector = f"button[name=days][value='{self.target_days}']"
            logger.info(f"Looking for button with selector: {day_button_selector}")
            
            # Try multiple times to find and click the button
            button_clicked = False
            for attempt in range(3):
                try:
                    day_button = await tab.select(day_button_selector)
                    if day_button:
                        await day_button.click()
                        logger.info(f"Clicked days button for {self.target_days} days")
                        button_clicked = True
                        break
                    else:
                        if attempt == 0:
                            logger.warning(f"Could not find days button for {self.target_days} days on attempt {attempt + 1}")
                        await asyncio.sleep(2)  # Wait a bit for page to load more
                except Exception as e:
                    logger.warning(f"Button click attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
            
            if not button_clicked:
                logger.warning(f"Could not click days button after 3 attempts")
                # Try to find what buttons are available
                try:
                    all_buttons = await tab.select_all("button")
                    logger.info(f"Found {len(all_buttons)} buttons on page")
                    for i, button in enumerate(all_buttons[:10]):  # Log first 10 buttons
                        button_text = await button.get_text() if hasattr(button, 'get_text') else "Unknown"
                        logger.info(f"Button {i}: {button_text}")
                except Exception as btn_error:
                    logger.error(f"Error inspecting buttons: {btn_error}")

            # Wait for chart data to be captured
            logger.info("Waiting for chart data...")
            
            # Check if we already have data from initial page load
            if self.found_chart_data:
                logger.info("Chart data already available from page load")
                return self.found_chart_data
            
            try:
                await asyncio.wait_for(self.data_found_event.wait(), timeout=25)
                
                if self.found_chart_data:
                    logger.info("Successfully captured price data")
                    return self.found_chart_data
                else:
                    logger.warning("Data event set but no chart data found")
                    return None
                    
            except asyncio.TimeoutError:
                # Check if we got data despite timeout
                if self.found_chart_data:
                    logger.info("Price data captured successfully despite timeout")
                    return self.found_chart_data
                else:
                    logger.error("Timeout waiting for chart data")
                    return None

        except Exception as e:
            logger.error(f"Error extracting price data: {e}")
            # Add more detailed error information
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
            
        finally:
            if browser:
                try:
                    await browser.stop()
                    logger.info("Browser closed")
                except Exception as e:
                    logger.error(f"Error closing browser: {e}")

    def extract_all_datasets(self, chart_data: Dict[str, Any]) -> Dict[str, List[List[str]]]:
        """
        Extract all datasets from chart data response.
        
        Args:
            chart_data: Raw chart data from API
            
        Returns:
            Dictionary with metric names as keys and [date, value] pairs as values
        """
        if not chart_data or "datasets" not in chart_data:
            return {}
        
        datasets = {}
        for dataset in chart_data["datasets"]:
            metric = dataset.get("metric")
            if metric:
                datasets[metric] = dataset.get("values", [])
        
        return datasets

    def extract_price_values(self, chart_data: Dict[str, Any]) -> List[List[str]]:
        """
        Extract price values from chart data response.
        
        Args:
            chart_data: Raw chart data from API
            
        Returns:
            List of [date, price] pairs
        """
        if not chart_data or "datasets" not in chart_data:
            return []
        
        # Find the price dataset
        for dataset in chart_data["datasets"]:
            if dataset.get("metric") == "Price":
                return dataset.get("values", [])
        
        return []

    def format_price_data_for_rag(self, chart_data: Dict[str, Any]) -> str:
        """
        Format comprehensive price data for inclusion in RAG prompts.
        Includes Price, DMA50, DMA200, and Volume data.
        
        Args:
            chart_data: Raw chart data from API
            
        Returns:
            Formatted string for RAG context
        """
        if not chart_data:
            return "No price data available."
        
        # Extract all datasets
        all_datasets = self.extract_all_datasets(chart_data)
        if not all_datasets:
            return "No datasets found in chart response."
        
        # Extract price data for basic calculations
        price_values = all_datasets.get("Price", [])
        if not price_values:
            return "No price data found in response."
        
        total_records = len(price_values)
        if total_records == 0:
            return "No price records found."
        
        # Get first and last prices for summary
        first_date, first_price = price_values[0][0], price_values[0][1]
        last_date, last_price = price_values[-1][0], price_values[-1][1]
        
        # Calculate price change
        try:
            price_change = float(last_price) - float(first_price)
            price_change_pct = (price_change / float(first_price)) * 100
        except (ValueError, ZeroDivisionError):
            price_change = 0
            price_change_pct = 0
        
        # Get current moving averages
        dma50_values = all_datasets.get("DMA50", [])
        dma200_values = all_datasets.get("DMA200", [])
        volume_values = all_datasets.get("Volume", [])
        
        current_dma50 = dma50_values[-1][1] if dma50_values else "N/A"
        current_dma200 = dma200_values[-1][1] if dma200_values else "N/A"
        current_volume = volume_values[-1][1] if volume_values else "N/A"
        
        # Format recent prices (last 10 days)
        recent_prices = price_values[-10:] if len(price_values) >= 10 else price_values
        recent_data_lines = []
        
        for i, (date, price) in enumerate(recent_prices):
            dma50 = dma50_values[-(10-i)][1] if i < len(dma50_values) and (10-i) <= len(dma50_values) else "N/A"
            dma200 = dma200_values[-(10-i)][1] if i < len(dma200_values) and (10-i) <= len(dma200_values) else "N/A"
            volume = volume_values[-(10-i)][1] if i < len(volume_values) and (10-i) <= len(volume_values) else "N/A"
            
            recent_data_lines.append(f"  {date}: Price=₹{price}, DMA50=₹{dma50}, DMA200=₹{dma200}, Volume={volume}")
        
        recent_data_str = "\n".join(recent_data_lines)
        
        # Create comprehensive formatted data
        formatted_data = f"""
COMPREHENSIVE STOCK DATA ANALYSIS:
=====================================
Period: {first_date} to {last_date}
Total Trading Days: {total_records}

CURRENT VALUES (as of {last_date}):
• Stock Price: ₹{last_price}
• 50-Day Moving Average (DMA50): ₹{current_dma50}
• 200-Day Moving Average (DMA200): ₹{current_dma200}
• Trading Volume: {current_volume}

PRICE PERFORMANCE:
• Starting Price ({first_date}): ₹{first_price}
• Current Price ({last_date}): ₹{last_price}
• Price Change: ₹{price_change:.2f} ({price_change_pct:+.2f}%)

AVAILABLE DATASETS:
• Price Data: {len(price_values)} data points
• DMA50 Data: {len(dma50_values)} data points  
• DMA200 Data: {len(dma200_values)} data points
• Volume Data: {len(volume_values)} data points

RECENT DATA (Last {len(recent_prices)} trading days):
{recent_data_str}

TECHNICAL ANALYSIS CONTEXT:
- DMA50 (50-day moving average): Shows short-term price trend
- DMA200 (200-day moving average): Shows long-term price trend  
- Price vs DMA200: Stock trading {"above" if current_dma200 != "N/A" and float(last_price) > float(current_dma200) else "at"} 200-day moving average
- Price vs DMA50: Stock trading {"above" if current_dma50 != "N/A" and float(last_price) > float(current_dma50) else "at"} 50-day moving average
"""
        return formatted_data


# Global instance for easy access
price_scraper = PriceDataScraper()


async def get_company_price_data(
    company_symbol: str,
    days_range: Optional[str] = "365"
) -> Optional[str]:
    """
    High-level function to get formatted price data for RAG integration.
    
    Args:
        company_symbol: Company symbol on screener.in
        days_range: Time range for data (30, 180, 365, etc.)
        
    Returns:
        Formatted price data string for RAG context
    """
    try:
        logger.info(f"Getting price data for {company_symbol} with range {days_range}")
        chart_data = await price_scraper.get_price_data(company_symbol, days_range)
        
        if chart_data:
            logger.info(f"Successfully retrieved chart data for {company_symbol}")
            formatted_data = price_scraper.format_price_data_for_rag(chart_data)
            logger.info(f"Successfully formatted price data for {company_symbol}")
            return formatted_data
        else:
            error_msg = f"Could not retrieve price data for {company_symbol}"
            logger.error(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error retrieving price data: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return error_msg
