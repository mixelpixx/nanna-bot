# stock_ai_analysis.py

import logging
import google.generativeai as generative_ai
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s',
                   handlers=[
                       logging.FileHandler('stock_ai.log'),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

class StockAIAnalyst:
    def __init__(self, api_key):
        """Initialize the AI analyst with API key."""
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        try:
            generative_ai.configure(api_key=api_key)
            self.model = generative_ai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=self.generation_config
            )
            self._test_connection()
        except Exception as exception:
            logger.error(f"Failed to initialize Gemini API: {exception}")
            raise

    def _test_connection(self):
        """Test the API connection."""
        try:
            response = self.model.generate_content("Test connection")
            if not response:
                raise Exception("No response from Gemini API")
        except Exception as exception:
            logger.error(f"Connection test failed: {exception}")
            raise

    def format_data_for_analysis(self, symbol, details, historical_data, greeks):
        """Format stock data for AI analysis."""
        try:
            recent_data = historical_data.tail(5).to_dict(orient='records')
            current_price = historical_data['close'].iloc[-1]
            price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0] * 100
            volume_average = historical_data['volume'].mean()
            
            return f"""
            Stock Analysis Request for {symbol}
            
            Basic Information:
            - Company: {details.get('longName', 'N/A')}
            - Current Price: ${current_price:.2f}
            - Market Cap: ${details.get('marketCap', 'N/A'):,.2f}
            - Exchange: {details.get('exchange', 'N/A')}
            
            Recent Performance:
            - Price Change: {price_change:.2f}%
            - Average Volume: {volume_average:,.0f}
            
            Technical Indicators:
            - 5-day High: ${historical_data['high'].tail(5).max():.2f}
            - 5-day Low: ${historical_data['low'].tail(5).min():.2f}
            
            Options Analysis:
            - Delta: {greeks['delta']:.4f}
            - Gamma: {greeks['gamma']:.4f}
            - Theta: {greeks['theta']:.4f}
            - Vega: {greeks['vega']:.4f}
            
            Recent Trading Data:
            {recent_data}
            
            Please provide:
            1. Technical Analysis: Review recent price action and volume patterns
            2. Options Market Sentiment: Interpret the Greeks values
            3. Key Risks and Opportunities
            4. Short-term and Long-term Outlook
            """
        except Exception as exception:
            logger.error(f"Error formatting data: {exception}")
            logger.debug(f"Data formatting failed with details: {str(exception)}")
            raise

    def analyze_stock(self, symbol, details, historical_data, greeks):
        """
        Perform AI analysis on stock data.
        
        Args:
            symbol (str): Stock symbol
            details (dict): Stock details from yfinance
            historical_data (pd.DataFrame): Historical price data
            greeks (dict): Option Greeks values
            
        Returns:
            str: AI-generated analysis
        """
        try:
            # Format the data
            logger.debug(f"Starting AI analysis for {symbol}")
            prompt = self.format_data_for_analysis(symbol, details, historical_data, greeks)
            
            # Get analysis from AI
            logger.debug("Sending request to Gemini API")
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                logger.error("Received empty response from Gemini API")
                raise Exception("No analysis generated")
                
            return response.text.strip()
            
        except Exception as exception:
            logger.error(f"Error in AI analysis: {exception}")
            raise
