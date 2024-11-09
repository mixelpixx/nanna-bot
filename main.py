import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
from polygon.rest import RESTClient
from polygon import exceptions as polygon_exceptions
import yfinance as yf
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_ai_analysis import StockAIAnalyst

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s',
                    handlers=[
                        logging.FileHandler('stock_app.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Hide Streamlit Components ---
st.set_page_config(page_title="Stock Analysis App", layout="wide")
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        border: none;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    div.stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 8px;
        border-radius: 4px;
    }
    .main > div {
        padding: 2rem 3rem;
    }
    .stTab {
        background-color: #ffffff;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    /* Additional styling for AI analysis section */
    .ai-analysis {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 20px 0;
    }
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- API Configuration ---
def get_api_key(key_name):
    """Get API key from environment variables."""
    try:
        # First try to get from environment variables
        api_key = os.getenv(key_name)
        if api_key:
            return api_key
            
        # Only try secrets as fallback if available
        try:
            if hasattr(st, "secrets") and key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass  # Ignore secrets-related errors
            
        # If we get here, no valid key was found
        if not api_key:
            raise ValueError(f"Missing {key_name} - Please set this environment variable")
            
    except Exception as e:
        logger.error(f"Error getting API key for {key_name}: {str(e)}")
        raise ValueError(f"Failed to get {key_name}: {str(e)}")

def initialize_apis():
    """Initialize API clients."""
    try:
        # Get API keys
        polygon_api_key = get_api_key("POLYGON_API_KEY")
        gemini_api_key = get_api_key("GEMINI_API_KEY")
        
        if not polygon_api_key or not gemini_api_key:
            st.error("Missing required API keys. Please check environment variables.")
            st.stop()
            
        # Initialize clients
        polygon_client = RESTClient(api_key=polygon_api_key)
        ai_analyst = StockAIAnalyst(gemini_api_key)
        
        return polygon_client, ai_analyst
        
    except Exception as e:
        st.error(f"Failed to initialize APIs: {str(e)}")
        st.stop()

# --- Stock Data Functions ---
def search_stock_symbol(company_name):
    if not company_name.strip():
        return None
    try:
        ticker = yf.Ticker(company_name)
        info = ticker.info
        return ticker.ticker if info else None
    except Exception as e:
        logger.error(f"Error searching stock symbol: {e}", exc_info=True)
        return None

def get_stock_details(symbol):
    if not symbol.strip():
        return None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info if info else None
    except Exception as e:
        logger.error(f"Error fetching stock details: {e}", exc_info=True)
        return None

def get_historical_data(client, symbol, start_date, end_date, timeframe):
    if not all([symbol, start_date, end_date, timeframe]):
        return None
        
    try:
        timeframe_mapping = {
            "1 Day": (1, "day"),
            "1 Hour": (1, "hour"),
            "30 Minutes": (30, "minute"),
            "15 Minutes": (15, "minute"),
            "5 Minutes": (5, "minute"),
            "1 Minute": (1, "minute")
        }
        
        multiplier, span = timeframe_mapping.get(timeframe, (1, "day"))
        
        data = list(client.list_aggs(
            ticker=symbol,
            multiplier=multiplier,
            timespan=span,
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            limit=50000
        ))
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}", exc_info=True)
        return None

def get_option_greeks(symbol):
    try:
        ticker = yf.Ticker(symbol)
        options = ticker.options
        
        if not options:
            return None
            
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        expiration = options[0]
        option_chain = ticker.option_chain(expiration)
        
        if option_chain.calls.empty:
            return None
            
        call = option_chain.calls.iloc[0]
        S = current_price
        K = call['strike']
        T = 30 / 365
        r = 0.01
        sigma = call['impliedVolatility']
        
        return {
            'delta': delta('c', S, K, T, r, sigma),
            'gamma': gamma('c', S, K, T, r, sigma),
            'theta': theta('c', S, K, T, r, sigma),
            'vega': vega('c', S, K, T, r, sigma)
        }
    except Exception as e:
        logger.error(f"Error calculating greeks: {e}", exc_info=True)
        return None

# --- Visualization Functions ---
def create_combined_chart(df, symbol):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume'),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        title_text=f"{symbol} Stock Analysis",
        showlegend=False
    )
    
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

# --- Main Application ---
def main():
    # Initialize APIs
    polygon_client, ai_analyst = initialize_apis()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Analysis", "Settings"])
    
    with tab1:
        st.title("Stock Analysis Application")
        
        # Search and Input
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Enter company name to search for symbol")
            if st.button("Search Symbol"):
                if company_name:
                    symbol = search_stock_symbol(company_name)
                    if symbol:
                        st.success(f"Symbol for {company_name}: {symbol}")
                    else:
                        st.error(f"Could not find symbol for {company_name}")

        with col2:
            symbol = st.text_input("Enter a stock symbol", "AAPL")
            
        # Analysis Parameters
        col3, col4 = st.columns(2)
        with col3:
            date_range = st.selectbox(
                "Select Date Range",
                ["1 Day", "3 Days", "1 Month", "3 Months", "1 Year"]
            )
            
        with col4:
            timeframe = st.selectbox(
                "Select Timeframe for Graph",
                ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "1 Day"]
            )
            
        # Calculate dates
        end_date = datetime.now()
        date_mapping = {
            "1 Day": timedelta(days=1),
            "3 Days": timedelta(days=3),
            "1 Month": timedelta(days=30),
            "3 Months": timedelta(days=90),
            "1 Year": timedelta(days=365)
        }
        start_date = end_date - date_mapping[date_range]
        
        # Analysis Button
        if st.button("Analyze Stock"):
            if not symbol.strip():
                st.error("Please enter a stock symbol.")
                return
                
            with st.spinner("Analyzing stock data..."):
                try:
                    # Fetch all data
                    details = get_stock_details(symbol)
                    historical_data = get_historical_data(polygon_client, symbol, start_date, end_date, timeframe)
                    greeks = get_option_greeks(symbol)
                    
                    if all([details, historical_data is not None, not historical_data.empty, greeks]):
                        # Display results
                        st.subheader("Stock Details")
                        col5, col6, col7, col8 = st.columns(4)
                        with col5:
                            st.metric("Symbol", details.get('symbol', 'N/A'))
                        with col6:
                            st.metric("Company", details.get('longName', 'N/A'))
                        with col7:
                            st.metric("Market Cap", f"${details.get('marketCap', 0):,.0f}")
                        with col8:
                            st.metric("Exchange", details.get('exchange', 'N/A'))
                        
                        st.subheader("Historical Data Analysis")
                        fig = create_combined_chart(historical_data, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col9, col10 = st.columns(2)
                        with col9:
                            st.subheader("Basic Statistics")
                            st.write(historical_data['close'].describe())
                            
                        with col10:
                            st.subheader("Option Greeks")
                            st.write(f"Delta: {greeks['delta']:.4f}")
                            st.write(f"Gamma: {greeks['gamma']:.4f}")
                            st.write(f"Theta: {greeks['theta']:.4f}")
                            st.write(f"Vega: {greeks['vega']:.4f}")
                        
                        # AI Analysis
                        if st.button("Generate AI Analysis"):
                            with st.spinner("Generating AI analysis..."):
                                try:
                                    ai_analysis = ai_analyst.analyze_stock(
                                        symbol,
                                        details,
                                        historical_data,
                                        greeks
                                    )
                                    if ai_analysis:
                                        st.markdown("<div class='ai-analysis'>", unsafe_allow_html=True)
                                        st.subheader("AI-Generated Stock Analysis")
                                        st.markdown(ai_analysis)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    else:
                                        st.error("Failed to generate AI analysis.")
                                except Exception as e:
                                    logger.error(f"AI analysis error: {e}", exc_info=True)
                                    st.error(f"Error generating AI analysis: {str(e)}")
                        
                        # Download button
                        csv = historical_data.to_csv(index=False)
                        st.download_button(
                            label="Download Historical Data as CSV",
                            data=csv,
                            file_name=f"{symbol}_historical_data.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("Insufficient data to perform complete analysis.")
                        
                except Exception as e:
                    logger.error(f"Analysis error: {e}", exc_info=True)
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    with tab2:
        st.header("Settings & Information")
        st.write("This application uses data from:")
        st.write("- Polygon.io for real-time market data")
        st.write("- yfinance for additional stock information")
        st.write("- Google's Gemini for AI-powered analysis")
        
        st.subheader("API Configuration")
        st.write("Ensure you have the following environment variables set:")
        st.code("""
        POLYGON_API_KEY=your_polygon_api_key
        GEMINI_API_KEY=your_gemini_api_key
        """)
        
        st.subheader("Data Usage")
        st.write("Please ensure you comply with the terms of service of all data providers.")

if __name__ == "__main__":
    main()