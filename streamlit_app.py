import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import datetime
import time
import numpy as np
import yfinance as yf
from curl_cffi import requests as curl_requests

# Set page config
st.set_page_config(
    page_title="Futures Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📈 Futures Data Dashboard")
st.markdown("Interactive dashboard for analyzing crude oil futures data from Barchart")

# Sidebar controls
st.sidebar.header("Settings")
num_contracts = st.sidebar.slider("Number of contracts to analyze", 5, 20, 10)
auto_refresh = st.sidebar.checkbox("Auto refresh (30s)", value=False)

# Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_futures_data():
    """Fetch futures contract data from Barchart API"""
    
    # URLs and headers
    geturl = 'https://www.barchart.com/futures/quotes/CLF25/options/jan-25?futuresOptionsView=merged'
    apiurl = 'https://www.barchart.com/proxies/core-api/v1/quotes/get'

    getheaders = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
    }

    getpay = {'page': 'all'}

    try:
        s = requests.Session()
        r = s.get(geturl, params=getpay, headers=getheaders)

        headers = {
            'accept': 'application/json',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'referer': 'https://www.barchart.com/futures/quotes/CLF25/options/oct-24?futuresOptionsView=merged',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36',
            'x-xsrf-token': unquote(unquote(s.cookies.get_dict()['XSRF-TOKEN']))
        }

        payload = {
            'fields': 'symbol,contractSymbol,lastPrice,priceChange,openPrice,highPrice,lowPrice,previousPrice,volume,openInterest,tradeTime,symbolCode,symbolType,hasOptions',
            'list': 'futures.contractInRoot',
            'root': 'CL',
            'meta': 'field.shortName,field.type,field.description',
            'hasOptions': 'true',
            'raw': '1'
        }

        r = s.get(apiurl, params=payload, headers=headers)
        j = r.json()

        # Convert JSON to DataFrame
        df = pd.json_normalize(j['data'])
        df = df.drop(0).reset_index(drop=True)
        
        # Clean the 'lastPrice' column
        df['lastPrice'] = df['lastPrice'].str.replace('s', '', regex=False).astype(float)
        df['priceChange'] = pd.to_numeric(df['priceChange'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching futures data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_oil_data(ticker='CL=F', start_date='2020-01-01', end_date=None):
    """Load oil data using yfinance with curl_cffi session"""
    try:
        # Create a curl_cffi session that mimics Chrome
        session = curl_requests.Session(impersonate="chrome")
        
        # If no end_date provided, use current date
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date, session=session)
        if data.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns, keeping only the first level (OHLCV names)
            data.columns = data.columns.get_level_values(0)
        
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        st.error(f"Error loading oil data: {str(e)}")
        return pd.DataFrame()
@st.cache_data(ttl=300)
def fetch_options_data(symbols):
    """Fetch options data for given symbols"""
    
    base_url = "https://www.barchart.com/futures/quotes/{}/options/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.google.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    data = []
    progress_bar = st.progress(0)
    
    for i, contract in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        
        url = base_url.format(contract)
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                toolbar = soup.find('div', class_='row bc-options-toolbar__second-row')

                if toolbar:
                    columns = toolbar.find_all('div', class_='column')
                    
                    if len(columns) >= 2:
                        days_to_expiry = columns[0].get_text(strip=True)
                        implied_volatility = columns[1].get_text(strip=True)

                        # Extract numerical values
                        try:
                            expiration_date = days_to_expiry.split()[3][2:]
                            days_to_expiry_num = int(days_to_expiry.split()[0])
                            implied_volatility_num = float(implied_volatility.split(":")[1].replace("%", "").strip())
                            
                            data.append([contract, expiration_date, days_to_expiry_num, implied_volatility_num])
                        except (IndexError, ValueError) as e:
                            st.warning(f"Could not parse data for {contract}: {str(e)}")
                            
        except requests.exceptions.RequestException as e:
            st.warning(f"Request failed for {contract}: {str(e)}")
        
        # Small delay to be respectful to the server
        time.sleep(0.5)
    
    progress_bar.empty()
    
    if data:
        options_df = pd.DataFrame(data, columns=["Contract", "Expiration Date", "Options Days to Expiry", "Futures Implied Volatility"])
        return options_df
    else:
        return pd.DataFrame()

# Main app logic
def main():
    # Fetch data
    with st.spinner("Fetching futures data..."):
        df = fetch_futures_data()
    
    if df is None or df.empty:
        st.error("Failed to fetch futures data. Please try again later.")
        return
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Contracts", len(df))
    
    with col2:
        avg_price = df['lastPrice'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col3:
        total_volume = pd.to_numeric(df['volume'], errors='coerce').sum()
        st.metric("Total Volume", f"{total_volume:,.0f}")
    
    with col4:
        total_open_interest = pd.to_numeric(df['openInterest'], errors='coerce').sum()
        st.metric("Total Open Interest", f"{total_open_interest:,.0f}")

    # Fetch options data upfront for the forward curve
    first_symbols = df['symbol'].iloc[:num_contracts].tolist()
    
    with st.spinner("Fetching options data for forward curve..."):
        options_df = fetch_options_data(first_symbols)
    
    # Merge futures and options data
    if not options_df.empty:
        merged_df = options_df.merge(df[['symbol', 'lastPrice', 'priceChange', 'volume', 'openInterest']], 
                                   left_on='Contract', 
                                   right_on='symbol', 
                                   how='left')
        merged_df = merged_df.rename(columns={'lastPrice': 'Last Price'})
    else:
        # Fallback if options data fails
        merged_df = df[['symbol', 'lastPrice', 'priceChange', 'volume', 'openInterest']].head(num_contracts).copy()
        merged_df = merged_df.rename(columns={'symbol': 'Contract', 'lastPrice': 'Last Price'})
        merged_df['Expiration Date'] = 'N/A'
        merged_df['Options Days to Expiry'] = 0
        merged_df['Futures Implied Volatility'] = 0

    # Create tabs
    tab1, tab2 = st.tabs(["Futures Curve with Options", "Front Month Data"])
    
    with tab1:
        st.subheader("Futures Curve with Options Data")
        
        # Display the merged data table
        st.dataframe(merged_df[['Contract', 'Last Price', 'Expiration Date', 'Options Days to Expiry', 'Futures Implied Volatility']], 
                    use_container_width=True)
        
        # Forward curve chart with options data
        fig_curve = go.Figure()
        
        # Add price line
        fig_curve.add_trace(go.Scatter(
            x=merged_df['Contract'],
            y=merged_df['Last Price'],
            mode='lines+markers',
            name='Last Price',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig_curve.update_layout(
            title='Crude Oil Futures Forward Curve',
            xaxis_title='Contract Symbol',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)
        
        # Additional charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Days to expiry chart
            if 'Options Days to Expiry' in merged_df.columns and merged_df['Options Days to Expiry'].sum() > 0:
                fig_days = px.bar(merged_df,
                                 x='Contract',
                                 y='Options Days to Expiry',
                                 title='Days to Options Expiry',
                                 color='Options Days to Expiry',
                                 color_continuous_scale='viridis')
                st.plotly_chart(fig_days, use_container_width=True)
            else:
                st.info("Options expiry data not available")
        
        with col2:
            # Implied volatility chart
            if 'Futures Implied Volatility' in merged_df.columns and merged_df['Futures Implied Volatility'].sum() > 0:
                fig_iv = px.bar(merged_df,
                               x='Contract',
                               y='Futures Implied Volatility',
                               title='Implied Volatility (%)',
                               color='Futures Implied Volatility',
                               color_continuous_scale='RdYlBu')
                st.plotly_chart(fig_iv, use_container_width=True)
            else:
                st.info("Implied volatility data not available")
    
    with tab2:
        st.subheader("WTI Crude Oil Price")
        
        # Date range selector
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.datetime(2020, 1, 1),
                max_value=datetime.datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.datetime.now(),
                max_value=datetime.datetime.now()
            )
        
        # Load oil data button
        if st.button("Load WTI Data", type="primary"):
            with st.spinner("Loading WTI data..."):
                df_wti = load_oil_data('CL=F', start_date=start_date.strftime('%Y-%m-%d'), 
                                     end_date=end_date.strftime('%Y-%m-%d'))
            
            if not df_wti.empty and len(df_wti) > 0 and 'Close' in df_wti.columns:
                st.success(f"Successfully loaded {len(df_wti)} days of WTI data")
                
                # Calculate metrics
                latest_price = df_wti['Close'].dropna().iloc[-1] if not df_wti['Close'].dropna().empty else 0
                price_change = (df_wti['Close'].dropna().iloc[-1] - df_wti['Close'].dropna().iloc[-2]) if len(df_wti['Close'].dropna()) > 1 else 0
                
                # Calculate 52-week (252 trading days) high and low
                trading_days_52w = min(252, len(df_wti))
                recent_52w = df_wti.tail(trading_days_52w)
                high_52w = recent_52w['High'].max() if 'High' in df_wti.columns else latest_price
                low_52w = recent_52w['Low'].min() if 'Low' in df_wti.columns else latest_price
                
                avg_volume = df_wti['Volume'].mean() if 'Volume' in df_wti.columns and df_wti['Volume'].notna().any() else 0
                
                # Calculate volatility for different periods
                volatility_1m = volatility_3m = volatility_1y = 0
                
                if len(df_wti['Close'].dropna()) > 1:
                    # 1 Month volatility (21 trading days)
                    if len(df_wti) >= 21:
                        vol_1m_data = df_wti['Close'].tail(21).pct_change().std() * np.sqrt(252) * 100
                        volatility_1m = vol_1m_data
                    
                    # 3 Month volatility (63 trading days)
                    if len(df_wti) >= 63:
                        vol_3m_data = df_wti['Close'].tail(63).pct_change().std() * np.sqrt(252) * 100
                        volatility_3m = vol_3m_data
                    
                    # 1 Year volatility (252 trading days)
                    trading_days_1y = min(252, len(df_wti))
                    vol_1y_data = df_wti['Close'].tail(trading_days_1y).pct_change().std() * np.sqrt(252) * 100
                    volatility_1y = vol_1y_data
                
                # Display metrics in two rows
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${latest_price:.2f}", delta=f"{price_change:+.2f}")
                
                with col2:
                    st.metric("52W High", f"${high_52w:.2f}")
                
                with col3:
                    st.metric("52W Low", f"${low_52w:.2f}")
                
                with col4:
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
                
                # Volatility metrics row
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.metric("1M Volatility", f"{volatility_1m:.1f}%")
                
                with col6:
                    st.metric("3M Volatility", f"{volatility_3m:.1f}%")
                
                with col7:
                    st.metric("1Y Volatility", f"{volatility_1y:.1f}%")
                
                with col8:
                    st.metric("", "")  # Empty for spacing
                
                # Chart period selector
                st.subheader("Chart Period")
                chart_period = st.selectbox(
                    "Select time period:",
                    options=["3 Months", "YTD", "Full Period"],
                    index=2  # Default to Full Period
                )
                
                # Filter data based on selected period
                if chart_period == "3 Months":
                    # Show last 3 months (approximately 63 trading days)
                    chart_data = df_wti.tail(min(63, len(df_wti)))
                    chart_title = "WTI Crude Oil Price - Last 3 Months"
                elif chart_period == "YTD":
                    # Show Year-to-Date data
                    current_year = datetime.datetime.now().year
                    ytd_start = datetime.datetime(current_year, 1, 1)
                    chart_data = df_wti[df_wti['Date'] >= ytd_start]
                    chart_title = f"WTI Crude Oil Price - YTD {current_year}"
                else:
                    # Show full period
                    chart_data = df_wti
                    chart_title = "WTI Crude Oil Price - Full Period"
                
                # Line chart
                fig_price = px.line(chart_data,
                                   x='Date',
                                   y='Close',
                                   title=chart_title,
                                   labels={'Close': 'Price ($)', 'Date': 'Date'})
                
                fig_price.update_layout(
                    height=600,
                    showlegend=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Store in session state
                st.session_state['wti_data'] = df_wti
                
            else:
                st.error("Failed to load WTI data. Please try again.")
        
        # Show cached data if available
        elif 'wti_data' in st.session_state:
            st.info("Showing previously loaded data. Click 'Load WTI Data' to refresh.")
            df_wti = st.session_state['wti_data']
            
            if not df_wti.empty and 'Close' in df_wti.columns:
                # Show metrics for cached data too
                latest_price = df_wti['Close'].dropna().iloc[-1] if not df_wti['Close'].dropna().empty else 0
                price_change = (df_wti['Close'].dropna().iloc[-1] - df_wti['Close'].dropna().iloc[-2]) if len(df_wti['Close'].dropna()) > 1 else 0
                high_52w = df_wti['High'].max() if 'High' in df_wti.columns else latest_price
                low_52w = df_wti['Low'].min() if 'Low' in df_wti.columns else latest_price
                avg_volume = df_wti['Volume'].mean() if 'Volume' in df_wti.columns and df_wti['Volume'].notna().any() else 0
                
                if len(df_wti['Close'].dropna()) > 1:
                    volatility = df_wti['Close'].pct_change().std() * np.sqrt(252) * 100
                else:
                    volatility = 0
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Current Price", f"${latest_price:.2f}", delta=f"{price_change:+.2f}")
                
                with col2:
                    st.metric("52W High", f"${high_52w:.2f}")
                
                with col3:
                    st.metric("52W Low", f"${low_52w:.2f}")
                
                with col4:
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
                
                with col5:
                    st.metric("Volatility", f"{volatility:.1f}%")
                
                fig_price = px.line(df_wti,
                               x='Date',
                               y='Close',
                               title='WTI Crude Oil Price (Cached Data)')
                fig_price.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_price, use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("**Data Sources:** Barchart.com, Yahoo Finance | **Last Updated:** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))