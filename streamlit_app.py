import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
import datetime
import time
import numpy as np
import yfinance as yf
from curl_cffi import requests as curl_requests
from scipy import stats
from scipy.interpolate import interp1d
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# Set page config with enhanced styling
st.set_page_config(
    page_title="CrudeLabs",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better dark theme integration
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #615fff 0%, #4338ca 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #0f172b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #314158;
        margin: 0.5rem 0;
    }
    .analysis-section {
        background: #1d293d;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #314158;
        margin: 1rem 0;
    }
    .risk-alert {
        background: #dc2626;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .bullish-signal {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .neutral-signal {
        background: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced title with gradient background
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">📊 Crude Labs</h1>
    <p style="color: #e2e8f0; margin: 0.5rem 0 0 0;">Comprehensive crude oil futures and options analysis platform</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with better organization
st.sidebar.markdown("### ⚙️ Dashboard Settings")
num_contracts = st.sidebar.slider("📈 Contracts to Analyze", 5, 25, 12, help="Number of futures contracts to include in analysis")
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
advanced_mode = st.sidebar.checkbox("🎯 Advanced Analytics", value=True, help="Enable sophisticated analysis features")

st.sidebar.markdown("### 📊 Analysis Parameters")
vol_window = st.sidebar.slider("Volatility Window (days)", 10, 60, 21)
risk_threshold = st.sidebar.slider("Risk Alert Threshold (%)", 1, 10, 5)

# Add refresh button with enhanced styling
if st.sidebar.button("🔄 Refresh All Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Advanced analytics functions
@st.cache_data(ttl=300)
def calculate_advanced_metrics(df, price_col='lastPrice'):
    """Calculate advanced market metrics"""
    prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
    
    if len(prices) < 2:
        return {}
    
    # Price momentum
    momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
    
    # Contango/Backwardation analysis
    front_month = prices.iloc[0] if len(prices) > 0 else 0
    back_month = prices.iloc[-1] if len(prices) > 1 else front_month
    curve_slope = (back_month - front_month) / len(prices) if len(prices) > 1 else 0
    
    market_structure = "Contango" if curve_slope > 0 else "Backwardation" if curve_slope < 0 else "Flat"
    
    # Calculate volatility across the curve
    price_volatility = prices.std() / prices.mean() * 100 if prices.mean() != 0 else 0
    
    return {
        'momentum': momentum,
        'curve_slope': curve_slope,
        'market_structure': market_structure,
        'price_volatility': price_volatility,
        'front_month_price': front_month,
        'back_month_price': back_month
    }

@st.cache_data(ttl=300)
def calculate_greeks_approximation(options_df, underlying_price, risk_free_rate=0.05):
    """Approximate options Greeks calculation"""
    if options_df.empty:
        return options_df
    
    df = options_df.copy()
    
    # Ensure numeric types
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['bidPrice'] = pd.to_numeric(df['bidPrice'], errors='coerce')
    
    # Simple Delta approximation (closer to ATM = higher delta)
    df['moneyness'] = df['strike'] / underlying_price
    df['delta_approx'] = np.where(
        df['optionType'] == 'Call',
        np.maximum(0, 1 - abs(df['moneyness'] - 1) * 2),
        np.maximum(0, abs(df['moneyness'] - 1) * 2 - 1)
    )
    
    # Gamma approximation (highest at ATM)
    df['gamma_approx'] = np.exp(-((df['moneyness'] - 1) ** 2) * 10) * 0.1
    
    return df

@st.cache_data(ttl=300)
def analyze_options_flow(calls_df, puts_df):
    """Analyze options flow and sentiment"""
    if calls_df.empty and puts_df.empty:
        return {}
    
    # Volume analysis
    call_volume = calls_df['volume'].sum() if not calls_df.empty and 'volume' in calls_df.columns else 0
    put_volume = puts_df['volume'].sum() if not puts_df.empty and 'volume' in puts_df.columns else 0
    
    # Open Interest analysis
    call_oi = calls_df['openInterest'].sum() if not calls_df.empty and 'openInterest' in calls_df.columns else 0
    put_oi = puts_df['openInterest'].sum() if not puts_df.empty and 'openInterest' in puts_df.columns else 0
    
    # Calculate ratios
    put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
    oi_ratio = put_oi / call_oi if call_oi > 0 else 0
    
    # Sentiment analysis
    if put_call_ratio > 1.2:
        sentiment = "Bearish"
    elif put_call_ratio < 0.8:
        sentiment = "Bullish" 
    else:
        sentiment = "Neutral"
    
    return {
        'call_volume': call_volume,
        'put_volume': put_volume,
        'call_oi': call_oi,
        'put_oi': put_oi,
        'put_call_ratio': put_call_ratio,
        'oi_ratio': oi_ratio,
        'sentiment': sentiment
    }

# Original data fetching functions (keeping the same)
@st.cache_data(ttl=300)
def fetch_detailed_options_data(symbols):
    """Fetch detailed options data for given symbols using the provided method"""
    url = 'https://www.barchart.com/proxies/core-api/v1/quotes/get?'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    }
    
    try:
        with requests.Session() as req:
            req.headers.update(headers)
            
            # Get XSRF token
            r = req.get(url[:25])
            if 'XSRF-TOKEN' in r.cookies.get_dict():
                req.headers.update({'X-XSRF-TOKEN': unquote(r.cookies.get_dict()['XSRF-TOKEN'])})
            
            results = {}
            
            for symbol in symbols:
                params = {
                    "symbol": symbol,
                    "list": "futures.options",
                    "fields": "strike,openPrice,highPrice,lowPrice,lastPrice,priceChange,bidPrice,askPrice,volume,openInterest,premium,tradeTime,longSymbol,optionType,symbol",
                    "orderBy": "strike",
                    "orderDir": "asc",
                    "meta": "field.shortName,field.description,field.type",
                    "futureOptions": "true",
                    "noPagination": "true",
                    "showExpandLink": "false"
                }
                
                r = req.get(url, params=params)
                
                if r.status_code == 200:
                    data = r.json()
                    
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])
                        
                        # Clean and process the data
                        df['strike'] = df['strike'].astype(str).str.replace(r'[CP]', '', regex=True)
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).astype(int)
                        df['bidPrice'] = df['bidPrice'].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
                        df['bidPrice'] = pd.to_numeric(df['bidPrice'], errors='coerce').fillna(0)
                        
                        # Clean other price fields
                        for col in ['priceChange', 'bidPrice', 'askPrice']:
                            if col in df.columns:
                                df[col] = df[col].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
                                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                        # Separate calls and puts
                        calls_df = df[df['optionType'] == 'Call'].copy()
                        puts_df = df[df['optionType'] == 'Put'].copy()
                        
                        # Sort by strike price
                        calls_df['strike_num'] = pd.to_numeric(calls_df['strike'], errors='coerce')
                        puts_df['strike_num'] = pd.to_numeric(puts_df['strike'], errors='coerce')
                        
                        calls_df = calls_df.sort_values('strike_num').drop('strike_num', axis=1)
                        puts_df = puts_df.sort_values('strike_num').drop('strike_num', axis=1)
                        
                        results[symbol] = (calls_df, puts_df)
                    else:
                        results[symbol] = (pd.DataFrame(), pd.DataFrame())
                else:
                    st.warning(f"Failed to fetch options data for {symbol}. Status code: {r.status_code}")
                    results[symbol] = (pd.DataFrame(), pd.DataFrame())
            
            return results
            
    except Exception as e:
        st.error(f"Error fetching detailed options data: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def fetch_futures_data():
    """Fetch futures contract data from Barchart API"""
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
        
        # Clean the data
        df['lastPrice'] = df['lastPrice'].str.replace('s', '', regex=False).astype(float)
        df['priceChange'] = pd.to_numeric(df['priceChange'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching futures data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_oil_data(ticker='CL=F', start_date='2020-01-01', end_date=None):
    """Load oil data using yfinance with curl_cffi session"""
    try:
        session = curl_requests.Session(impersonate="chrome")
        
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date, session=session)
        if data.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
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

                        try:
                            expiration_date = days_to_expiry.split()[3][2:]
                            days_to_expiry_num = int(days_to_expiry.split()[0])
                            implied_volatility_num = float(implied_volatility.split(":")[1].replace("%", "").strip())
                            
                            data.append([contract, expiration_date, days_to_expiry_num, implied_volatility_num])
                        except (IndexError, ValueError) as e:
                            st.warning(f"Could not parse data for {contract}: {str(e)}")
                            
        except requests.exceptions.RequestException as e:
            st.warning(f"Request failed for {contract}: {str(e)}")
        
        time.sleep(0.3)  # Reduced delay for better UX
    
    progress_bar.empty()
    
    if data:
        options_df = pd.DataFrame(data, columns=["Contract", "Expiration Date", "Options Days to Expiry", "Futures Implied Volatility"])
        return options_df
    else:
        return pd.DataFrame()

# Main application logic
def main():
    # Fetch core data
    with st.spinner("🔄 Loading futures data..."):
        df = fetch_futures_data()
    
    if df is None or df.empty:
        st.error("❌ Failed to fetch futures data. Please try again later.")
        return
    
    # Calculate advanced metrics
    if advanced_mode:
        with st.spinner("🧠 Calculating advanced analytics..."):
            advanced_metrics = calculate_advanced_metrics(df.head(num_contracts))
    
    # Enhanced header metrics with better styling
    st.markdown("### 📊 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        front_month_contract = df['symbol'].iloc[0] if not df.empty else "N/A"
        front_month_price = df['lastPrice'].iloc[0] if not df.empty else 0
        price_change = df['priceChange'].iloc[0] if not df.empty and 'priceChange' in df.columns else 0
        st.metric("🛢️ Front Month", f"{front_month_contract}", delta=None)
        st.metric("💰 Price", f"${front_month_price:.2f}", delta=f"{price_change:+.2f}")
    
    with col2:
        if advanced_mode and advanced_metrics:
            market_structure = advanced_metrics.get('market_structure', 'Unknown')
            curve_slope = advanced_metrics.get('curve_slope', 0)
            
            structure_color = "🟢" if market_structure == "Contango" else "🔴" if market_structure == "Backwardation" else "🟡"
            st.metric("📈 Market Structure", f"{structure_color} {market_structure}")
            st.metric("📐 Curve Slope", f"{curve_slope:.3f}")
    
    with col3:
        total_volume = df['volume'].sum() if 'volume' in df.columns else 0
        total_oi = df['openInterest'].sum() if 'openInterest' in df.columns else 0
        st.metric("📊 Total Volume", f"{total_volume:,.0f}")
        st.metric("🔢 Total OI", f"{total_oi:,.0f}")
    
    with col4:
        if advanced_mode and advanced_metrics:
            momentum = advanced_metrics.get('momentum', 0)
            volatility = advanced_metrics.get('price_volatility', 0)
            
            momentum_emoji = "🚀" if momentum > 2 else "📉" if momentum < -2 else "➡️"
            st.metric("⚡ Momentum", f"{momentum_emoji} {momentum:.2f}%")
            st.metric("📊 Volatility", f"{volatility:.2f}%")

    # Risk alerts
    if advanced_mode and advanced_metrics:
        risk_level = max(abs(advanced_metrics.get('momentum', 0)), advanced_metrics.get('price_volatility', 0))
        if risk_level > risk_threshold:
            st.markdown(f"""
            <div class="risk-alert">
                ⚠️ <strong>Risk Alert:</strong> High volatility detected ({risk_level:.1f}% > {risk_threshold}% threshold)
            </div>
            """, unsafe_allow_html=True)

    # Fetch options data for analysis
    first_symbols = df['symbol'].iloc[:num_contracts].tolist()
    
    with st.spinner("📈 Fetching options data..."):
        options_df = fetch_options_data(first_symbols)
    
    # Merge data
    if not options_df.empty:
        merged_df = options_df.merge(df[['symbol', 'lastPrice', 'priceChange', 'volume', 'openInterest']], 
                                   left_on='Contract', 
                                   right_on='symbol', 
                                   how='left')
        merged_df = merged_df.rename(columns={'lastPrice': 'Last Price'})
    else:
        merged_df = df[['symbol', 'lastPrice', 'priceChange', 'volume', 'openInterest']].head(num_contracts).copy()
        merged_df = merged_df.rename(columns={'symbol': 'Contract', 'lastPrice': 'Last Price'})
        merged_df['Expiration Date'] = 'N/A'
        merged_df['Options Days to Expiry'] = 0
        merged_df['Futures Implied Volatility'] = 0

    # Initialize session state for options data
    if "initial_options_data" not in st.session_state:
        if not merged_df.empty:
            initial_contracts = merged_df['Contract'].head(15).tolist()  # Increased to 15

            with st.spinner("🔄 Pre-loading options data..."):
                options_data = fetch_detailed_options_data(initial_contracts)

            for contract in initial_contracts:
                if options_data and contract in options_data:
                    st.session_state[f'options_data_{contract}'] = options_data[contract]

            if initial_contracts:
                st.session_state["selected_contract"] = initial_contracts[0]

            st.session_state["initial_options_data"] = True
            st.success("✅ Options data loaded successfully!")

    # Enhanced tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Analysis", 
        "⚡ Options Flow", 
        "📈 Price Discovery", 
        "🔍 Contract Deep Dive",
    ])
    
    # TAB 1: Enhanced Market Analysis
    with tab1:
        st.markdown("### 📊 Comprehensive Market Analysis")
        
        # Enhanced forward curve with multiple views
        fig_curve = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Forward Curve', 'Implied Volatility', 'Volume Profile', 'Open Interest'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Forward curve
        fig_curve.add_trace(
            go.Scatter(x=merged_df['Contract'], y=merged_df['Last Price'],
                      mode='lines+markers', name='Price',
                      line=dict(color='#615fff', width=3),
                      marker=dict(size=8, color='#615fff')),
            row=1, col=1
        )
        
        # Implied Volatility
        if 'Futures Implied Volatility' in merged_df.columns:
            fig_curve.add_trace(
                go.Bar(x=merged_df['Contract'], y=merged_df['Futures Implied Volatility'],
                      name='IV', marker_color='#f59e0b'),
                row=1, col=2
            )
        
        # Volume
        if 'volume' in merged_df.columns:
            fig_curve.add_trace(
                go.Bar(x=merged_df['Contract'], y=merged_df['volume'],
                      name='Volume', marker_color='#10b981'),
                row=2, col=1
            )
        
        # Open Interest
        if 'openInterest' in merged_df.columns:
            fig_curve.add_trace(
                go.Bar(x=merged_df['Contract'], y=merged_df['openInterest'],
                      name='OI', marker_color='#ef4444'),
                row=2, col=2
            )
        
        fig_curve.update_layout(
            height=700,
            title_text="Multi-Dimensional Market Analysis",
            showlegend=False,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)
        
        # Enhanced data table with conditional formatting
        st.markdown("#### 📋 Contract Details")
        
        # Add technical indicators to the display
        display_df = merged_df.copy()
        if advanced_mode:
            # Calculate relative strength
            display_df['Relative_Strength'] = (display_df['Last Price'] / display_df['Last Price'].mean() - 1) * 100
            
            # Add momentum indicators
            display_df['Price_Momentum'] = display_df['priceChange'] / display_df['Last Price'] * 100
        
        st.dataframe(
            display_df.style.format({
                'Last Price': '${:.2f}',
                'priceChange': '{:+.2f}',
                'Futures Implied Volatility': '{:.1f}%',
                'Relative_Strength': '{:+.1f}%',
                'Price_Momentum': '{:+.2f}%'
            }).background_gradient(subset=['Last Price'], cmap='RdYlGn'),
            use_container_width=True
        )

    # TAB 2: Options Flow Analysis
    with tab2:
        st.markdown("### ⚡ Options Flow & Sentiment Analysis")
        
        # Contract selector for options analysis
        cached_contracts = [c for c in merged_df['Contract'].head(15).tolist() 
                           if f'options_data_{c}' in st.session_state]
        
        if cached_contracts:
            selected_contract = st.selectbox(
                "🎯 Select Contract for Options Analysis:",
                options=cached_contracts,
                index=0
            )
            
            if f'options_data_{selected_contract}' in st.session_state:
                calls_df, puts_df = st.session_state[f'options_data_{selected_contract}']
                # Contract overview
                contract_data = merged_df[merged_df['Contract'] == selected_contract].iloc[0]
                
                # Calculate options flow metrics
                flow_metrics = analyze_options_flow(calls_df, puts_df)
                
                # Display sentiment dashboard
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment = flow_metrics.get('sentiment', 'Unknown')
                    sentiment_color = {
                        'Bullish': '#10b981',
                        'Bearish': '#ef4444', 
                        'Neutral': '#f59e0b'
                    }.get(sentiment, '#6b7280')
                    
                    st.markdown(f"""
                    <div style="background: {sentiment_color}; padding: 1rem; border-radius: 8px; text-align: center;">
                        <h3 style="color: white; margin: 0;">Market Sentiment</h3>
                        <h2 style="color: white; margin: 0;">{sentiment}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    pcr = flow_metrics.get('put_call_ratio', 0)
                    st.metric("📊 Put/Call Ratio", f"{pcr:.2f}")
                    st.metric("🔄 Volume Ratio", f"{flow_metrics.get('oi_ratio', 0):.2f}")
                
                with col3:
                    st.metric("📞 Call Volume", f"{flow_metrics.get('call_volume', 0):,}")
                    st.metric("📉 Put Volume", f"{flow_metrics.get('put_volume', 0):,}")
                
                colA, colB, colC= st.columns(3)

                with colA:
                    st.metric("📊 Underlying Price", f"${contract_data['Last Price']:.2f}")
                    st.metric("📈 Price Change", f"{contract_data.get('priceChange', 0):+.2f}")
                
                with colB:
                    days_to_expiry = contract_data.get('Options Days to Expiry', 0)
                    st.metric("⏰ Days to Expiry", f"{days_to_expiry}")
                    
                    iv = contract_data.get('Futures Implied Volatility', 0)
                    st.metric("📊 Implied Vol", f"{iv:.1f}%")
                
                with colC:
                    # Calculate moneyness for ATM options
                    underlying = contract_data['Last Price']
                    if not calls_df.empty:
                        calls_df['strike_num'] = pd.to_numeric(calls_df['strike'], errors='coerce')
                        atm_call = calls_df.loc[(calls_df['strike_num'] - underlying).abs().idxmin()]
                        st.metric("💰 ATM Call", f"${atm_call['bidPrice']:.2f}")
                    
                    if not puts_df.empty:
                        puts_df['strike_num'] = pd.to_numeric(puts_df['strike'], errors='coerce')
                        atm_put = puts_df.loc[(puts_df['strike_num'] - underlying).abs().idxmin()]
                        st.metric("💰 ATM Put", f"${atm_put['bidPrice']:.2f}")
                

                # Options chain visualization
                if not calls_df.empty and not puts_df.empty:

                    
                    # Ensure strike is numeric
                    calls_df['strike'] = pd.to_numeric(calls_df['strike'], errors='coerce')
                    puts_df['strike'] = pd.to_numeric(puts_df['strike'], errors='coerce')

                    # Drop any rows where strike couldn't be converted
                    calls_df = calls_df.dropna(subset=['strike'])
                    puts_df = puts_df.dropna(subset=['strike'])

                    # Get ATM price
                    underlying_price = merged_df[merged_df['Contract'] == selected_contract]['Last Price'].iloc[0]

                    # Filter strikes within ±20 of ATM
                    lower_bound = underlying_price - 20
                    upper_bound = underlying_price + 20

                    calls_near_atm = calls_df[(calls_df['strike'] >= lower_bound) & (calls_df['strike'] <= upper_bound)]
                    puts_near_atm = puts_df[(puts_df['strike'] >= lower_bound) & (puts_df['strike'] <= upper_bound)]


                    # Enhanced options chain chart
                    fig_options = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Calls Volume', 'Puts Volume'),
                        shared_yaxes=True
                    )

                    # Add volume bars for calls
                    fig_options.add_trace(
                        go.Bar(x=calls_near_atm['volume'], y=calls_near_atm['strike'],
                            orientation='h', name='Calls', 
                            marker_color='#10b981', opacity=0.7),
                        row=1, col=1
                    )

                    # Add volume bars for puts (negative for visual separation)
                    fig_options.add_trace(
                        go.Bar(x=-puts_near_atm['volume'], y=puts_near_atm['strike'],
                            orientation='h', name='Puts',
                            marker_color='#ef4444', opacity=0.7),
                        row=1, col=2
                    )

                    # Add ATM line
                    fig_options.add_hline(y=underlying_price, line_dash="dash", 
                                        line_color="white", annotation_text="ATM")

                    fig_options.update_layout(
                        title=f"Options Flow Analysis - {selected_contract}",
                        height=600,
                        template="plotly_dark",
                        showlegend=True
                    )

                    st.plotly_chart(fig_options, use_container_width=True)
                    
                    # Greeks analysis if advanced mode
                    if advanced_mode:
                        st.markdown("#### 🔬 Option Ladder")

                        # Ensure strike is numeric
                        calls_df['strike'] = pd.to_numeric(calls_df['strike'], errors='coerce')
                        puts_df['strike'] = pd.to_numeric(puts_df['strike'], errors='coerce')
                        calls_df = calls_df.dropna(subset=['strike'])
                        puts_df = puts_df.dropna(subset=['strike'])

                        # Calculate approximate Greeks
                        calls_with_greeks = calculate_greeks_approximation(calls_df, underlying_price)
                        puts_with_greeks = calculate_greeks_approximation(puts_df, underlying_price)

                        # Filter strikes within ±20 of ATM
                        lower_bound = underlying_price - 20
                        upper_bound = underlying_price + 20

                        calls_near_atm = calls_with_greeks[(calls_with_greeks['strike'] >= lower_bound) &
                                                        (calls_with_greeks['strike'] <= upper_bound)]
                        puts_near_atm = puts_with_greeks[(puts_with_greeks['strike'] >= lower_bound) &
                                                        (puts_with_greeks['strike'] <= upper_bound)]

                        # Function to highlight the row nearest ATM
                        def highlight_atm(df, atm_price):
                            if df.empty:
                                return pd.DataFrame("", index=df.index, columns=df.columns)
                            nearest_idx = (df['Strike'] - atm_price).abs().idxmin()
                            return pd.DataFrame(
                                [["background-color: rgba(255, 255, 0, 0.3)" if i == nearest_idx else "" for _ in df.columns] 
                                for i in df.index],
                                index=df.index, columns=df.columns
                            )

                        col1, col2 = st.columns(2)

                        with col1:
                            if not calls_near_atm.empty:
                                st.markdown("##### 📈 Calls Greeks")
                                greeks_display = calls_near_atm[['strike', 'bidPrice', 'delta_approx', 'gamma_approx']].copy()
                                greeks_display.columns = ['Strike', 'Price', 'Delta*', 'Gamma*']
                                st.dataframe(
                                    greeks_display.style.format({
                                        'Price': '${:.2f}',
                                        'Delta*': '{:.3f}',
                                        'Gamma*': '{:.3f}'
                                    }).apply(highlight_atm, atm_price=underlying_price, axis=None)
                                )

                        with col2:
                            if not puts_near_atm.empty:
                                st.markdown("##### 📉 Puts Greeks")
                                greeks_display = puts_near_atm[['strike', 'bidPrice', 'delta_approx', 'gamma_approx']].copy()
                                greeks_display.columns = ['Strike', 'Price', 'Delta*', 'Gamma*']
                                st.dataframe(
                                    greeks_display.style.format({
                                        'Price': '${:.2f}',
                                        'Delta*': '{:.3f}',
                                        'Gamma*': '{:.3f}'
                                    }).apply(highlight_atm, atm_price=underlying_price, axis=None)
                                )

                        st.caption("*Approximate values")

                        # Options chain with enhanced visualization
                        if not calls_df.empty or not puts_df.empty:
                            st.markdown("#### 🔗 Enhanced Options Chain")
                            
                            # ✅ Remove options with zero premium
                            calls_df = calls_df[calls_df['bidPrice'] > 0]
                            puts_df = puts_df[puts_df['bidPrice'] > 0]
                            
                            # Create sophisticated options visualization
                            fig_chain = go.Figure()
                            
                            if not calls_df.empty:
                                fig_chain.add_trace(
                                    go.Scatter(
                                        x=calls_df['strike'], 
                                        y=calls_df['bidPrice'],
                                        mode='markers', 
                                        name='Calls',
                                        marker=dict(size=10,   # ✅ Fixed standard bubble size
                                                    color='#10b981', opacity=0.7),
                                        hovertemplate='<b>Call</b><br>Strike: $%{x}<br>Price: $%{y}<br>Volume: %{text}<extra></extra>',
                                        text=calls_df['volume']
                                    )
                                )
                            
                            if not puts_df.empty:
                                fig_chain.add_trace(
                                    go.Scatter(
                                        x=puts_df['strike'], 
                                        y=puts_df['bidPrice'],
                                        mode='markers', 
                                        name='Puts',
                                        marker=dict(size=10,   # ✅ Fixed standard bubble size
                                                    color='#ef4444', opacity=0.7),
                                        hovertemplate='<b>Put</b><br>Strike: $%{x}<br>Price: $%{y}<br>Volume: %{text}<extra></extra>',
                                        text=puts_df['volume']
                                    )
                                )
                            
                            # Add ATM line
                            fig_chain.add_vline(
                                x=underlying, line_dash="dash", 
                                line_color="white", annotation_text="ATM"
                            )
                            
                            # Cap y-axis to max premium = 5
                            fig_chain.update_layout(
                                title=f"Options Chain - {selected_contract} ",
                                xaxis_title="Strike Price ($)",
                                yaxis_title="Option Price ($)",
                                yaxis=dict(range=[0, 5]),
                                template="plotly_dark",
                                height=600
                            )
                            
                            st.plotly_chart(fig_chain, use_container_width=True, key='curve')



    # TAB 3: Price Discovery & Historical Analysis
    with tab3:
        st.markdown("### 📈 Price Discovery & Historical Analysis")
        
        # Enhanced date controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lookback_period = st.selectbox(
                "📅 Analysis Period",
                ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
                index=2
            )
        
        with col2:
            chart_type = st.selectbox(
                "📊 Chart Type",
                ["Price", "Returns", "Volatility", "Volume"],
                index=0
            )
        
        with col3:
            overlay_futures = st.checkbox("🔗 Overlay Futures Curve", value=True)
        
        # Calculate date range
        end_date = datetime.datetime.now()
        period_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}
        start_date = end_date - datetime.timedelta(days=period_map[lookback_period])
        
        if st.button("📊 Load Historical Analysis", type="primary"):
            with st.spinner("📈 Loading historical data..."):
                df_wti = load_oil_data('CL=F', 
                                     start_date=start_date.strftime('%Y-%m-%d'), 
                                     end_date=end_date.strftime('%Y-%m-%d'))
            
            if not df_wti.empty and 'Close' in df_wti.columns:
                st.success(f"✅ Loaded {len(df_wti)} days of historical data")
                
                # Enhanced metrics calculation
                latest_price = df_wti['Close'].dropna().iloc[-1]
                price_change = (df_wti['Close'].dropna().iloc[-1] - df_wti['Close'].dropna().iloc[0])
                percent_change = price_change / df_wti['Close'].dropna().iloc[0] * 100
                
                # Volatility calculations
                returns = df_wti['Close'].pct_change().dropna()
                volatility_ann = returns.std() * np.sqrt(252) * 100
                
                # Risk metrics
                var_95 = np.percentile(returns, 5) * latest_price
                max_drawdown = ((df_wti['Close'] / df_wti['Close'].expanding().max()) - 1).min() * 100
                
                # Display enhanced metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("💰 Current Price", f"${latest_price:.2f}", 
                             delta=f"{percent_change:+.1f}%")
                
                with col2:
                    st.metric("📊 Volatility (Ann.)", f"{volatility_ann:.1f}%")
                
                with col3:
                    st.metric("⚠️ VaR (95%)", f"${var_95:.2f}")
                
                with col4:
                    st.metric("📉 Max Drawdown", f"{max_drawdown:.1f}%")
                
                with col5:
                    avg_volume = df_wti['Volume'].mean() if 'Volume' in df_wti.columns else 0
                    st.metric("📈 Avg Volume", f"{avg_volume:,.0f}")
                
                # Create comprehensive chart
                fig_historical = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Price Action', 'Daily Returns', 'Rolling Volatility'),
                    vertical_spacing=0.08,
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Price chart with technical indicators
                fig_historical.add_trace(
                    go.Scatter(x=df_wti['Date'], y=df_wti['Close'],
                              mode='lines', name='WTI Price',
                              line=dict(color='#615fff', width=2)),
                    row=1, col=1
                )
                
                # Add moving averages
                if len(df_wti) > 20:
                    df_wti['MA20'] = df_wti['Close'].rolling(20).mean()
                    fig_historical.add_trace(
                        go.Scatter(x=df_wti['Date'], y=df_wti['MA20'],
                                  mode='lines', name='MA20',
                                  line=dict(color='orange', width=1, dash='dash')),
                        row=1, col=1
                    )
                
                if len(df_wti) > 50:
                    df_wti['MA50'] = df_wti['Close'].rolling(50).mean()
                    fig_historical.add_trace(
                        go.Scatter(x=df_wti['Date'], y=df_wti['MA50'],
                                  mode='lines', name='MA50',
                                  line=dict(color='red', width=1, dash='dot')),
                        row=1, col=1
                    )
                
                # Returns
                fig_historical.add_trace(
                    go.Bar(x=df_wti['Date'][1:], y=returns * 100,
                          name='Daily Returns (%)',
                          marker_color=np.where(returns > 0, '#10b981', '#ef4444')),
                    row=2, col=1
                )
                
                # Rolling volatility
                rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
                fig_historical.add_trace(
                    go.Scatter(x=df_wti['Date'][21:], y=rolling_vol[21:],
                              mode='lines', name='21-Day Volatility',
                              line=dict(color='#f59e0b', width=2)),
                    row=3, col=1
                )
                
                fig_historical.update_layout(
                    height=800,
                    title=f"WTI Crude Oil - {lookback_period} Analysis",
                    template="plotly_dark",
                    showlegend=True
                )
                
                st.plotly_chart(fig_historical, use_container_width=True)
                
                # Statistical summary
                if advanced_mode:
                    st.markdown("#### 📊 Statistical Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        stats_df = pd.DataFrame({
                            'Metric': ['Mean Return (Daily)', 'Std Dev (Daily)', 'Skewness', 'Kurtosis', 'Sharpe Ratio*'],
                            'Value': [
                                f"{returns.mean() * 100:.3f}%",
                                f"{returns.std() * 100:.3f}%",
                                f"{returns.skew():.3f}",
                                f"{returns.kurtosis():.3f}",
                                f"{returns.mean() / returns.std() * np.sqrt(252):.3f}"
                            ]
                        })
                        st.dataframe(stats_df, hide_index=True)
                    
                    with col2:
                        # Distribution plot
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(x=returns * 100, nbinsx=50,
                                                       name='Returns Distribution',
                                                       marker_color='#615fff', opacity=0.7))
                        fig_dist.update_layout(
                            title="Returns Distribution",
                            xaxis_title="Daily Returns (%)",
                            yaxis_title="Frequency",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)


    # TAB 4: Contract Deep Dive
    with tab4:
        st.markdown("### 🔍 Individual Contract Deep Dive")
        
        # Enhanced contract selector with search
        selected_contract = st.selectbox(
            "🎯 Select Contract for Detailed Analysis:",
            options=merged_df['Contract'].tolist() if not merged_df.empty else [],
            index=0,
            help="Choose a contract for comprehensive analysis"
        )
        
        if selected_contract and f'options_data_{selected_contract}' in st.session_state:
            calls_df, puts_df = st.session_state[f'options_data_{selected_contract}']
            
            # Contract overview
            contract_data = merged_df[merged_df['Contract'] == selected_contract].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Last Price", f"${contract_data['Last Price']:.2f}")
                st.metric("📈 Price Change", f"{contract_data.get('priceChange', 0):+.2f}")
            
            with col2:
                st.metric("📊 Volume", f"{contract_data.get('volume', 0):,}")
                st.metric("🔢 Open Interest", f"{contract_data.get('openInterest', 0):,}")
            
            with col3:
                days_to_expiry = contract_data.get('Options Days to Expiry', 0)
                st.metric("⏰ Days to Expiry", f"{days_to_expiry}")
                
                iv = contract_data.get('Futures Implied Volatility', 0)
                st.metric("📊 Implied Vol", f"{iv:.1f}%")
            
            with col4:
                # Calculate moneyness for ATM options
                underlying = contract_data['Last Price']
                if not calls_df.empty:
                    calls_df['strike_num'] = pd.to_numeric(calls_df['strike'], errors='coerce')
                    atm_call = calls_df.loc[(calls_df['strike_num'] - underlying).abs().idxmin()]
                    st.metric("💰 ATM Call", f"${atm_call['bidPrice']:.2f}")
                
                if not puts_df.empty:
                    puts_df['strike_num'] = pd.to_numeric(puts_df['strike'], errors='coerce')
                    atm_put = puts_df.loc[(puts_df['strike_num'] - underlying).abs().idxmin()]
                    st.metric("💰 ATM Put", f"${atm_put['bidPrice']:.2f}")
            
            # Options chain with enhanced visualization
            if not calls_df.empty or not puts_df.empty:
                st.markdown("#### 🔗 Enhanced Options Chain")
                
                # Create sophisticated options visualization
                fig_chain = go.Figure()
                
                if not calls_df.empty:
                    fig_chain.add_trace(
                        go.Scatter(x=calls_df['strike'], y=calls_df['bidPrice'],
                                  mode='markers', name='Calls',
                                  marker=dict(size=calls_df['volume']/10, 
                                            color='#10b981', opacity=0.7,
                                            sizemode='area', sizeref=2),
                                  hovertemplate='<b>Call</b><br>Strike: $%{x}<br>Price: $%{y}<br>Volume: %{marker.size}<extra></extra>')
                    )
                
                if not puts_df.empty:
                    fig_chain.add_trace(
                        go.Scatter(x=puts_df['strike'], y=puts_df['bidPrice'],
                                  mode='markers', name='Puts',
                                  marker=dict(size=puts_df['volume']/10,
                                            color='#ef4444', opacity=0.7,
                                            sizemode='area', sizeref=2),
                                  hovertemplate='<b>Put</b><br>Strike: $%{x}<br>Price: $%{y}<br>Volume: %{marker.size}<extra></extra>')
                    )
                
                # Add ATM line
                fig_chain.add_vline(x=underlying, line_dash="dash", 
                                   line_color="white", annotation_text="ATM")
                
                fig_chain.update_layout(
                    title=f"Options Chain - {selected_contract} (Bubble size = Volume)",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Option Price ($)",
                    template="plotly_dark",
                    height=600
                )
                
                st.plotly_chart(fig_chain, use_container_width=True)
                
                # Detailed options tables
                col1, col2 = st.columns(2)
                
                with col1:
                    if not calls_df.empty:
                        st.markdown("##### 📞 Calls Detail")
                        calls_display = calls_df[['strike', 'lastPrice', 'bidPrice', 'askPrice', 'volume', 'openInterest']].copy()
                        calls_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI']
                        st.dataframe(calls_display.head(15), use_container_width=True)
                
                with col2:
                    if not puts_df.empty:
                        st.markdown("##### 📉 Puts Detail")
                        puts_display = puts_df[['strike', 'lastPrice', 'bidPrice', 'askPrice', 'volume', 'openInterest']].copy()
                        puts_display.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI']
                        st.dataframe(puts_display.head(15), use_container_width=True)

    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Footer with enhanced information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Data Sources:**")
    st.markdown("- Barchart.com (Futures & Options)")
    st.markdown("- Yahoo Finance (Historical)")

with col2:
    st.markdown("**🔄 Last Updated:**")
    st.markdown(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.markdown("**⚙️ System Status:**")
    if advanced_mode:
        st.markdown("🟢 Advanced Analytics: ON")
    else:
        st.markdown("🟡 Basic Mode: ON")

if __name__ == "__main__":
    main()