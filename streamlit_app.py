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
from pymongo import MongoClient

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
                        df['lastPrice'] = df['lastPrice'].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
                        df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce').fillna(0)
                        
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
    
    # Display basic info - front month only
    col1, col2 = st.columns(2)
    
    with col1:
        front_month_contract = df['symbol'].iloc[0] if not df.empty else "N/A"
        st.metric("Front Month Contract", front_month_contract)
    
    with col2:
        front_month_price = df['lastPrice'].iloc[0] if not df.empty else 0
        st.metric("Front Month Price", f"${front_month_price:.2f}")

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
    
    # === Fetch first 10 contracts automatically at startup ===
    if "initial_options_data" not in st.session_state:
        if not merged_df.empty:
            initial_contracts = merged_df['Contract'].head(10).tolist()

            with st.spinner("Fetching options data for the first 10 contracts..."):
                options_data = fetch_detailed_options_data(initial_contracts)

            # Store all contracts data in session_state
            for contract in initial_contracts:
                if options_data and contract in options_data:
                    st.session_state[f'options_data_{contract}'] = options_data[contract]

            # Auto-select the first contract
            if initial_contracts:
                st.session_state["selected_contract"] = initial_contracts[0]

            st.session_state["initial_options_data"] = True
            st.success("Fetched options data for the first 10 contracts ✅")
        else:
            st.warning("No contracts available to fetch initially.")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Options Curve", 
    "Expiry Distribution", 
    "Contract Comparison", 
    "2D Visualization", 
    "Trade Ledger"
])

    
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
                latest_date = df_wti['Date'].iloc[-1].strftime('%Y-%m-%d') if not df_wti.empty else "N/A"
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
                    st.metric("Last Closing Price", f"${latest_price:.2f}", delta=f"{price_change:+.2f}")
                    st.caption(f"Date: {latest_date}")
                
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
                
                # Create all four charts
                st.subheader("WTI Price Analysis")
                
                # Prepare data for different periods
                current_year = datetime.datetime.now().year
                ytd_start = datetime.datetime(current_year, 1, 1)
                
                # 1. 3 Months chart
                chart_3m = df_wti.tail(min(63, len(df_wti)))
                
                # 2. YTD chart  
                chart_ytd = df_wti[df_wti['Date'] >= ytd_start]
                
                # 3. Full period chart
                chart_full = df_wti
                
                # 4. Yearly rebased chart data preparation
                df_rebased = df_wti.copy()
                df_rebased['Year'] = df_rebased['Date'].dt.year
                df_rebased['DayOfYear'] = df_rebased['Date'].dt.dayofyear
                df_rebased['Quarter'] = df_rebased['Date'].dt.quarter
                
                # Rebase function (per year)
                def rebase(series):
                    if len(series) > 0:
                        return (series / series.iloc[0]) * 100
                    return series
                
                df_rebased['Rebased'] = df_rebased.groupby('Year')['Close'].transform(rebase)
                
                # Create charts in 2x2 grid
                col1, col2 = st.columns(2)
                
                with col1:
                    # 3 Months chart
                    fig_3m = px.line(chart_3m,
                                    x='Date',
                                    y='Close',
                                    title='WTI - Last 3 Months',
                                    labels={'Close': 'Price ($)', 'Date': 'Date'})
                    fig_3m.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_3m, use_container_width=True)
                
                with col2:
                    # YTD chart
                    fig_ytd = px.line(chart_ytd,
                                     x='Date',
                                     y='Close',
                                     title=f'WTI - YTD {current_year}',
                                     labels={'Close': 'Price ($)', 'Date': 'Date'})
                    fig_ytd.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_ytd, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Full period chart
                    fig_full = px.line(chart_full,
                                      x='Date',
                                      y='Close',
                                      title='WTI - Full Period',
                                      labels={'Close': 'Price ($)', 'Date': 'Date'})
                    fig_full.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_full, use_container_width=True)
                
                with col4:
                    # Yearly rebased chart
                    fig_rebased = go.Figure()
                    
                    # Add quarterly shaded regions
                    quarter_colors = ['#e0f3f8', '#ccebc5', '#fddbc7', '#f2f0f7']
                    quarter_ranges = {
                        1: (1, 90),
                        2: (91, 181), 
                        3: (182, 273),
                        4: (274, 366)
                    }
                    
                    for q, (start, end) in quarter_ranges.items():
                        fig_rebased.add_vrect(
                            x0=start, x1=end,
                            fillcolor=quarter_colors[q-1],
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                            annotation_text=f"Q{q}",
                            annotation_position="top"
                        )
                    
                    # Choose years to highlight (last 3 years)
                    all_years = sorted(df_rebased['Year'].unique())
                    highlight_years = all_years[-3:] if len(all_years) >= 3 else all_years
                    highlight_colors = ['crimson', 'royalblue', 'darkorange']
                    muted_color = 'lightgray'
                    
                    # Plot yearly rebased lines
                    for year, group in df_rebased.groupby('Year'):
                        if year in highlight_years and len(highlight_years) <= 3:
                            idx = highlight_years.index(year) % len(highlight_colors)
                            color = highlight_colors[idx]
                            line_width = 3
                        else:
                            color = muted_color
                            line_width = 1.5
                        
                        fig_rebased.add_trace(
                            go.Scatter(
                                x=group['DayOfYear'],
                                y=group['Rebased'],
                                mode='lines',
                                name=str(year),
                                line=dict(color=color, width=line_width)
                            )
                        )
                    
                    fig_rebased.update_layout(
                        height=400,
                        title="WTI - Yearly Rebased (Q Highlights)",
                        xaxis_title="Day of Year",
                        yaxis_title="Rebased Price (first day = 100)",
                        template="plotly_white",
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left", 
                            x=0.01,
                            font=dict(size=10)
                        )
                    )
                    
                    st.plotly_chart(fig_rebased, use_container_width=True)

    with tab3:
        st.subheader("Options Prices")

        # Dropdown with auto-selected first contract
        selected_contract = st.selectbox(
            "Select contract for options data:",
            options=merged_df['Contract'].tolist() if not merged_df.empty else [],
            index=0 if not merged_df.empty else None,
            key="options_contract_selector"
        )

        if f'options_data_{selected_contract}' in st.session_state:
            calls_df, puts_df = st.session_state[f'options_data_{selected_contract}']

            if not calls_df.empty and not puts_df.empty:
                # Display Calls and Puts side by side
                col1, col2 = st.columns(2)

                columns_to_show = ['strike', 'lastPrice', 'priceChange', 'bidPrice', 'askPrice', 'volume', 'openInterest']

                with col1:
                    st.subheader(f"📈 Calls - {selected_contract}")
                    display_calls = calls_df.copy()
                    if all(col in display_calls.columns for col in columns_to_show):
                        display_calls = display_calls[columns_to_show]
                        display_calls.columns = ['Strike', 'Last Price', 'Change', 'Bid', 'Ask', 'Volume', 'Open Interest']

                        for col in ['Last Price', 'Change', 'Bid', 'Ask']:
                            display_calls[col] = display_calls[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        for col in ['Volume', 'Open Interest']:
                            display_calls[col] = display_calls[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")

                        st.dataframe(display_calls, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(calls_df, use_container_width=True)

                with col2:
                    st.subheader(f"📉 Puts - {selected_contract}")
                    display_puts = puts_df.copy()
                    if all(col in display_puts.columns for col in columns_to_show):
                        display_puts = display_puts[columns_to_show]
                        display_puts.columns = ['Strike', 'Last Price', 'Change', 'Bid', 'Ask', 'Volume', 'Open Interest']

                        for col in ['Last Price', 'Change', 'Bid', 'Ask']:
                            display_puts[col] = display_puts[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        for col in ['Volume', 'Open Interest']:
                            display_puts[col] = display_puts[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "0")

                        st.dataframe(display_puts, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(puts_df, use_container_width=True)

                # Summary stats
                st.subheader("Options Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_call_volume = calls_df['volume'].sum() if 'volume' in calls_df.columns else 0
                    st.metric("Total Call Volume", f"{total_call_volume:,}")

                with col2:
                    total_put_volume = puts_df['volume'].sum() if 'volume' in puts_df.columns else 0
                    st.metric("Total Put Volume", f"{total_put_volume:,}")

                with col3:
                    call_put_ratio = total_call_volume / total_put_volume if total_put_volume > 0 else 0
                    st.metric("Call/Put Ratio", f"{call_put_ratio:.2f}")

                with col4:
                    total_options = len(calls_df) + len(puts_df)
                    st.metric("Total Options", f"{total_options}")
            else:
                st.warning(f"No options data available for {selected_contract}")
        else:
            st.warning("Please wait for the initial contracts data to load.")

    # === TAB 4: 2D Overlay Plot (Ask Price + ATM Annotation) ===
    with tab4:
        st.subheader("2D Options Visualization (Overlay)")

        # Collect all cached contracts (first 10 pulled earlier)
        cached_contracts = [c for c in merged_df['Contract'].head(10).tolist() 
                            if f'options_data_{c}' in st.session_state]

        if cached_contracts:
            # Let user pick one or multiple contracts (default = first one)
            default_contract = [cached_contracts[0]]
            selected_contracts = st.multiselect(
                "Select contract(s) to display:",
                options=cached_contracts,
                default=default_contract
            )

            all_data = []
            for contract in selected_contracts:
                calls_df, puts_df = st.session_state[f'options_data_{contract}']
                if not calls_df.empty:
                    all_data.append(calls_df.assign(optionType="Call", Contract=contract))
                if not puts_df.empty:
                    all_data.append(puts_df.assign(optionType="Put", Contract=contract))

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)

                # Filter toggle
                option_filter = st.radio(
                    "Select Option Type:",
                    options=["Both", "Calls Only", "Puts Only"],
                    horizontal=True
                )

                if option_filter == "Calls Only":
                    filtered_df = combined_df[combined_df["optionType"] == "Call"]
                elif option_filter == "Puts Only":
                    filtered_df = combined_df[combined_df["optionType"] == "Put"]
                else:
                    filtered_df = combined_df

                # Ensure numeric
                filtered_df["strike"] = pd.to_numeric(filtered_df["strike"], errors="coerce")
                filtered_df["askPrice"] = pd.to_numeric(filtered_df["askPrice"], errors="coerce")

                if not filtered_df.empty and all(col in filtered_df.columns for col in ['strike', 'askPrice', 'Contract']):
                    st.info("Overlay of Ask Price vs. Strike Price (with ATM line & annotation)")

                    # Get ATM mapping from futures last price
                    atm_map = dict(zip(merged_df['Contract'], merged_df['Last Price']))

                    # Build overlay scatter
                    fig = px.scatter(
                        filtered_df,
                        x="strike",
                        y="askPrice",
                        color="Contract",   # different color per contract
                        symbol="optionType",  # Call vs Put symbols
                        hover_data=["optionType", "bidPrice", "lastPrice", "volume", "openInterest"],
                        labels={"strike": "Strike Price", "askPrice": "Ask Price"}
                    )

                    fig.update_traces(marker=dict(size=7, opacity=0.7))

                    # Add ATM lines + annotations
                    y_max = filtered_df["askPrice"].max()
                    for contract in selected_contracts:
                        if contract in atm_map:
                            atm_strike = atm_map[contract]

                            # Add vertical ATM line
                            fig.add_shape(
                                type="line",
                                x0=atm_strike, x1=atm_strike,
                                y0=0, y1=y_max,
                                line=dict(color="red", dash="dash"),
                                xref="x", yref="y"
                            )

                            # Annotate ATM
                            fig.add_annotation(
                                x=atm_strike,
                                y=y_max * 0.95,  # place near top
                                text=f"ATM {atm_strike:.2f} ({contract})",
                                showarrow=False,
                                font=dict(color="red"),
                                bgcolor="white",
                                opacity=0.8
                            )

                    fig.update_layout(
                        title="Options Across Selected Contracts (Ask Price, ATM Annotated)",
                        template="plotly_white",
                        legend_title="Contract",
                        height=700
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data available for 2D visualization.")
            else:
                st.warning("No options data available for visualization.")
        else:
            st.warning("Please wait for the initial contracts data to load.")

    # === TAB 5: Enhanced Trade Ledger ===
    with tab5:
        st.subheader("📊 Enhanced Trade Ledger")

        # --- MongoDB Connection via Streamlit Secrets ---
        try:
            MONGO_URI = st.secrets["mongodb"]["uri"]
            client = MongoClient(MONGO_URI)
            db = client["OptionTrades_db"]
            collection = db["OptionTrades"]
            mongo_connected = True
        except Exception as e:
            st.error(f"MongoDB connection failed: {str(e)}")
            st.warning("Trade ledger will work in session-only mode.")
            mongo_connected = False
            # Initialize session state for trades if MongoDB is not available
            if 'trades_data' not in st.session_state:
                st.session_state.trades_data = []

        # --- Data Import/Mapping Functions ---
        def map_csv_to_trade_format(csv_df):
            """Map CSV columns to the expected trade format"""
            mapped_trades = []
            
            for _, row in csv_df.iterrows():
                # Determine status based on whether it's expired or has a close date
                status = "Closed" if (row.get('Expired?') == 'Expired' or 
                                    pd.notna(row.get('Close Date')) or 
                                    pd.notna(row.get('Closing Price'))) else "Open"
                
                # Parse dates carefully
                trade_date = None
                expiry_date = None
                close_date = None
                
                try:
                    if pd.notna(row.get('Trade Date')):
                        trade_date = pd.to_datetime(row['Trade Date'], format='%d/%m/%Y').date().isoformat()
                except:
                    try:
                        trade_date = pd.to_datetime(row['Trade Date']).date().isoformat()
                    except:
                        trade_date = datetime.date.today().isoformat()
                
                try:
                    if pd.notna(row.get('Expiration Date')):
                        expiry_date = pd.to_datetime(row['Expiration Date'], format='%d/%m/%Y').date().isoformat()
                except:
                    try:
                        expiry_date = pd.to_datetime(row['Expiration Date']).date().isoformat()
                    except:
                        expiry_date = None
                
                try:
                    if pd.notna(row.get('Close Date')):
                        close_date = pd.to_datetime(row['Close Date'], format='%d/%m/%Y').date().isoformat()
                except:
                    try:
                        close_date = pd.to_datetime(row['Close Date']).date().isoformat()
                    except:
                        close_date = None
                
                mapped_trade = {
                    "trade_no": int(row.get('Trade No', 0)),
                    "trade_date": trade_date,
                    "underlying": str(row.get('Underlying', 'N/A')),
                    "expiration_date": expiry_date,
                    "days_to_expiry": int(row.get('Days to Expiry', 0)) if pd.notna(row.get('Days to Expiry')) and str(row.get('Days to Expiry')).replace('-','').isdigit() else 0,
                    "option_type": str(row.get('Option Type', 'Call')),
                    "direction": str(row.get('Direction', 'Buy')),
                    "strike_price": float(row.get('Strike Price', 0)) if pd.notna(row.get('Strike Price')) else 0,
                    "contracts": 1,  # Default to 1 contract
                    "entry_price": float(row.get('Entry Price', 0)) if pd.notna(row.get('Entry Price')) else 0,
                    "commission": float(row.get('Commision', 0)) if pd.notna(row.get('Commision')) else 0,  # Note: 'Commision' spelling
                    "status": status,
                    "entry_timestamp": datetime.datetime.now().isoformat(),
                    "current_price": float(row.get('Closing Price', 0)) if pd.notna(row.get('Closing Price')) and status == "Closed" else None,
                    "unrealized_pnl": 0.0,
                    "final_pnl": float(row.get('Final P&L', 0)) if pd.notna(row.get('Final P&L')) else 0,
                    "close_date": close_date,
                    "close_price": float(row.get('Closing Price', 0)) if pd.notna(row.get('Closing Price')) else None,
                    "settlement_price": float(row.get('Settlement Price', 0)) if pd.notna(row.get('Settlement Price')) else None,
                    "notes": f"Imported from CSV. Tenure: {row.get('Tenure', 'N/A')}"
                }
                
                mapped_trades.append(mapped_trade)
            
            return mapped_trades

        def import_csv_trades(csv_data):
            """Import trades from CSV data"""
            try:
                # Parse CSV
                from io import StringIO
                csv_df = pd.read_csv(StringIO(csv_data))
                
                # Map to trade format
                mapped_trades = map_csv_to_trade_format(csv_df)
                
                # Add to database/session
                for trade in mapped_trades:
                    add_trade(trade)
                
                return len(mapped_trades)
            except Exception as e:
                st.error(f"Error importing CSV: {str(e)}")
                return 0
        def get_trades():
            if mongo_connected:
                trades = list(collection.find({}, {"_id": 0}))
                return pd.DataFrame(trades)
            else:
                return pd.DataFrame(st.session_state.trades_data)

        def add_trade(trade):
            if mongo_connected:
                collection.insert_one(trade)
            else:
                st.session_state.trades_data.append(trade)

        def update_trade(trade_no, updates):
            if mongo_connected:
                collection.update_one({"trade_no": trade_no}, {"$set": updates})
            else:
                for i, trade in enumerate(st.session_state.trades_data):
                    if trade.get("trade_no") == trade_no:
                        st.session_state.trades_data[i].update(updates)
                        break

        def delete_trade(trade_no):
            if mongo_connected:
                collection.delete_one({"trade_no": trade_no})
            else:
                st.session_state.trades_data = [t for t in st.session_state.trades_data 
                                            if t.get("trade_no") != trade_no]

        def get_next_trade_no():
            df = get_trades()
            if df.empty or 'trade_no' not in df.columns:
                return 1
            return int(df['trade_no'].max()) + 1

        # --- Helper function to calculate P&L ---
        def calculate_pnl(entry_price, exit_price, direction, contracts=1):
            if direction == "Buy":
                return (exit_price - entry_price) * contracts * 1000  # 1000 barrels per contract
            else:  # Sell
                return (entry_price - exit_price) * contracts * 1000

        def calculate_days_to_expiry(expiry_date):
            today = datetime.date.today()
            if isinstance(expiry_date, str):
                expiry_date = datetime.datetime.fromisoformat(expiry_date).date()
            return (expiry_date - today).days

        # --- Main Trade Ledger Interface ---
        
        # Create tabs within the trade ledger
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
            "📋 All Trades", 
            "➕ New Trade", 
            "🔄 Manage Trades", 
            "📊 Analytics",
            "📁 Import/Export"
        ])
        
        with subtab1:
            st.subheader("All Trades")
            
            # Get and display trades
            trades_df = get_trades()
            
            if not trades_df.empty:
                # Calculate additional metrics
                trades_display = trades_df.copy()
                
                # Calculate current days to expiry for open trades
                if 'expiration_date' in trades_display.columns and 'status' in trades_display.columns:
                    trades_display['current_days_to_expiry'] = trades_display.apply(
                        lambda row: calculate_days_to_expiry(row['expiration_date']) 
                        if row.get('status') == 'Open' else None, axis=1
                    )
                
                # Format columns for better display
                if 'entry_price' in trades_display.columns:
                    trades_display['entry_price_formatted'] = trades_display['entry_price'].apply(
                        lambda x: f"${x:.3f}" if pd.notna(x) else "N/A"
                    )
                
                if 'current_price' in trades_display.columns:
                    trades_display['current_price_formatted'] = trades_display['current_price'].apply(
                        lambda x: f"${x:.3f}" if pd.notna(x) else "N/A"
                    )
                
                if 'unrealized_pnl' in trades_display.columns:
                    trades_display['unrealized_pnl_formatted'] = trades_display['unrealized_pnl'].apply(
                        lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
                    )
                
                if 'final_pnl' in trades_display.columns:
                    trades_display['final_pnl_formatted'] = trades_display['final_pnl'].apply(
                        lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
                    )
                
                # Display columns selection
                display_cols = ['trade_no', 'trade_date', 'underlying', 'option_type', 'direction', 
                            'strike_price', 'entry_price_formatted', 'status']
                
                if 'current_price_formatted' in trades_display.columns:
                    display_cols.append('current_price_formatted')
                
                if 'current_days_to_expiry' in trades_display.columns:
                    display_cols.append('current_days_to_expiry')
                    
                if 'unrealized_pnl_formatted' in trades_display.columns:
                    display_cols.append('unrealized_pnl_formatted')
                    
                if 'final_pnl_formatted' in trades_display.columns:
                    display_cols.append('final_pnl_formatted')
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in trades_display.columns]
                
                st.dataframe(
                    trades_display[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_trades = len(trades_df)
                    st.metric("Total Trades", total_trades)
                
                with col2:
                    open_trades = len(trades_df[trades_df.get('status', '') == 'Open']) if 'status' in trades_df.columns else 0
                    st.metric("Open Trades", open_trades)
                
                with col3:
                    closed_trades = len(trades_df[trades_df.get('status', '') == 'Closed']) if 'status' in trades_df.columns else 0
                    st.metric("Closed Trades", closed_trades)
                
                with col4:
                    total_pnl = trades_df['final_pnl'].sum() if 'final_pnl' in trades_df.columns else 0
                    st.metric("Total Realized P&L", f"${total_pnl:,.2f}")
                    
            else:
                st.info("No trades found. Add your first trade in the 'New Trade' tab.")
        
        with subtab2:
            st.subheader("➕ Add New Trade")
            
            # Pre-populate with current market data
            available_contracts = merged_df['Contract'].tolist() if not merged_df.empty else []
            
            with st.form("add_trade_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    trade_no = st.number_input("Trade No", value=get_next_trade_no(), step=1)
                    trade_date = st.date_input("Trade Date", value=datetime.date.today())
                    
                    # Use available contracts from fetched data
                    if available_contracts:
                        underlying = st.selectbox("Contract", available_contracts)
                        
                        # Auto-populate current price if available
                        if underlying in merged_df['Contract'].values:
                            current_futures_price = merged_df[merged_df['Contract'] == underlying]['Last Price'].iloc[0]
                            st.info(f"Current futures price for {underlying}: ${current_futures_price:.2f}")
                    else:
                        underlying = st.text_input("Underlying")
                    
                    option_type = st.selectbox("Option Type", ["Call", "Put"])
                    direction = st.selectbox("Direction", ["Buy", "Sell"])
                    
                with col2:
                    # Smart strike price suggestions based on current futures price
                    if available_contracts and underlying in merged_df['Contract'].values:
                        current_price = merged_df[merged_df['Contract'] == underlying]['Last Price'].iloc[0]
                        suggested_strikes = [
                            current_price - 5, current_price - 2, current_price, 
                            current_price + 2, current_price + 5
                        ]
                        strike_options = [f"{s:.2f}" for s in suggested_strikes]
                        strike_selection = st.selectbox("Strike Price", strike_options, index=2)  # Default to ATM
                        strike_price = float(strike_selection)
                    else:
                        strike_price = st.number_input("Strike Price", step=0.01)
                    
                    expiration_date = st.date_input("Expiration Date")
                    contracts = st.number_input("Number of Contracts", value=1, min_value=1, step=1)
                    entry_price = st.number_input("Entry Price (per contract)", step=0.001, format="%.3f")
                    commission = st.number_input("Commission", value=0.0, step=0.01)
                
                # Calculate position value
                position_value = entry_price * contracts * 1000  # 1000 barrels per contract
                st.info(f"Position Value: ${position_value:,.2f}")
                
                submitted = st.form_submit_button("Add Trade", type="primary")
                
                if submitted:
                    # Validate inputs
                    if entry_price <= 0:
                        st.error("Entry price must be greater than 0")
                    elif expiration_date <= trade_date:
                        st.error("Expiration date must be after trade date")
                    else:
                        # Calculate days to expiry
                        days_to_expiry = (expiration_date - trade_date).days
                        
                        trade_data = {
                            "trade_no": int(trade_no),
                            "trade_date": trade_date.isoformat(),
                            "underlying": underlying,
                            "expiration_date": expiration_date.isoformat(),
                            "days_to_expiry": days_to_expiry,
                            "option_type": option_type,
                            "direction": direction,
                            "strike_price": float(strike_price),
                            "contracts": int(contracts),
                            "entry_price": float(entry_price),
                            "commission": float(commission),
                            "status": "Open",
                            "entry_timestamp": datetime.datetime.now().isoformat(),
                            "current_price": None,
                            "unrealized_pnl": 0.0,
                            "final_pnl": 0.0,
                            "close_date": None,
                            "close_price": None,
                            "notes": ""
                        }
                        
                        try:
                            add_trade(trade_data)
                            st.success(f"Trade #{trade_no} added successfully! ✅")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding trade: {str(e)}")
        
        with subtab3:
            st.subheader("🔄 Manage Open Trades")
            
            trades_df = get_trades()
            
            if not trades_df.empty:
                # Filter open trades
                open_trades_df = trades_df[trades_df.get('status', '') == 'Open'] if 'status' in trades_df.columns else trades_df
                
                if not open_trades_df.empty:
                    # Select trade to manage - with error handling for missing columns
                    trade_options = []
                    required_cols = ['trade_no', 'underlying', 'option_type', 'strike_price']
                    
                    for _, row in open_trades_df.iterrows():
                        try:
                            trade_no = row.get('trade_no', 'N/A')
                            underlying = row.get('underlying', 'N/A')
                            option_type = row.get('option_type', 'N/A')
                            strike_price = row.get('strike_price', 'N/A')
                            
                            trade_options.append(f"Trade #{trade_no} - {underlying} {option_type} {strike_price}")
                        except Exception as e:
                            trade_options.append(f"Trade #{row.name} - Error loading details")
                    
                    selected_trade_idx = st.selectbox("Select Trade to Manage", 
                                                    range(len(trade_options)), 
                                                    format_func=lambda x: trade_options[x])
                    
                    selected_trade = open_trades_df.iloc[selected_trade_idx]
                    trade_no = selected_trade.get('trade_no', 'N/A')
                    
                    # Display trade details with safe column access
                    st.write("**Trade Details:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Trade #:** {trade_no}")
                        st.write(f"**Underlying:** {selected_trade.get('underlying', 'N/A')}")
                        st.write(f"**Type:** {selected_trade.get('option_type', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Direction:** {selected_trade.get('direction', 'N/A')}")
                        st.write(f"**Strike:** ${selected_trade.get('strike_price', 0)}")
                        st.write(f"**Entry Price:** ${selected_trade.get('entry_price', 0):.3f}")
                    
                    with col3:
                        contracts = selected_trade.get('contracts', 1)
                        st.write(f"**Contracts:** {contracts}")
                        
                        expiry_date = selected_trade.get('expiration_date')
                        if expiry_date:
                            days_left = calculate_days_to_expiry(expiry_date)
                            st.write(f"**Days to Expiry:** {days_left}")
                            
                            # Color code based on days left
                            if days_left < 0:
                                st.error("⚠️ EXPIRED")
                            elif days_left < 7:
                                st.warning("🔥 Expires Soon")
                            else:
                                st.success("✅ Active")
                        else:
                            st.write("**Days to Expiry:** N/A")
                    
                    # Update current market price
                    st.subheader("Update Market Price")
                    
                    # Try to get current price from live data
                    current_market_price = None
                    underlying = selected_trade.get('underlying')
                    
                    if underlying and not merged_df.empty and underlying in merged_df['Contract'].values:
                        # Get current futures price (as reference)
                        current_futures_price = merged_df[merged_df['Contract'] == underlying]['Last Price'].iloc[0]
                        st.info(f"Current futures price: ${current_futures_price:.2f}")
                        
                        # Try to get current options price from cached data
                        contract_symbol = underlying
                        if f'options_data_{contract_symbol}' in st.session_state:
                            calls_df, puts_df = st.session_state[f'options_data_{contract_symbol}']
                            
                            # Find matching option
                            option_type = selected_trade.get('option_type', 'Call')
                            strike_price = selected_trade.get('strike_price', 0)
                            
                            target_df = calls_df if option_type == 'Call' else puts_df
                            if not target_df.empty and 'strike' in target_df.columns:
                                # Find closest strike
                                target_df['strike_num'] = pd.to_numeric(target_df['strike'], errors='coerce')
                                closest_strike_idx = (target_df['strike_num'] - strike_price).abs().idxmin()
                                
                                if pd.notna(closest_strike_idx):
                                    current_option = target_df.loc[closest_strike_idx]
                                    current_market_price = current_option.get('lastPrice', current_option.get('askPrice'))
                                    if pd.notna(current_market_price) and current_market_price > 0:
                                        st.success(f"Current option price: ${current_market_price:.3f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Manual price update
                        new_current_price = st.number_input(
                            "Current Option Price", 
                            value=float(current_market_price) if current_market_price else 0.0,
                            step=0.001, 
                            format="%.3f"
                        )
                        
                        if st.button("Update Price"):
                            # Calculate unrealized P&L
                            entry_price = selected_trade.get('entry_price', 0)
                            direction = selected_trade.get('direction', 'Buy')
                            
                            unrealized_pnl = calculate_pnl(
                                entry_price, 
                                new_current_price, 
                                direction, 
                                contracts
                            )
                            
                            update_trade(trade_no, {
                                "current_price": float(new_current_price),
                                "unrealized_pnl": float(unrealized_pnl),
                                "last_updated": datetime.datetime.now().isoformat()
                            })
                            
                            st.success("Price updated!")
                            time.sleep(1)
                            st.rerun()
                    
                    with col2:
                        # Close trade
                        st.subheader("Close Trade")
                        close_price = st.number_input("Close Price", step=0.001, format="%.3f")
                        close_commission = st.number_input("Close Commission", value=0.0, step=0.01)
                        
                        if st.button("Close Trade", type="primary"):
                            if close_price <= 0:
                                st.error("Close price must be greater than 0")
                            else:
                                # Calculate final P&L
                                final_pnl = calculate_pnl(
                                    selected_trade['entry_price'], 
                                    close_price, 
                                    selected_trade['direction'], 
                                    contracts
                                ) - selected_trade.get('commission', 0) - close_commission
                                
                                update_trade(trade_no, {
                                    "status": "Closed",
                                    "close_date": datetime.date.today().isoformat(),
                                    "close_price": float(close_price),
                                    "close_commission": float(close_commission),
                                    "final_pnl": float(final_pnl),
                                    "closed_timestamp": datetime.datetime.now().isoformat()
                                })
                                
                                st.success(f"Trade #{trade_no} closed! Final P&L: ${final_pnl:,.2f}")
                                time.sleep(2)
                                st.rerun()
                    
                    # Display current P&L if price is available
                    current_price = selected_trade.get('current_price')
                    entry_price = selected_trade.get('entry_price', 0)
                    direction = selected_trade.get('direction', 'Buy')
                    
                    if current_price and pd.notna(current_price):
                        current_pnl = calculate_pnl(
                            entry_price, 
                            current_price, 
                            direction, 
                            contracts
                        )
                        
                        pnl_color = "green" if current_pnl >= 0 else "red"
                        st.markdown(f"**Unrealized P&L:** <span style='color:{pnl_color}'>${current_pnl:,.2f}</span>", 
                                unsafe_allow_html=True)
                    
                else:
                    st.info("No open trades to manage.")
            else:
                st.info("No trades found.")
            
            # Bulk actions
            st.subheader("Bulk Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⚠️ Delete All Trades", help="This will delete ALL trades permanently"):
                    if mongo_connected:
                        collection.delete_many({})
                    else:
                        st.session_state.trades_data = []
                    st.warning("All trades deleted!")
                    st.rerun()
            
            with col2:
                # Export trades
                trades_df = get_trades()
                if not trades_df.empty:
                    csv_data = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Trades to CSV",
                        data=csv_data,
                        file_name=f"options_trades_{datetime.date.today()}.csv",
                        mime="text/csv"
                    )
        
        with subtab4:
            st.subheader("📊 Trade Analytics")
            
            trades_df = get_trades()
            
            if not trades_df.empty and len(trades_df) > 0:
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    win_rate = len(trades_df[trades_df.get('final_pnl', 0) > 0]) / len(trades_df) * 100 if 'final_pnl' in trades_df.columns else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col2:
                    avg_pnl = trades_df['final_pnl'].mean() if 'final_pnl' in trades_df.columns else 0
                    st.metric("Avg P&L", f"${avg_pnl:.2f}")
                
                with col3:
                    best_trade = trades_df['final_pnl'].max() if 'final_pnl' in trades_df.columns else 0
                    st.metric("Best Trade", f"${best_trade:.2f}")
                
                with col4:
                    worst_trade = trades_df['final_pnl'].min() if 'final_pnl' in trades_df.columns else 0
                    st.metric("Worst Trade", f"${worst_trade:.2f}")
                
                # Charts
                if 'final_pnl' in trades_df.columns and trades_df['final_pnl'].notna().any():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # P&L distribution
                        fig_pnl = px.histogram(
                            trades_df, 
                            x='final_pnl', 
                            title='P&L Distribution',
                            nbins=20
                        )
                        fig_pnl.update_layout(
                            xaxis_title="P&L ($)",
                            yaxis_title="Number of Trades"
                        )
                        st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    with col2:
                        # Cumulative P&L over time
                        if 'trade_date' in trades_df.columns:
                            pnl_over_time = trades_df.copy()
                            pnl_over_time['trade_date'] = pd.to_datetime(pnl_over_time['trade_date'])
                            pnl_over_time = pnl_over_time.sort_values('trade_date')
                            pnl_over_time['cumulative_pnl'] = pnl_over_time['final_pnl'].cumsum()
                            
                            fig_cum = px.line(
                                pnl_over_time,
                                x='trade_date',
                                y='cumulative_pnl',
                                title='Cumulative P&L Over Time'
                            )
                            fig_cum.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Cumulative P&L ($)"
                            )
                            st.plotly_chart(fig_cum, use_container_width=True)
                
                # Trade breakdown by type
                if 'option_type' in trades_df.columns and 'direction' in trades_df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # By option type
                        type_summary = trades_df.groupby('option_type')['final_pnl'].agg(['count', 'sum', 'mean']).reset_index()
                        type_summary.columns = ['Option Type', 'Count', 'Total P&L', 'Avg P&L']
                        st.subheader("By Option Type")
                        st.dataframe(type_summary, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # By direction
                        dir_summary = trades_df.groupby('direction')['final_pnl'].agg(['count', 'sum', 'mean']).reset_index()
                        dir_summary.columns = ['Direction', 'Count', 'Total P&L', 'Avg P&L']
                        st.subheader("By Direction")
                        st.dataframe(dir_summary, use_container_width=True, hide_index=True)
                
            else:
                st.info("No trades available for analysis. Add some trades to see analytics.")
        
        with subtab5:
            st.subheader("📁 Import/Export Data")
            
            # Import required for CSV processing
            from io import StringIO
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 Import Trades")
                st.info("Upload your existing CSV file to import trades into the system.")
                
                uploaded_file = st.file_uploader("Choose CSV file", type="csv")
                
                if uploaded_file is not None:
                    # Preview the uploaded data
                    try:
                        csv_content = uploaded_file.read().decode('utf-8')
                        preview_df = pd.read_csv(StringIO(csv_content))
                        
                        st.write("**Preview of uploaded data:**")
                        st.dataframe(preview_df.head(), use_container_width=True)
                        
                        st.write(f"**Total rows:** {len(preview_df)}")
                        st.write("**Columns found:**", list(preview_df.columns))
                        
                        # Show mapping
                        st.write("**Column Mapping:**")
                        mapping_info = {
                            "Trade No → trade_no": "✅",
                            "Trade Date → trade_date": "✅", 
                            "Underlying → underlying": "✅",
                            "Expiration Date → expiration_date": "✅",
                            "Option Type → option_type": "✅",
                            "Direction → direction": "✅",
                            "Strike Price → strike_price": "✅",
                            "Entry Price → entry_price": "✅",
                            "Final P&L → final_pnl": "✅",
                            "Closing Price → close_price": "✅",
                            "Close Date → close_date": "✅"
                        }
                        
                        for mapping, status in mapping_info.items():
                            st.write(f"{status} {mapping}")
                        
                        # Import button
                        if st.button("Import All Trades", type="primary"):
                            with st.spinner("Importing trades..."):
                                imported_count = import_csv_trades(csv_content)
                                
                            if imported_count > 0:
                                st.success(f"Successfully imported {imported_count} trades!")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("Failed to import trades. Please check the format.")
                                
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.write("Please ensure your CSV file has the correct format.")
                
                # Show expected format
                st.subheader("Expected CSV Format")
                st.write("Your CSV should contain these columns:")
                expected_columns = [
                    "Trade No", "Trade Date", "Underlying", "Expiration Date",
                    "Option Type", "Direction", "Strike Price", "Entry Price", 
                    "Final P&L", "Closing Price", "Close Date"
                ]
                st.write(", ".join([f"`{col}`" for col in expected_columns]))
            
            with col2:
                st.subheader("📤 Export Trades")
                
                trades_df = get_trades()
                
                if not trades_df.empty:
                    st.info(f"Found {len(trades_df)} trades to export")
                    
                    # Export options
                    export_format = st.selectbox(
                        "Export Format",
                        ["CSV", "Excel", "JSON"]
                    )
                    
                    export_type = st.radio(
                        "Export Type",
                        ["All Trades", "Open Trades Only", "Closed Trades Only"]
                    )
                    
                    # Filter data based on selection
                    if export_type == "Open Trades Only":
                        export_df = trades_df[trades_df.get('status', '') == 'Open'] if 'status' in trades_df.columns else pd.DataFrame()
                    elif export_type == "Closed Trades Only":
                        export_df = trades_df[trades_df.get('status', '') == 'Closed'] if 'status' in trades_df.columns else pd.DataFrame()
                    else:
                        export_df = trades_df
                    
                    if not export_df.empty:
                        st.write(f"Exporting {len(export_df)} trades")
                        
                        # Generate filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        if export_format == "CSV":
                            export_data = export_df.to_csv(index=False)
                            filename = f"trades_export_{timestamp}.csv"
                            mime_type = "text/csv"
                        
                        elif export_format == "Excel":
                            from io import BytesIO
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                export_df.to_excel(writer, index=False, sheet_name='Trades')
                            export_data = buffer.getvalue()
                            filename = f"trades_export_{timestamp}.xlsx"
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        
                        else:  # JSON
                            export_data = export_df.to_json(orient='records', date_format='iso')
                            filename = f"trades_export_{timestamp}.json"
                            mime_type = "application/json"
                        
                        st.download_button(
                            label=f"Download {export_format}",
                            data=export_data,
                            file_name=filename,
                            mime=mime_type,
                            type="primary"
                        )
                    else:
                        st.warning(f"No trades found for {export_type.lower()}")
                else:
                    st.warning("No trades available to export")
                
                # Clear all data option
                st.subheader("⚠️ Data Management")
                
                if st.button("Clear All Trades", help="This will permanently delete all trades"):
                    if mongo_connected:
                        collection.delete_many({})
                    else:
                        st.session_state.trades_data = []
                    st.success("All trades cleared!")
                    time.sleep(1)
                    st.rerun()

        # Connection status indicator
        if mongo_connected:
            st.success("✅ Connected to MongoDB")
        else:
            st.warning("⚠️ Running in session-only mode (data will be lost on refresh)")

        # Auto-refresh for open positions
        trades_df = get_trades()
        if not trades_df.empty:
        open_count = len(trades_df[trades_df.get('status', '') == 'Open']) if 'status' in trades_df.columns else 0
        if open_count > 0:
            auto_update = st.checkbox(f"Auto-update open positions ({open_count} open)", value=False)
            if auto_update:
                st.info("🔄 Auto-updating every 60 seconds...")
                time.sleep(60)
                st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("**Data Sources:** Barchart.com, Yahoo Finance | **Last Updated:** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))