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


if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("**Data Sources:** Barchart.com, Yahoo Finance | **Last Updated:** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))