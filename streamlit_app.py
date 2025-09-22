import streamlit as st
import requests
import pandas as pd
from urllib.parse import unquote
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import datetime
import time

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
    tab1, tab2, tab3, tab4 = st.tabs(["Futures Curve with Options", "Individual Contract Analysis", "Raw Data", "Analytics"])
    
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
        st.subheader("Individual Contract Analysis")
        
        # Contract selector
        selected_contract = st.selectbox(
            "Select a contract to analyze:",
            options=merged_df['Contract'].tolist(),
            index=0
        )
        
        # Get data for selected contract
        contract_data = merged_df[merged_df['Contract'] == selected_contract].iloc[0]
        
        # Display contract metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Last Price", f"${contract_data['Last Price']:.2f}")
        
        with col2:
            if 'priceChange' in contract_data and pd.notna(contract_data['priceChange']):
                change = contract_data['priceChange']
                st.metric("Price Change", f"${change:.2f}", delta=f"{change:.2f}")
            else:
                st.metric("Price Change", "N/A")
        
        with col3:
            if 'Options Days to Expiry' in contract_data:
                st.metric("Days to Expiry", f"{contract_data['Options Days to Expiry']:.0f}")
            else:
                st.metric("Days to Expiry", "N/A")
        
        with col4:
            if 'Futures Implied Volatility' in contract_data:
                st.metric("Implied Volatility", f"{contract_data['Futures Implied Volatility']:.2f}%")
            else:
                st.metric("Implied Volatility", "N/A")
        
        # Create a single contract curve (historical simulation)
        st.subheader(f"Price Analysis for {selected_contract}")
        
        # Since we don't have historical data, create a simulated curve based on current price
        import numpy as np
        current_price = contract_data['Last Price']
        
        # Generate sample data points (in a real app, this would be historical data)
        days = list(range(-30, 1))  # Last 30 days
        np.random.seed(42)  # For consistent results
        price_volatility = 0.02  # 2% daily volatility
        random_walks = np.cumsum(np.random.normal(0, price_volatility, len(days)))
        prices = [current_price + (current_price * walk) for walk in random_walks]
        
        # Create DataFrame for the curve
        curve_df = pd.DataFrame({
            'Days': days,
            'Price': prices,
            'Date': [datetime.datetime.now() + datetime.timedelta(days=d) for d in days]
        })
        
        # Plot the individual contract curve
        fig_individual = go.Figure()
        
        fig_individual.add_trace(go.Scatter(
            x=curve_df['Date'],
            y=curve_df['Price'],
            mode='lines+markers',
            name=f'{selected_contract} Price',
            line=dict(color='green', width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add current price line
        fig_individual.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig_individual.update_layout(
            title=f'{selected_contract} Price Curve (Simulated Historical Data)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_individual, use_container_width=True)
        
        # Additional contract details
        st.subheader("Contract Details")
        
        details_df = pd.DataFrame({
            'Attribute': ['Contract Symbol', 'Last Price', 'Price Change', 'Volume', 'Open Interest', 
                         'Expiration Date', 'Days to Expiry', 'Implied Volatility'],
            'Value': [
                contract_data['Contract'],
                f"${contract_data['Last Price']:.2f}",
                f"${contract_data.get('priceChange', 'N/A'):.2f}" if pd.notna(contract_data.get('priceChange')) else 'N/A',
                f"{contract_data.get('volume', 'N/A'):,.0f}" if pd.notna(contract_data.get('volume')) else 'N/A',
                f"{contract_data.get('openInterest', 'N/A'):,.0f}" if pd.notna(contract_data.get('openInterest')) else 'N/A',
                contract_data.get('Expiration Date', 'N/A'),
                f"{contract_data.get('Options Days to Expiry', 'N/A'):.0f}" if pd.notna(contract_data.get('Options Days to Expiry')) else 'N/A',
                f"{contract_data.get('Futures Implied Volatility', 'N/A'):.2f}%" if pd.notna(contract_data.get('Futures Implied Volatility')) else 'N/A'
            ]
        })
        
        st.dataframe(details_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Raw Data")
        
        # Display raw futures data
        st.subheader("Futures Data")
        display_df = df.head(num_contracts).copy()
        display_df['lastPrice'] = display_df['lastPrice'].apply(lambda x: f"${x:.2f}")
        display_df['priceChange'] = display_df['priceChange'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        display_df['openInterest'] = display_df['openInterest'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df[['symbol', 'lastPrice', 'priceChange', 'volume', 'openInterest']], 
                    use_container_width=True)
        
        # Display options data if available
        if not options_df.empty:
            st.subheader("Options Data")
            st.dataframe(options_df, use_container_width=True)
        
        # Volume and Open Interest charts
        col1, col2 = st.columns(2)
        
        with col1:
            df_volume = df.head(num_contracts).copy()
            df_volume['volume_clean'] = pd.to_numeric(df_volume['volume'], errors='coerce').fillna(0)
            fig_volume = px.bar(df_volume,
                               x='symbol',
                               y='volume_clean',
                               title='Trading Volume by Contract',
                               labels={'volume_clean': 'Volume'})
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            df_oi = df.head(num_contracts).copy()
            df_oi['openInterest_clean'] = pd.to_numeric(df_oi['openInterest'], errors='coerce').fillna(0)
            fig_oi = px.bar(df_oi,
                           x='symbol',
                           y='openInterest_clean',
                           title='Open Interest by Contract',
                           labels={'openInterest_clean': 'Open Interest'})
            st.plotly_chart(fig_oi, use_container_width=True)
    
    with tab4:
        st.subheader("Analytics")
        
        # Price distribution
        fig_hist = px.histogram(df.head(num_contracts),
                               x='lastPrice',
                               title='Price Distribution',
                               nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Correlation matrix if options data is available
        if 'options_data' in st.session_state:
            options_data = st.session_state['options_data']
            
            # Numeric columns for correlation
            numeric_cols = ['lastPrice', 'Options Days to Expiry', 'Futures Implied Volatility']
            correlation_data = options_data[numeric_cols].corr()
            
            fig_corr = px.imshow(correlation_data,
                               text_auto=True,
                               aspect="auto",
                               title='Correlation Matrix')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.head(num_contracts).describe(), use_container_width=True)

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("**Data Source:** Barchart.com | **Last Updated:** " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))