import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests

class StockPortfolioDashboard:
    def __init__(self):
        # Configure Streamlit page
        st.set_page_config(
            page_title="Stock Portfolio Dashboard",
            page_icon=":chart_with_upwards_trend:",
            layout="wide"
        )
        
        # Initialize session state for portfolios
        if 'portfolios' not in st.session_state:
            st.session_state.portfolios = {
                'Tech Innovators': [
                    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'CRM', 'ADBE', 'INTC'
                ],
                'Healthcare Disruptors': [
                    'UNH', 'JNJ', 'MRNA', 'GILD', 'ISRG', 'BIIB', 'ABC'
                ]
            }
        
        # Fetch market indicators
        self.market_indicators = self.get_market_indicators()

    def get_market_indicators(self):
        """
        Fetch key market indicators
        Returns a dictionary of current market sentiment indicators
        """
        try:
            # VIX from Yahoo Finance
            vix = yf.Ticker('^VIX')
            vix_info = vix.history(period='1d')
            
            # Fetch put/call ratio and other indicators
            return {
                'VIX': round(vix_info['Close'].iloc[-1], 2) if not vix_info.empty else 'N/A',
                'Market Trend': self.get_market_trend(),
                'Advance-Decline Ratio': self.get_advance_decline_ratio()
            }
        except Exception as e:
            st.error(f"Error fetching market indicators: {e}")
            return {
                'VIX': 'N/A',
                'Market Trend': 'N/A',
                'Advance-Decline Ratio': 'N/A'
            }

    def get_market_trend(self):
        """
        Determine overall market trend
        """
        try:
            sp500 = yf.Ticker('^GSPC')
            sp500_hist = sp500.history(period='1mo')
            
            if len(sp500_hist) < 2:
                return 'Insufficient Data'
            
            price_change = sp500_hist['Close'].iloc[-1] - sp500_hist['Close'].iloc[0]
            percent_change = (price_change / sp500_hist['Close'].iloc[0]) * 100
            
            if percent_change > 1:
                return 'Bullish'
            elif percent_change < -1:
                return 'Bearish'
            else:
                return 'Neutral'
        except Exception as e:
            st.error(f"Error determining market trend: {e}")
            return 'N/A'

    def get_advance_decline_ratio(self):
        """
        Estimate advance-decline ratio (mock implementation)
        """
        try:
            # This is a simplified mock. In a real-world scenario, you'd use a financial API
            nyse_adv = yf.Ticker('^NYA')  # NYSE Composite Index
            nyse_hist = nyse_adv.history(period='1d')
            
            if nyse_hist.empty:
                return 'N/A'
            
            today_close = nyse_hist['Close'].iloc[-1]
            prev_close = nyse_hist['Close'].iloc[0]
            
            return round(today_close / prev_close, 2)
        except Exception as e:
            st.error(f"Error calculating advance-decline ratio: {e}")
            return 'N/A'

    def fetch_stock_data(self, symbols, start_date, end_date):
        """
        Fetch comprehensive stock data for given symbols
        """
        stock_data = {}
        for symbol in symbols:
            try:
                # Fetch stock data
                stock = yf.Ticker(symbol)
                
                # Historical prices
                hist = stock.history(start=start_date, end=end_date)
                
                # Fundamental data
                info = stock.info
                
                stock_data[symbol] = {
                    'historical_data': hist,
                    'current_price': info.get('currentPrice', 'N/A'),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'dividend_yield': info.get('dividendYield', 'N/A'),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                    '52_week_low': info.get('fiftyTwoWeekLow', 'N/A')
                }
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
                stock_data[symbol] = None
        
        return stock_data

    def performance_analysis(self, stock_data):
        """
        Comprehensive performance analysis for stocks
        """
        performance_metrics = {}
        for symbol, data in stock_data.items():
            if data is None:
                continue
            
            hist = data['historical_data']
            if hist.empty:
                continue
            
            # Calculate returns
            returns = hist['Close'].pct_change()
            
            performance_metrics[symbol] = {
                'Total Return': returns.cumsum()[-1] * 100,
                'Volatility': returns.std() * np.sqrt(252) * 100,  # Annualized
                'Sharpe Ratio': self.calculate_sharpe_ratio(returns)
            }
        
        return performance_metrics

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe Ratio
        """
        try:
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            return round(sharpe_ratio, 2)
        except Exception:
            return 'N/A'

    def create_performance_chart(self, stock_data):
        """
        Create interactive performance chart
        """
        chart_data = []
        for symbol, data in stock_data.items():
            if data is None:
                continue
            
            hist = data['historical_data']
            if hist.empty:
                continue
            
            # Normalize prices
            normalized_prices = hist['Close'] / hist['Close'].iloc[0] * 100
            
            for date, price in zip(hist.index, normalized_prices):
                chart_data.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Normalized Price': price
                })
        
        df = pd.DataFrame(chart_data)
        
        # Create interactive line chart
        fig = px.line(
            df, 
            x='Date', 
            y='Normalized Price', 
            color='Symbol',
            title='Stock Performance (Normalized)',
            labels={'Normalized Price': 'Performance (%)'}
        )
        
        return fig

    def run_dashboard(self):
        """
        Main Streamlit dashboard
        """
        st.title("📈 Stock Portfolio Dashboard")
        
        # Sidebar for portfolio selection
        st.sidebar.header("Portfolio Selection")
        selected_portfolio = st.sidebar.selectbox(
            "Choose a Portfolio", 
            list(st.session_state.portfolios.keys())
        )
        
        # Date range selection
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
        end_date = st.sidebar.date_input(
            "End Date", 
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Fetch stock data
        symbols = st.session_state.portfolios[selected_portfolio]
        stock_data = self.fetch_stock_data(symbols, start_date, end_date)
        
        # Performance Analysis
        performance_metrics = self.performance_analysis(stock_data)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "Portfolio Overview", 
            "Performance Analysis", 
            "Market Indicators"
        ])
        
        with tab1:
            # Portfolio Overview
            st.header(f"{selected_portfolio} Portfolio")
            
            # Stock Data Table
            stock_overview_data = []
            for symbol, data in stock_data.items():
                if data is None:
                    continue
                stock_overview_data.append({
                    'Symbol': symbol,
                    'Current Price': data['current_price'],
                    'P/E Ratio': data['pe_ratio'],
                    'Market Cap': data['market_cap'],
                    'Dividend Yield': data['dividend_yield']
                })
            
            st.dataframe(pd.DataFrame(stock_overview_data))
            
            # Performance Chart
            performance_chart = self.create_performance_chart(stock_data)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        with tab2:
            # Detailed Performance Analysis
            st.header("Performance Metrics")
            
            # Performance Metrics Table
            perf_metrics_data = []
            for symbol, metrics in performance_metrics.items():
                perf_metrics_data.append({
                    'Symbol': symbol,
                    'Total Return (%)': round(metrics['Total Return'], 2),
                    'Volatility (%)': round(metrics['Volatility'], 2),
                    'Sharpe Ratio': metrics['Sharpe Ratio']
                })
            
            st.dataframe(pd.DataFrame(perf_metrics_data))
        
        with tab3:
            # Market Indicators
            st.header("Market Sentiment Indicators")
            
            # Display market indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Volatility Index (VIX)", 
                    value=self.market_indicators['VIX']
                )
            
            with col2:
                st.metric(
                    label="Market Trend", 
                    value=self.market_indicators['Market Trend']
                )
            
            with col3:
                st.metric(
                    label="Advance-Decline Ratio", 
                    value=self.market_indicators['Advance-Decline Ratio']
                )

# Streamlit app entry point
def main():
    dashboard = StockPortfolioDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
