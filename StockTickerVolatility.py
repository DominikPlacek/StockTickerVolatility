import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock prices for the specified ticker between start_date and end_date.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_daily_returns(stock_data):
    """
    Calculates daily returns from the stock's close prices.
    """
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    return stock_data

def calculate_volatility(stock_data):
    """
    Calculates the annualized volatility based on daily returns.
    """
    return np.std(stock_data['Daily_Return']) * np.sqrt(252)  # Annualizing volatility

def forecast_volatility(stock_data, window=20):
    """
    Forecasts volatility using a rolling window approach.
    """
    return stock_data['Daily_Return'].rolling(window=window).std() * np.sqrt(252)

def plot_combined(stock_data):
    """
    Combines enhanced volatility and daily return distribution plots in a single interactive window.
    """
    # Create subplots
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Stock Price and Volatility Over Time", "Distribution of Stock Daily Returns"))

    # Plot 1: Stock Price
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)

    # Add Volatility on secondary y-axis
    fig.add_trace(go.Scatter(x=stock_data.index, y=forecast_volatility(stock_data), name='Volatility', line=dict(color='red')), row=1, col=1, secondary_y=True)

    # Plot 2: Distribution of Daily Returns
    returns = stock_data['Daily_Return'].dropna()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Daily Returns', marker_color='blue', opacity=0.6), row=2, col=1)

    # Update yaxis properties
    fig.update_layout(yaxis_title='Close Price', yaxis2_title='Volatility', yaxis3_title='Density')
    fig.update_layout(height=800, showlegend=False)

    # Update layout for better readability
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    fig.show()

def main():
    """
    Main function to run the enhanced project workflow with combined interactive plotting.
    """
    ticker = input("Enter stock ticker: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = calculate_daily_returns(stock_data)

    print(f"Calculated Volatility: {calculate_volatility(stock_data):.2%}")

    plot_combined(stock_data)

if __name__ == "__main__":
    main()
