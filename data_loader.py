import yfinance as yf
import pandas as pd

import pandas as pd
import os

def load_portfolio(file):
    """
    Loads the portfolio CSV file containing stock tickers.

    Expected CSV format:
        Ticker
        RELIANCE.NS
        TCS.NS
        INFY.NS

    :param file: file path (str) or file-like object
    :return: List of tickers
    """
   
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        file.seek(0) 
        df = pd.read_csv(file)

    if "Ticker" not in df.columns:
        raise ValueError("Portfolio CSV must contain a 'Ticker' column.")

    return df


import yfinance as yf
import pandas as pd
from datetime import datetime

def get_price_data(tickers, file_path=None, period="1y", interval="1d"):
    """
    Fetch price data for given tickers.
    If file_path is provided, reads from CSV with datetime parsing.
    Otherwise, fetches historical data from Yahoo Finance.

    :param tickers: List of stock tickers
    :param file_path: Path to CSV file (optional)
    :param period: Yahoo Finance data period (e.g., '1y', '6mo')
    :param interval: Data interval (e.g., '1d', '1wk', '1mo')
    :return: Dict {ticker: DataFrame}
    """
    data = {}

    # Load from CSV if provided
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values(by='Date')

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        one_year_ago = datetime.now() - pd.DateOffset(years=1)
        df = df[df['Date'] >= one_year_ago]

        data["CSV_Data"] = df
        return data

    # Otherwise fetch from Yahoo Finance
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval)
            if df.empty:
                print(f"⚠ No data found for {ticker}")
                continue
            data[ticker] = df
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")

    return data

if __name__ == "__main__":
    file_path = r"C:\Users\admin\Desktop\ai_portfolio-risk-advisor\data\sample_portfolio.csv"
    tickers = ["AAPL", "MSFT", "GOOGL"]

    price_data = get_price_data(tickers, file_path=file_path)
    for name, df in price_data.items():
        print(f"\n{name} Data:")
        print(df.head())
