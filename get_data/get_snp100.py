#!/usr/bin/env python3
"""
S&P 100 Stock Data Downloader
Downloads historical stock data for all S&P 100 constituents from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse

# S&P 100 ticker symbols (as of recent composition)
SP100_TICKERS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
    'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C',
    'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
    'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD',
    'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC',
    'INTU', 'ISRG', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA',
    'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT',
    'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL',
    'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TSLA',
    'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
]
output_dir='sp100_data'
def download_sp100_data(start_date=None, end_date=None, period='1y', interval = '1d'):
    """
    Download historical stock data for S&P 100 companies
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, optional
        Period to download if start/end dates not specified
        Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    Returns:
    --------
    dict : Dictionary containing DataFrames for each ticker
    """
    
    print(f"Downloading data for {len(SP100_TICKERS)} S&P 100 stocks...")
    print(f"Period: {period if not start_date else f'{start_date} to {end_date}'}")
    print("-" * 60)
    
    stock_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(SP100_TICKERS, 1):
        try:
            print(f"[{i}/{len(SP100_TICKERS)}] Downloading {ticker}...", end=' ')
            
            if start_date and end_date:
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            else:
                data = yf.download(ticker, period=period, interval=interval)
            
            if not data.empty:
                stock_data[ticker] = data
                print(f"✓ ({len(data)} rows)")
            else:
                print("✗ (No data)")
                failed_tickers.append(ticker)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"✗ (Error: {str(e)})")
            failed_tickers.append(ticker)
    
    print("-" * 60)
    print(f"Successfully downloaded: {len(stock_data)}/{len(SP100_TICKERS)} stocks")
    
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers)}")
    
    return stock_data

def save_to_csv(stock_data, output_dir=output_dir):
    """
    Save stock data to individual CSV files
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with ticker symbols as keys
    output_dir : str
        Directory to save CSV files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving data to '{output_dir}' directory...")
    
    for ticker, data in stock_data.items():
        filename = f"{output_dir}/{ticker}.csv"
        data.to_csv(filename)
        print(f"Saved: {filename}")
    
    print(f"\nAll files saved to '{output_dir}/' directory")

def create_combined_dataframe(stock_data, column='Close'):
    """
    Create a single DataFrame with specified column for all stocks
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with ticker symbols as keys
    column : str
        Column to extract ('Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close')
    
    Returns:
    --------
    DataFrame : Combined DataFrame with date index and ticker columns
    """
    combined_data = {}

    # aapl = stock_data['AAPL']

    # print(stock_data['AAPL'].head())
    
    # print('keys:', list(stock_data.keys()))

    # print(aapl.columns)
    # print(aapl[['Close']])
    # print(aapl[['Price']])


    # new_index = aapl[['Price', 'Close']]

    # new_index = new_index.rename(columns = {"Price":"Dates", "Close":"Close"})

    # new_index = new_index.drop([0,1])

    # print(new_index.head())


    # new_index = new_index.set_index('Dates')
    # print('reser')

    # print(new_index.head())



    for ticker, data in stock_data.items():
        if 'Close' in data.columns and 'Price' in data.columns:
            data = data[['Price', 'Close']]
            data = data.rename(columns = {"Price":"date", "Close":"close"})
            data = data.drop([0,1])
            data = data.set_index('date')
            data = data['close']
        # print(data.head())
        # print(data.index)

        combined_data[ticker] = data

    combined_df = pd.DataFrame(combined_data)
    return combined_df

def get_all_from_csv_dataframe(directory, tickers):

    stock_dict = {}

    for ticker in tickers:
        
        try:
            df = pd.read_csv(f"{directory}/{ticker}.csv")
            stock_dict[ticker] = df
        except:
            print(f"Couldnt get ticker {ticker} from {directory}/{ticker}.csv" )

    return stock_dict

if __name__ == "__main__":
    # Example usage: Download last 1 year of data
    print("=" * 60)
    print("S&P 100 Stock Data Downloader")
    print("=" * 60)
    

    
    parser = argparse.ArgumentParser(description='S&P100 data downloader')
    parser.add_argument('--download', action='store_true', help='Download data from Yahoo Finance', default=False)
    args = parser.parse_args()


    download = False

    if args.download:
        # Download data
        stock_data = download_sp100_data(start_date='2014-01-01', end_date='2026-01-01')
        
        # Save individual CSV files
        save_to_csv(stock_data)
    else:
        try:
            stock_data = get_all_from_csv_dataframe(output_dir, SP100_TICKERS)
        except:
            print('Data not already downloaded, please run "get_snp100.py true" ')
    
    # Create and save combined closing prices
    if stock_data:
        print("\nCreating combined dataset...")
        combined_close = create_combined_dataframe(stock_data, column='Close')
        combined_close.to_csv('sp100_combined_close.csv')
        print(f"Saved: sp100_combined_close.csv ({combined_close.shape[0]} rows × {combined_close.shape[1]} columns)")
        
        # Display sample
        print("\nSample of combined closing prices:")
        print(combined_close.tail())
    
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
    else:
        print('Dowload Failed')


