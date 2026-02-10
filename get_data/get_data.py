import pandas as pd

import yfinance as yf
import pandas as pd
from stockstats import StockDataFrame as Sdf

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config import INDICATORS



def return_sp100_tick():
    return [
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

def download_finrl_data(
    tickers,
    start_date,
    end_date,
    tech_indicators=None,
    save_path = None
):
    """
    Fetches and formats market data for FinRL environments.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (e.g. ["AAPL", "MSFT"])
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    tech_indicators : list[str], optional
        List of technical indicators (e.g. ["macd", "rsi_30", "cci_30"])

    Returns
    -------
    pd.DataFrame
        FinRL-compatible dataframe
    """

    all_data = []

    print(f'Downloading {len(tickers)} tickers')

    for tic in tickers:
        df = yf.download(
            tic,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            continue

        # print(df.head())

        df.reset_index(inplace=True)

        df.columns = [c[0].lower() for c in df.columns]
        # df.columns = [c.lower() for c in df.columns]

        # print(df.columns)

        df["tic"] = tic

        # Keep only FinRL-required columns
        df = df[[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic"
        ]]

        # print(df.head())

        all_data.append(df)

        # break 


    # data = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list = tickers).fetch_data()

    # print(data.head())

    # assert False

    data = pd.concat(all_data, ignore_index=True)

    # Sort is VERY important for FinRL
    data.sort_values(by=["date", "tic"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # # Add technical indicators if requested
    # if tech_indicators:
    #     stock_df = Sdf.retype(data.copy())
    #     for indicator in tech_indicators:
    #         stock_df[indicator]

    #     data = stock_df.reset_index()

    # data = check_and_fix_nans(data)

    if save_path is not None:
        pass


    return data



def check_and_fix_nans(
    df,
    group_col="tic",
    date_col="date",
    verbose=True
):
    """
    Checks and fixes NaNs in a FinRL-compatible dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        FinRL-style dataframe
    group_col : str
        Ticker column name
    date_col : str
        Date column name
    verbose : bool
        Print NaN diagnostics

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with no NaNs
    """

    if verbose:
        nan_summary = df.isna().sum()
        nan_summary = nan_summary[nan_summary > 0]

        if not nan_summary.empty:
            print("NaN summary BEFORE cleaning:")
            print(nan_summary)
        else:
            print("No NaNs detected.")

    # Sort first (important for ffill)
    df = df.sort_values(by=[group_col, date_col]).copy()

    # Forward + backward fill per ticker
    df = (
        df.groupby(group_col, group_keys=False)
        .apply(lambda x: x.ffill().bfill())
    )

    # Final safety drop
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        if verbose:
            print(f"Dropping {remaining_nans} remaining NaN values.")
        df = df.dropna()

    if verbose:
        print("NaN summary AFTER cleaning:")
        print(df.isna().sum())

    return df.reset_index(drop=True)

def get_data_file(path, tickers = None, download = False):

    if download and tickers is not None:
        #download data
        pass



