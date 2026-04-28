"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import hashlib
import pandas as pd
import yfinance as yf
import os

class TestYahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

        self.data_df = None

    def _cache_path(self, cache_dir: str) -> str:
        ticker_hash = hashlib.md5(",".join(sorted(self.ticker_list)).encode()).hexdigest()[:8]
        filename = f"{self.start_date}_{self.end_date}_{ticker_hash}.csv"
        return os.path.join(cache_dir, filename)

    def fetch_data(self, proxy=None, auto_adjust=False, cache_dir: str | None = None) -> pd.DataFrame:
        """Fetches data from Yahoo API, returning a cached CSV if one exists for the
        same date range and ticker list.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to read/write cached CSVs. Cache filenames encode the date
            range and an 8-char hash of the sorted ticker list so different
            configurations never collide.

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        if cache_dir is not None:
            cache_file = self._cache_path(cache_dir)
            if os.path.exists(cache_file):
                print(f"Loading cached data from {cache_file}")
                self.data_df = pd.read_csv(cache_file, index_col=0, parse_dates=["date"])
                self.data_df["date"] = self.data_df["date"].dt.strftime("%Y-%m-%d")
                return self.data_df
        # Download and save the data in a pandas DataFrame:
        dfs = []
        num_failures = 0
        failures = []
        for tic in self.ticker_list:
            if proxy:
                yf.config.network.proxy = proxy

            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=auto_adjust,
                multi_level_index=False
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                dfs.append(temp_df)
            else:
                num_failures += 1
        data_df = pd.concat(dfs, axis=0) if dfs else pd.DataFrame()
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            if not auto_adjust:
                data_df = self._adjust_prices(data_df)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df["date"].dt.strftime("%Y-%m-%d")
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date"]).reset_index(drop=True) 

        # ENV uses day to find days, so all indecies of the same day need same

        self.data_df = data_df

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = self._cache_path(cache_dir)
            data_df.to_csv(cache_file)
            print(f"Cached data saved to {cache_file}")

        return data_df

    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        # use adjusted close price instead of close price
        data_df["adj"] = data_df["adjcp"] / data_df["close"]
        for col in ["open", "high", "low", "close"]:
            data_df[col] *= data_df["adj"]

        # drop the adjusted close price column
        return data_df.drop(["adjcp", "adj"], axis=1)

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
    
    def save_as_csv(self, path, data = None):

        if data is None:
            data = self.data_df

        assert data is not None, 'No data has been proided'
        print(f'Donwloading data onto path {path}')

        # os.makedirs(path, exist_ok = True)
        data.to_csv(path)
