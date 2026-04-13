import yfinance as yf
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import warnings 
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from data.tickers import all_tickers 
from data.test_yahoo_downloader import TestYahooDownloader

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from enviroments.test_env_stocktrading import StockTradingEnv 

from agents.basicStrategies import Buy_And_Hold
from models.denseModel import train_dense

from stable_baselines3.common.logger import configure

from finrl.config import TRAINED_MODEL_DIR
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from agents.modelbased_TD3.mode_based_TD3 import ModelBasedTD3



def get_data(test_start, test_end, train_end, tickers, indicators = None, data_path = 'data/datasets'):

    test_sets = {}
    train_sets = {}
    for tic in tickers.keys():

        downloader = TestYahooDownloader(test_start, train_end, tickers[tic])
        df = downloader.fetch_data()
        os.makedirs(data_path, exist_ok = True)

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=indicators,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )

        df = fe.preprocess_data(df)

        train_data = data_split(df, test_start, test_end)
        df = data_split(df, test_end, train_end)

        downloader.save_as_csv(os.path.join(data_path, tic + '.csv' ), data = train_data)

        train_sets[tic] = train_data
        test_sets[tic] = df
        tickers[tic] = df.tic.unique()

    return train_sets, test_sets, tickers


if __name__ == '__main__':

    


