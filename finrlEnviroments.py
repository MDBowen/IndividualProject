import yfinance as yf
import numpy as np
import gymnasium as gym

import os


from sklearn.preprocessing import StandardScaler

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl import config_tickers

from get_data.get_data import return_sp100_tick
from get_data.get_data import download_finrl_data, return_sp100_tick
from get_data.test_yahoo_downloader import TestYahooDownloader


from models.denseModel import train_dense
from data_provider.data_factory import data_provider

from agents.basicStrategies import BasicStrategy, PredictorStrategy

from conf import get_train_config, get_single_asset_config




def fetch_data(start_date, end_date, tickers, indicators):
    '''Returns data in OHLCV form'''

    downloader = TestYahooDownloader(start_date, end_date, tickers)
    df = downloader.fetch_data()
    # os.makedirs('data/single_yahoo_data', exist_ok = True)
    # downloader.save_as_csv('data/single_yahoo_data/aapl.csv',)

    # If I want to add vix i have to make my own edits
    fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=indicators,
    use_vix=False,
    use_turbulence=True,
    user_defined_feature=False
    )

    df_processed = fe.preprocess_data(df)   

    return df_processed


def simulate_strategy(agent, data, tickers, indicators):

    stock_dimension = len(tickers)
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension

    env = StockTradingEnv(data, 
                          reward_scaling = 1e-3,
                          state_space = state_space,
                          action_space = stock_dimension ,
                          tech_indicator_list = indicators,
                          num_stock_shares = [0] * stock_dimension,
                          stock_dim = stock_dimension, 
                          hmax = 100,
                          initial_amount = 10000,
                          buy_cost_pct = [0.001] * stock_dimension,
                          sell_cost_pct = [0.001] * stock_dimension,
                          turbulence_threshold = 100,
                          print_verbosity = 1,
                          )
    
    state, dict = env.reset()
    done = False

    states = []
    rewards = []
    actions = []
   
    print(f'Start state: {state} /n')
    steps = 0
    while not done:
        steps+=1
        action = agent.get_action(state)
        actions.append(action)
        state, reward, done, _, dict = env.step(action)
        states.append(state)
        rewards.append(rewards)
        
        print(f' At step {steps} took action {action} got {reward} ')

    return states, actions, rewards

if __name__ == '__main__':

    indicators = [              # 8 standard indicators
        'macd',
        'boll_ub',
        'boll_lb',
        'rsi_30',
        'cci_30',
        'dx_30',
        'close_30_sma',
        'close_60_sma'
    ]
    start_date = "2018-01-01"
    end_date = "2023-01-01"
    tickers = ['aapl']

    df = fetch_data(start_date, end_date, tickers, indicators)

    args, setting = get_single_asset_config(root_path = 'data/single_yahoo_data', data_path = 'aapl.csv', data_name = 'aapl')

    print(args.scale)
    args.scale = True

    train_set, training_loader = data_provider(args, flag = 'train')
    # test_set, test_loader = data_provider(args, flag='test')

    print(f'Training model for {len(training_loader)}')
    #  'checkpoints/denseModel/dense_model_checkpoint.pth'
    
    dense = train_dense(training_loader, args, load_path = None)

    print(f'Training finshed')

    scaler = train_set.scaler

    print('Data frame:')
    print(df.head(), '\n')

    agent = BasicStrategy(args, dense, scaler)

    states, actions, rewards = simulate_strategy(agent, df, tickers, indicators)




