
import yfinance as yf
import numpy as np
import gymnasium as gym
import pandas as pd
import os

from data.tickers import all_tickers
from data.test_yahoo_downloader import TestYahooDownloader

from finrl import config_tickers
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from test_env_stocktrading import StockTradingEnv

from models.denseModel import train_dense
from exp.exp_main import Exp_Main

from conf import get_config

from agents.basicStrategies import BasicStrategy, BasicStrategy_auto, Buy_And_Hold



def create_args(all_tickers, indicators = None):

    args = {}
    settings = {}

    for tic in all_tickers.keys():
        args[tic], settings[tic] = get_config(all_tickers[tic], tic, indicators = indicators)

    return args, settings

def get_data(args, tickers, indicators = None):

    test_sets = {}

    print('Indicators:',indicators)

    for tics, arg in args.items():
        print(f'{tics} start:{arg.start_training} end:{arg.end_training} training end:{arg.end_testing}')

        downloader = TestYahooDownloader(arg.start_training, arg.end_testing, tickers[tics])
        df = downloader.fetch_data()
        os.makedirs(arg.root_path, exist_ok = True)

        train_data = data_split(df, arg.start_training, arg.end_training)

        df = data_split(df, arg.end_training, arg.end_testing)

        print('Train tail:')
        print(train_data.tail(10))
        print('test head:')
        print(df.head(10))

        downloader.save_as_csv(os.path.join(arg.root_path, arg.data_path), data = train_data)

        fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
        )
        
        df_processed = fe.preprocess_data(df)

        test_sets[tics] = df_processed

        print(df_processed.head(10))

    return test_sets


def run_strategy(data, agent, tickers, indicators = []):

    stock_dimension = len(tickers)
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension

    env = StockTradingEnv(data, 
                          reward_scaling = 1e-3,
                          state_space = stock_dimension,
                          action_space = stock_dimension,
                          tech_indicator_list = indicators,
                          num_stock_shares = [0] * stock_dimension,
                          stock_dim = stock_dimension, 
                          hmax = 100,
                          initial_amount = 10000,
                          buy_cost_pct = [0.001] * stock_dimension,
                          sell_cost_pct = [0.001] * stock_dimension,
                          turbulence_threshold = None,
                          print_verbosity = 1,
                          )
    
    state, dict = env.reset()
    done = False

    states = []
    rewards = []
    actions = []
   
    print(f'Start state: {state} \n')
    steps = 0
    while not done:
        steps+=1
        action = agent.get_action(state, env._get_date())
        actions.append(action)
        print(action.shape)
        state, reward, done, _, dict = env.step(action)
        states.append(state)
        rewards.append(rewards)
        
        print(f' At step {steps} took action {action} got {reward} ')

        print('s',state)
        print('p',state[1])
        print('date',env._get_date())

        if steps>10:
            break

    return states, actions, rewards



if __name__ == '__main__':

    # call arguments

    # call training datasets with corresponding enviroment test sets

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

    csi100_tic = all_tickers['csi100']
    sp100_tic = all_tickers['sp100']
    nasdaq100_tic = all_tickers['nasdaq100']

    all_tickers = { 'csi100':csi100_tic }
    all_tickers = {'amg':['AAPL','MSFT','GOOGL']}
    #all_tickers = {'aapl':['AAPL']}
    
    all_args, all_settings = create_args(all_tickers)

    print(all_args)

    test_sets = get_data(all_args, all_tickers, indicators = indicators)

    for tic in all_tickers.keys():
        print(f'Now running the {tic} test and train scheme')
        args = all_args[tic]
        tickers = all_tickers[tic]

        print('Training Dense')
        args.train_epochs = 1

        # dense = train_dense(args)

        buy_and_hold = Buy_And_Hold(args)
        df = test_sets[tic]


        df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        print(df.head(10))

        bnh_rewards, bnh_actions, bnh_states = run_strategy(df, buy_and_hold, tickers, indicators=indicators)

        break

        print('Training Autoformer')

        autoformer = Exp_Main(args)
        autoformer.train(all_settings[tic])

        print('Training Done')

        dense_strat = BasicStrategy(args, dense, dense.scaler)
        auto_strat = BasicStrategy_auto(args, autoformer, autoformer.scaler)

        df = test_sets[tic]

        dense_rewards, dense_actions, dense_states = run_strategy(df, dense_strat, tickers)
        auto_rewards, auto_actions, auto_states = run_strategy(df, auto_strat, tickers)
        bnh_rewards, bnh_actions, bnh_states = run_strategy(df, buy_and_hold, tickers)






    

    # call models

    # train models

    # run models on enviroments

    # turn data to graphs
