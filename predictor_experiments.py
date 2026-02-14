
import yfinance as yf
import numpy as np
import gymnasium as gym
import pandas as pd
import os
import torch
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

from results.plot_results import plot_results


def create_args(all_tickers, indicators = None):

    args = {}
    settings = {}

    for tic in all_tickers.keys():
        args[tic], settings[tic] = get_config(all_tickers[tic], tic)

    return args, settings

def get_data(args, tickers, indicators = None):

    test_sets = {}

    print('Indicators:',indicators)

    for tics, arg in args.items():
        print(f'{tics} start:{arg.start_training} end:{arg.end_training} then testing end:{arg.end_testing}')

        downloader = TestYahooDownloader(arg.start_training, arg.end_testing, tickers[tics])
        df = downloader.fetch_data()
        os.makedirs(arg.root_path, exist_ok = True)

        fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
        )
        
        df = fe.preprocess_data(df)

        train_data = data_split(df, arg.start_training, arg.end_training)
        df = data_split(df, arg.end_training, arg.end_testing)
        # Save the training data as csv for the training loops to access
        downloader.save_as_csv(os.path.join(arg.root_path, arg.data_path), data = train_data)

        print(f'For dataset {tics}, training set starts: {train_data['date'].min()} and ends {train_data['date'].max()}')
        print(f'For dataset {tics}, training set starts: {df['date'].min()} and ends {df['date'].max()}')

        # Add to the dict for testing in env
        test_sets[tics] = df

    return test_sets


def run_strategy(data, agent, tickers, indicators = [], name = 'n/a', dataset_name = 'n/a'):

    print(f'Running {name} on dataset {dataset_name} from {data['date'].min()} to {data['date'].max()} in StockTradingEnv \n')

    stock_dimension = len(tickers)
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension

    print('Stock dims:',stock_dimension)
    print('state space:',state_space)

    env = StockTradingEnv(data, 
                          reward_scaling = 1e-3,
                          state_space = state_space,
                          action_space = stock_dimension,
                          tech_indicator_list = indicators,
                          num_stock_shares = [0] * stock_dimension,
                          stock_dim = stock_dimension, 
                          hmax = 100,
                          initial_amount = 100_000,
                          buy_cost_pct = [0.001] * stock_dimension,
                          sell_cost_pct = [0.001] * stock_dimension,
                          turbulence_threshold = None,
                          print_verbosity = 1,
                          make_plots=True
                          )
    
    state, dict = env.reset()
    done = False


    states = [state]
    rewards = []
    actions = []
    predictions = []
    actuals = []

    data = {'states':[states], 'rewards':rewards, 'actions':actions, 'predicitons':predictions, 'actuals':actuals }


    print(f'Start state: {state} on date {env._get_date()} \n')
    steps = 0
    while not done:
        steps+=1
        action, prediction = agent.get_action(state, env._get_date())
        predictions.append(prediction)
        actions.append(action)

        # assert  env.action_space.contains(action), f'Action of shape {action.shape} and {action} \n invalid as space{env.action_space} sample: {env.action_space.sample()}'

        # print(action.shape)
        state, reward, done, _, dict = env.step(action)
        actuals.append(state[1:args.enc_in + 1])
        states.append(state)
        rewards.append(rewards)
        
    return data

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

    all_args, all_settings = create_args(all_tickers, '')

    test_sets = get_data(all_args, all_tickers, indicators = indicators)

    n_trials = 1

    trials = {}
    

    for trial in range(1, n_trials+1):

        results = {}
        trials[trial] = results
        
        for tic in all_tickers.keys():

            results[tic] = {'MLP Prediction':{}, 'Autoformer Prediction':{}, 'Buy And Hold': {}}        

            print(f'Now running the {tic} test and train scheme')
            args = all_args[tic]
            tickers = all_tickers[tic]

            print('Training Dense')

            dense = train_dense(args)

            buy_and_hold = Buy_And_Hold(args, hmax = 100)
            df = test_sets[tic]

            results[tic]['Buy And Hold'] = run_strategy(df, 
                                                                                                        buy_and_hold, 
                                                                                                        tickers, 
                                                                                                        indicators=indicators, 
                                                                                                        name= 'buy and hold', 
                                                                                                        dataset_name=tic)
            
            dense_strat = BasicStrategy_auto(args, dense, dense.scaler, hmax = 100)

            dense_results = results[tic]['MLP Prediction'] 

            data = run_strategy(df,
                                                                                                                                                    dense_strat, 
                                                                                                                                                    tickers, 
                                                                                                                                                    indicators=indicators, 
                                                                                                                                                    name = 'dense prediction', 
                                                                                                                                                    dataset_name=tic)

            print('Training Autoformer')

            autoformer = Exp_Main(args)
            autoformer.train(all_settings[tic])

            print('Training Done')

            auto_strat = BasicStrategy_auto(args, autoformer, autoformer.scaler, hmax = 100)

            auto_results = results[tic]['Autoformer Prediction'] 

            data  = run_strategy(df, auto_strat, tickers, indicators=indicators, name = 'autoformer prediction', dataset_name=tic)

    plot_results(results = results, features=3)

    print(list(trials.keys()))
    print(list(trials[1].keys()))
    
    # save_results()
