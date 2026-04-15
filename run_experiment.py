import argparse
import yfinance as yf
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import warnings 
import matplotlib.pyplot as plt
from yfinance import data

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

def get_data(train_start, train_end, val_end, test_end, tickers, indicators = None, data_path = 'data/datasets'):

    downloader = TestYahooDownloader(train_start, test_end, tickers)
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

    train_df = data_split(df, train_start, train_end)
    val_df = data_split(df, train_end, val_end)
    test_df = data_split(df, val_end, test_end)

    return train_df, val_df, test_df, df.tic.unique()

def eval_agent(agent, test_set, tickers):
    pass

def get_dynamics_model(model_kwargs):
    from models.get_model import get_model
    print(f'Getting dynamics model {model_kwargs.get("model_name")} with kwargs {model_kwargs}')
    return get_model(**model_kwargs)        

def get_agent(agent_name, agent_class, agent_kwargs, env_train, dynamics_kwargs = None):
    agent = DRLAgent(env = env_train)
    baselines = ['ddpg','ppo','td3']
    basic_strategies = ['buy_and_hold']
    model_based_agents = ['dense_td3', 'autoformer_td3']

    if agent_name in baselines:
        model = agent.get_model(agent_name, model_kwargs = agent_kwargs)
    elif agent_name in basic_strategies:
        model = agent_class(**agent_kwargs)
    elif agent_name in model_based_agents:
        dynamics_model = get_dynamics_model(dynamics_kwargs)
        model = agent_class(
            policy="ModelBasedMLP",
            env=env_train,
            dynamics_model=dynamics_model,
            learning_starts = 150,
            tensorboard_log=None,
            verbose=1,
            policy_kwargs=None,
            seed=None,
            **agent_kwargs,
        )
    else:
        raise ValueError(f'Agent {agent_name} not recognized. Must be one of {baselines + basic_strategies + model_based_agents}')
    
    return model, agent
        
def sample_tickers(train_set, val_set, test_set, tickers, assets_per_ep):

    if assets_per_ep is None or assets_per_ep >= len(tickers):
        return train_set, val_set, test_set, tickers

    tics = np.random.choice(tickers, size = assets_per_ep, replace = False)

    train = train_set[train_set.tic.isin(tics)]
    val = val_set[val_set.tic.isin(tics)]
    test = test_set[test_set.tic.isin(tics)]

    return train, val, test, tics

def run_trial(agents, train_set, val_set, test_set, tickers, env_episodes = 100, episodes_per_env = 100, assets_per_ep = 10 , agents_kwargs = None, indicators = []):
    '''
    Run a single trial for the given agents on the given dataset.
    agents:: dict('names':'agent_class')
    train_set: pd.DataFrame
    val_set: pd.DataFrame
    test_set: pd.DataFrame
    tickers: array of tickers in the dataset
    episodes: int
    assets_per_ep: int
    agents_kwargs: dict of dicts of kwargs for each agent, e.g. {'agent_name': {'lr': 0.001, 'gamma': 0.99}}
    '''

    # def print_head(df, name):
    #     print(name)
    #     print(df.loc[:, ['date', 'close', 'tic']].head())

    # print_head(train_set, 'train')
    # print_head(val_set, 'val')
    # print_head(test_set, 'test')
    
    print(f'Running trial for agents {agents} on dataset with {len(tickers)} ticker(s)  \n')


    stock_dimension = assets_per_ep
    state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
            
    env_kwargs = {
    "hmax": 100,
    "initial_amount": 100_000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
    }

    for agent_name, agent_class in agents.items():
        print(f'Training agent {agent_name} \n')
        agent_kwargs = agents_kwargs[agent_name]

        weights = None

        for episode in range(1, env_episodes+1):

            train, val, test, tics = sample_tickers(train_set, val_set, test_set, tickers, assets_per_ep)
            print(f'Episode {episode} with tickers {tics} \n')
            # print_head(train, 'train')
            # print_head(val, 'val')
            # print_head(test, 'test')

            e_train_gym = StockTradingEnv(df = train, **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()

            print('action space',env_train.action_space)

            model, trainer = get_agent(agent_name, 
                              agent_class, 
                              agent_kwargs, 
                              env_train, 
                              dynamics_kwargs = agent_kwargs['dynamics_kwargs'] 
                              if 'dynamics_kwargs' in agent_kwargs else None)

            if weights is not None:
                model.load_weights(weights)

            trainer.train_model(model = model, tb_log_name=agent_name, total_timesteps=10000)

            weights = None # TODO save model weights and load in next episode for continued training

            assert False, 'Breakpoint'


def run_experiments(number_of_trials, agents, dataset, env_episodes = 100, episodes_per_env = 100, assets_per_ep = 10, agents_kwargs = None, indicators = []):

    for data_name in dataset.keys():
        train_set, val_set, test_set, tickers = get_data('2014-01-01', '2024-01-01', '2025-01-01', '2026-01-01', dataset[data_name], indicators)
        for trial in range(1, number_of_trials+1):
            print(f'Running trial {trial} \n')
            run_trial(agents, train_set, val_set, test_set, tickers, env_episodes, episodes_per_env, assets_per_ep , agents_kwargs, indicators)
    

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

    n_trials = 1
    assets_per_ep = 10
    env_episodes = 10
    episodes_per_env = 100
    agents = ['buy_and_hold', 'dense_model', 'dense_td3']
    agents = {'autoformer_td3': ModelBasedTD3}

    agent_kwargs = {
        'ppo':{
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
            },
        'td3':{"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
        'autoformer_td3':{"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001,  "dynamics_kwargs": {'model_name': 'Autoformer', "feature_dim":assets_per_ep}},
        'dense_td3':{"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001,  "dynamics_kwargs": {"model_name": 'Dense',"feature_dim":assets_per_ep}}
        }
    all_tickers = {'sp100':all_tickers['sp100']}

    run_experiments(n_trials, agents, all_tickers, env_episodes = env_episodes, episodes_per_env = episodes_per_env, assets_per_ep = assets_per_ep, agents_kwargs = agent_kwargs, indicators = indicators)


