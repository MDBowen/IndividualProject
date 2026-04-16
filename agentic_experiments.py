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

check_and_make_directories([TRAINED_MODEL_DIR])

def plot_values(value_dict):

    
    df_dji = TestYahooDownloader(
    start_date=train_end, end_date=test_end, ticker_list = ["^dji"]
    ).fetch_data()

    df_dji = df_dji[["date", "close"]]
    fst_day = df_dji["close"][0]
    dji = pd.merge(
        df_dji["date"],
        df_dji["close"].div(fst_day).mul(1000000),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")

   

    for agent, value in value_dict.items():
        value_dict[agent] =  value.set_index(value.columns[0])['account_value']

    value_dict['dji'] = dji['close']

    results = pd.DataFrame(value_dict)

    plt.rcParams["figure.figsize"] = (15,5)
    plt.figure()
    results.plot()
    plt.savefig('run_results.png')
    plt.show()
    
    

def get_data(train_start, train_end, test_end, tickers, indicators = None, data_path = 'data/datasets'):

    test_sets = {}
    train_sets = {}
    for tic in tickers.keys():

        downloader = TestYahooDownloader(train_start, test_end, tickers[tic])
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

        train_data = data_split(df, train_start, train_end)
        df = data_split(df, train_end, test_end)

        downloader.save_as_csv(os.path.join(data_path, tic + '.csv' ), data = train_data)

        train_sets[tic] = train_data
        test_sets[tic] = df
        tickers[tic] = df.tic.unique()

    return train_sets, test_sets, tickers

def train_agent(agent_n, data, model_kwargs, indicators = [], dataset_name = 'n/a'):
    
    print(f'Training {agent_n} on dataset {dataset_name} from {data['date'].min()} to {data['date'].max()} in StockTradingEnv \n')

    stock_dimension = len(data.tic.unique())
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

    baselines = ['ddpg','ppo','td3']
    MODELS = {'mb_td3': ModelBasedTD3}  #{'mb_td3':{"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001} }
    e_train_gym = StockTradingEnv(df = data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env = env_train)

    if agent_n in baselines:        
        model = agent.get_model(agent_n, model_kwargs = model_kwargs)
    else: 
        from configs.conf import get_config

        dynamics_model = train_dense(get_config( all_tickers['sp100']  ,'sp100')[0])

        model = MODELS[agent_n](
            policy="ModelBasedMLP",
            env=env_train,
            dynamics_model=dynamics_model,
            learning_starts = 150,
            tensorboard_log=None,
            verbose=1,
            policy_kwargs=None,
            seed=None,
            **model_kwargs,
        )
        
    return agent.train_model(model = model, tb_log_name=agent_n, total_timesteps=5_00)

def test_agent(agent, data, indicatos = [], name = 'n/a', dataset_name = 'n/a', baselines = False):

    print(f'Testing {name} on dataset {dataset_name} from {data['date'].min()} to {data['date'].max()} in StockTradingEnv \n')

    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(indicatos) * stock_dimension

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
        "tech_indicator_list": indicatos,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_trade_gym = StockTradingEnv(df = data, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(model = agent, environment = e_trade_gym)

    return df_account_value, df_actions

def run_strategy(data, agent, tickers, indicators = [], name = 'n/a', dataset_name = 'n/a'):

    print(f'Running {name} on dataset {dataset_name} from {data['date'].min()} to {data['date'].max()} in StockTradingEnv \n')

    stock_dimension = len(tickers)
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension

    env = StockTradingEnv(data, 
                          reward_scaling = 1,
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
    actions = []

    initial_worth = state[0] + np.sum( np.array(state[1:agent.feature_len + 1]) *np.array(state[agent.feature_len+1:agent.feature_len*2 + 1]))
    data = {'states':states, 'actions':actions, 'initial_worth':initial_worth}

    while not done:

        action = agent.get_action(state, env._get_date())
        actions.append(action)

        state, reward, done, _, dict = env.step(action)
        states.append(state)

    end_worth = state[0] + np.sum( np.array(state[1:agent.feature_len + 1]) *np.array(state[agent.feature_len+1:agent.feature_len*2 + 1]))
    data['final_worth'] = end_worth
    data['trades'] = env.trades
    asset_memory = np.array(env.asset_memory)

    rewards = asset_memory[1:] - asset_memory[:-1]
    data['rewards'] = rewards

    assert data['final_worth'] - data['initial_worth'] == np.sum(rewards), f"Net profit {data['final_worth'] - data['initial_worth']} does not match sum of rewards {np.sum(rewards)}"

    return data

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
    all_tickers = {'sp100':all_tickers['sp100']}
    train_start, train_end, test_end = '2014-01-01', '2024-01-01', '2026-01-01'

    train_sets, test_sets, all_tickers = get_data(train_start, train_end, test_end, all_tickers, indicators=indicators)

    n_trials = 1
    # agents = ['ddpg','ppo','td3']
    agents = ['ppo','td3']
    agents = ['td3','mb_td3']
    agent_kwargs = {
        'ppo':{
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
            },
        'td3':{"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
        'mb_td3':{"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
        }
    all_trials = {}

    # finrl_baseline = ['ddpg','ppo','td3']

    print(f'Running experiments for agents {agents} on datasets {all_tickers.keys()} with indicators {indicators} \n')

    values = {}
    for trial in range(1, n_trials+1):
        results = {}
        all_trials[trial] = results
        for agent in agents:
            results[agent] = {}
            for dataset in all_tickers.keys():

                tickers = all_tickers[dataset]
                df_train = train_sets[dataset]
                df_test = test_sets[dataset]


                trained_agent = train_agent(agent, df_train, model_kwargs=agent_kwargs[agent], indicators = indicators, dataset_name=dataset)

                test_values, actions = test_agent(trained_agent, df_test, indicatos=indicators, name = agent, dataset_name = dataset )

            values = {agent:test_values}

    plot_values(values)
                

        