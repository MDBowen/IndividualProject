import argparse
import random as rnd
import os
import warnings

import torch 

warnings.filterwarnings("ignore")

from data.tickers import all_tickers 
from data.test_yahoo_downloader import TestYahooDownloader

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from agents.basicStrategies import BuyAndHold, PredictionSignStrategy, TopKStrategy
from agents.modelbased_TD3.mode_based_TD3 import ModelBasedTD3
from agents.modelbased_PPO.model_based_PPO import ModelBasedPPO

from utils.custom_callback import CustomCallback
from utils.evaluation import evaluate_model

from results.plot_results import create_performance_report

from enviroments.test_env_stocktrading import StockTradingEnv 

from models.train_autoformer import train_autoformer_from_finrl

predictor_agents = ['dense_predictor', 'autoformer_predictor','autoformer_topK']
basic_agents = ['buy_and_hold']
model_based_agents = ['dense_td3', 'autoformer_td3','autoformer_ppo']
model_free_agents = ['ddpg','ppo','td3']

def get_data(train_start, train_end, val_end, test_end, tickers, indicators = None, data_path = 'data/datasets'):

    downloader = TestYahooDownloader(train_start, test_end, tickers)
    df = downloader.fetch_data(cache_dir='data/datasets')
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

def get_dynamics_model(model_kwargs):
    from models.get_model import get_model
    print(f'Getting dynamics model {model_kwargs.get("model_name")} with kwargs {model_kwargs}')
    return get_model(**model_kwargs)        

def sample_tickers(train_set, val_set, test_set, tickers, assets_per_ep):

    if assets_per_ep is None or assets_per_ep >= len(tickers):
        return train_set, val_set, test_set, tickers
    
    tics = rnd.sample(list(tickers), assets_per_ep)
    
    train = train_set[train_set.tic.isin(tics)]
    val = val_set[val_set.tic.isin(tics)]
    test = test_set[test_set.tic.isin(tics)]
    
    return train, val, test, tics

def train_dynamics_model(model, train_data, val = None):
    '''
    Train the dynamics model on the given training data.
    model: Exp_Main instance from get_model (Autoformer / Transformer / Dense)
    train_data: pd.DataFrame in FinRL long-format (columns: date, tic, close, ...)
    val: pd.DataFrame with same format as train_data, used for validation during training
    '''
    train_autoformer_from_finrl(model, train_data, val_finrl_df=val)
    return model

def train_agentic_model(model, env, timesteps, eval_callback = None):
    '''
    Train the agentic model on the given environment.
    model: RL model (e.g. TD3) with a .learn() method
    env: Stable Baselines3 environment
    val: pd.DataFrame with same format as train_data, used for validation during training
    '''
    model.learn(total_timesteps=timesteps, log_interval=100, callback=eval_callback)
    return model


def train_model_free_agent(agent_name, agent_class, timesteps, train, val, env_kwargs, agent_kwargs, callback_kwargs=None):
    # Implementation for training a model-free agent
    
    tickers = train.tic.unique()

    stock_dimension = len(tickers)
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

    _env = StockTradingEnv(df = val, **env_kwargs)
    val_env, _ = _env.get_sb_env()

    env = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = env.get_sb_env()

    model = DRLAgent(env_train).get_model(agent_name, model_kwargs = agent_kwargs)

    cb = callback_kwargs or {}
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=cb.get('max_no_improvement_evals', 10),
        min_evals=cb.get('min_evals', 5),
        verbose=cb.get('verbose', 1),
    )
    callback = EvalCallback(
        val_env,
        callback_after_eval=stop_cb,
        best_model_save_path=f'./logs/{agent_name}/',
        log_path=f'./logs/{agent_name}/',
        n_eval_episodes=cb.get('n_eval_episodes', 5),
        eval_freq=cb.get('eval_freq', 50),
        deterministic=True,
        render=False,
        verbose=cb.get('verbose', 1),
    )

    model = train_agentic_model(model, env_train, timesteps=timesteps, eval_callback=callback)

    best_path = f'./logs/{agent_name}/best_model.zip'
    if os.path.exists(best_path):
        model.set_parameters(best_path)
        print(f"[{agent_name}] Restored best model from {best_path}")

    return model
     
def train_model_based_agent(agent_name, agent_class, timesteps, train, val, env_kwargs, agent_kwargs, callback_kwargs=None):
    # Implementation for training a model-based agent

    def temporal_split(df, date_col, ratio):
        split_idx = int(len(df) * ratio)
        return df.iloc[:split_idx], df.iloc[split_idx:]
    
    pred_train, agent_train = temporal_split(train, 'date', 0.5)

    tickers = train.tic.unique()

    stock_dimension = len(tickers)
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

    env = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = env.get_sb_env()
    agent_kwargs['_dynamics_kwargs']['feature_dim'] = len(tickers)

    dynamics_model = get_dynamics_model(agent_kwargs['_dynamics_kwargs'])

    dynamics_model = train_dynamics_model(dynamics_model, pred_train, val=agent_train)

    model = agent_class(
            
            env=env_train,
            dynamics_model=dynamics_model,
            tensorboard_log=None,
            verbose=1,
            policy_kwargs=None,
            seed=None,
            **agent_kwargs,
        )
    _env = StockTradingEnv(df = val, **env_kwargs)
    val_env, _ = _env.get_sb_env()

    cb = callback_kwargs or {}
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=cb.get('max_no_improvement_evals', 10),
        min_evals=cb.get('min_evals', 5),
        verbose=cb.get('verbose', 1),
    )
    callback = CustomCallback(
        val_env,
        callback_after_eval=stop_cb,
        best_model_save_path=f'./logs/{agent_name}/',
        log_path=f'./logs/{agent_name}/',
        n_eval_episodes=cb.get('n_eval_episodes', 5),
        eval_freq=cb.get('eval_freq', 50),
        deterministic=True,
        render=False,
        verbose=cb.get('verbose', 1),
    )
    model = train_agentic_model(model, env_train, timesteps=timesteps, eval_callback=callback)

    best_path = f'./logs/{agent_name}/best_model.zip'
    if os.path.exists(best_path):
        model.set_parameters(best_path)
        print(f"[{agent_name}] Restored best model from {best_path}")

    return model

def train_predictor_agent(agent_name, agent_class, timesteps, train, val, env_kwargs, agent_kwargs):
    # Implementation for training a predictor agent
    
    tickers = train.tic.unique()

    stock_dimension = len(tickers)
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

    env = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = env.get_sb_env()

    dynamics_model = get_dynamics_model(agent_kwargs['_dynamics_kwargs'])
    dynamics_model = train_dynamics_model(dynamics_model, train, val=val)

    return agent_class(dynamics_model, env_kwargs['hmax'], stock_dimension)

def get_basic_agent(agent_name, agent_class, env_kwargs, agent_kwargs):
    # Implementation for getting a basic agent (e.g. Buy and Hold)
    return agent_class(env_kwargs['stock_dim'], device = 'cuda' if torch.cuda.is_available() else 'cpu')

def test_agent(agent, test_set, uses_predictor=False, env_kwargs=None, tickers=None, eval_episodes = 3):

    env = StockTradingEnv(df=test_set, **env_kwargs)
    env_test, _ = env.get_sb_env()

    rewards, actions, observations, predictions, trues = evaluate_model(
        agent,
        env_test,
        n_eval_episodes=eval_episodes,
        uses_predictor=uses_predictor,
    )

    return {
        'rewards': rewards,
        'actions': actions,
        'states': observations,
        'predictions': predictions,
        'actuals': trues,
        'stock_dim': env_kwargs['stock_dim'],
        'tickers': list(tickers) if tickers is not None else None,
    }

def run_experiments(number_of_trials, agents, dataset, timesteps, assets_per_ep, env_kwargs=None, agents_kwargs=None, indicators=[], eval_episodes=3, callback_kwargs=None):

    results = {}

    for data_name in dataset.keys():
        train_set, val_set, test_set, tickers = get_data('2000-01-01', '2023-01-01', '2024-01-01', '2026-01-01', dataset[data_name], indicators)

        #gets the whole dataframe of a 'market' (e.g. sp100) and the unique tickers in that dataframe

        'gets a subset of n assets from a market'
        # print('Number of training time steps:', timesteps_per_env*env_episodes)

        for trials in range(1, number_of_trials+1):

            results[trials] = {}
            results[trials][data_name] = {}

            train, val, test, tic = sample_tickers(train_set, val_set, test_set, tickers, assets_per_ep)
            
            for agent_name, agent_class in agents.items():

                results[trials][data_name][agent_name] = {}

                print(f'\n Running agent {agent_name} on dataset {data_name} with {tic} having {train["date"].nunique()} trading days \n')

                agent_kwargs = agents_kwargs[agent_name]

                if agent_name in model_free_agents:
                    agent = train_model_free_agent(agent_name,
                                                   agent_class,
                                                   timesteps,
                                                   train,
                                                   val,
                                                   env_kwargs,
                                                   agent_kwargs,
                                                   callback_kwargs=callback_kwargs)

                elif agent_name in model_based_agents:
                    agent = train_model_based_agent(agent_name,
                                                    agent_class,
                                                    timesteps,
                                                    train,
                                                    val,
                                                    env_kwargs,
                                                    agent_kwargs,
                                                    callback_kwargs=callback_kwargs)

                elif agent_name in predictor_agents:
                    agent = train_predictor_agent(agent_name, 
                                                  agent_class, 
                                                  timesteps, 
                                                  train, 
                                                  val, 
                                                  env_kwargs, 
                                                  agent_kwargs)
                elif agent_name in basic_agents:
                    agent = get_basic_agent(agent_name, agent_class, env_kwargs, agent_kwargs)
                else:
                    raise ValueError(f"Can't find agent {agent_name}")
                    
                results[trials][data_name][agent_name] = test_agent(
                    agent,
                    test,
                    uses_predictor=agent_name not in model_free_agents + basic_agents,
                    env_kwargs=env_kwargs,
                    tickers=tic,
                    eval_episodes=eval_episodes
                )
                
    create_performance_report(results, dataset)
    

if __name__ == '__main__':
    defaults_steps = 1000
    parser = argparse.ArgumentParser(description='Run RL trading experiments')
    parser.add_argument(
        '--n_trials',
        type=int,
        default=3,
        help='Number of independent trials per agent/dataset combination (default: 1)',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=defaults_steps,
        help='Total environment timesteps for RL agent training (default: 300)',
    )
    parser.add_argument(
        '--assets_per_ep',
        type=int,
        default=10,
        help='Assets randomly sampled from the market universe each trial (default: 10)',
    )
    parser.add_argument(
        '--model_epochs',
        type=int,
        default=10,
        help='Number of training epochs for dynamics models',
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='Verbosity level for eval callbacks (default: 1)',
    )
    parser.add_argument(
        '--n_eval_episodes',
        type=int,
        default=3,
        help='Number of episodes per callback evaluation (default: 5)',
    )
    parser.add_argument(
        '--eval_num',
        type=int,
        default=100,
        help='Number of evals (default: 100)',
    )
    parser.add_argument(
        '--max_no_improvement_evals',
        type=int,
        default=10,
        help='Stop training after this many evals with no new best (default: 10)',
    )
    parser.add_argument(
        '--min_evals',
        type=int,
        default=5,
        help='Minimum evaluations before early stopping can trigger (default: 5)',
    )
    parser.add_argument(
        '--only_sp100',
        type=bool,
        default=False,
        help = 'For testing only run sp100'
    )
    args = parser.parse_args()

    n_trials      = args.n_trials
    timesteps     = args.timesteps
    assets_per_ep = args.assets_per_ep

    callback_kwargs = {
        'verbose':                  args.verbose,
        'n_eval_episodes':          args.n_eval_episodes,
        'eval_freq':                timesteps // args.eval_num,
        'max_no_improvement_evals': args.max_no_improvement_evals,
        'min_evals':                args.min_evals,
    }

    print(
        f'Config — n_trials={n_trials}  timesteps={timesteps}  assets_per_ep={assets_per_ep}  '
        f'eval_freq={timesteps // args.eval_num,}  n_eval_episodes={args.n_eval_episodes}  '
        f'max_no_improvement_evals={args.max_no_improvement_evals}  min_evals={args.min_evals}'
    )

    indicators = [
        'macd',
        'boll_ub',
        'boll_lb',
        'rsi_30',
        'cci_30',
        'dx_30',
        'close_30_sma',
        'close_60_sma'
    ]

    agents = {
              'buy_and_hold': BuyAndHold, 
              'dense_td3': ModelBasedTD3,
              'td3': None, 'ppo': None,
              'dense_predictor': PredictionSignStrategy, 
              'autoformer_predictor': PredictionSignStrategy,
              'autoformer_td3': ModelBasedTD3,
              'autoformer_ppo':ModelBasedPPO }

    agents = {
              'buy_and_hold': BuyAndHold, 
              'dense_td3': ModelBasedTD3,
              'td3': None, 'ppo': None,
              'dense_predictor': PredictionSignStrategy, 
              'autoformer_predictor': PredictionSignStrategy,
            #   'autoformer_td3': ModelBasedTD3,
            # 'autoformer_ppo':ModelBasedPPO,
            'autoformer_topK': TopKStrategy }
    # agents = {
    #             'buy_and_hold': BuyAndHold, 
    #           'dense_predictor': PredictionSignStrategy, 
    #           'td3': None,

    # }

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
        "reward_scaling": 1e-4,
    }

    agents_kwargs = {
        'ppo': {
            "n_steps": min(2048, timesteps),
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": min(128, timesteps),
        },
        'td3': {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001, 'learning_starts' : 150},
        'autoformer_td3': {
            "batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001, 'learning_starts' : 150,
            "_dynamics_kwargs": {'model_name': 'Autoformer', "feature_dim": assets_per_ep, "epochs": 1},
        },
        'dense_td3': {
            "batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001, 'learning_starts' : 150,
            "_dynamics_kwargs": {"model_name": 'Dense', "feature_dim": assets_per_ep},
        },
        'dense_predictor': {"_dynamics_kwargs": {"model_name": 'Dense', "feature_dim": assets_per_ep}},
        'autoformer_predictor': {"_dynamics_kwargs": {'model_name': 'Autoformer', "feature_dim": assets_per_ep, "epochs": 1}},
        'buy_and_hold': {"_dynamics_kwargs": {"model_name": 'Dense', "feature_dim": assets_per_ep}},
        'autoformer_ppo': {
            "n_steps": min(2048, timesteps),
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": min(128, timesteps),
            "_dynamics_kwargs": {'model_name': 'Autoformer', "feature_dim": assets_per_ep, "epochs": 1}
        },
        'autoformer_topK': {"_dynamics_kwargs": {'model_name': 'Autoformer', "feature_dim": assets_per_ep, "epochs": 1}}
    }

    if args.only_sp100:
        all_tickers = {'sp100': all_tickers['sp100']}

    run_experiments(
        n_trials,
        agents,
        all_tickers,
        timesteps=timesteps,
        assets_per_ep=assets_per_ep,
        env_kwargs=env_kwargs,
        agents_kwargs=agents_kwargs,
        indicators=indicators,
        callback_kwargs=callback_kwargs,
    )

    print('All done!')

