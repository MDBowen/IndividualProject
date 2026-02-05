
from sklearn.preprocessing import StandardScaler

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import argparse

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, args, size=None, flag = 'train'):
        '''Initialize the trading environment.'''
        

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.seq_len = args.sequence_len # lenght of the total input sequence
        self.label_len = args.label_len # lenght of steps in the past we use to predict
        self.pred_len = args.pred_len # lenght of steps in the future we predict

        self.scale = args.scale
        self.freq = args.freq

        self.data, self.date_column = self.__read_data__(args)

        self.total_timesteps = self.data.shape[0]
        features = self.data.shape[1]

        assert self.total_timesteps > self.seq_len, "Data length must be larger than sequence length."
        
        self.features = features
        self.action_space = spaces.Box(low=-1, high=1, shape=(features,), dtype=np.float32)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(features + 1, 2), dtype=np.float32) # ([prices,holdings], [balance, total_invested])

        self.initial_balance = args.initial_balance if hasattr(args, 'initial_balance') else 10000  # Initial cash balance
        self.lower_barrier = 0.01 # defines how much increasing allows full allocation of balance
        self.reset()

    def __read_data__(self, args):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(args.root_path,
                                          args.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test


        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type] # start depending on train, vali, test 
        border2 = border2s[self.set_type] # end depending on train, vali, test

        cols_data = df_raw.columns[1:]

        dates = df_raw['date'][2:]

        df_data = df_raw[cols_data]
        df_data = df_data.to_numpy()
        df_data = df_data[2:]

        col_to_delete = []        

        for i in range(df_data.shape[1]):
            col = df_data[:, i]
            if np.isnan(col).any():
                mean_val = np.nanmean(col)
                col[np.isnan(col)] = mean_val
                df_data[:, i] = col
                print(f'found nan in column {i}, which belongs to {cols_data[i]}, column deleted')

                col_to_delete.append(cols_data[i])

        cols_data = list(set(cols_data) - set(col_to_delete))

        df_data = df_raw[cols_data].to_numpy()[2:]

        if np.isnan(df_data).any():
            print(f'nan in df_data: {np.isnan(df_data).any()} within read_data')

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.unscaled_data = df_data[border1:border2]

        if self.scale:    
            data = self.scaler.transform(df_data)
            print('StandardScaler applied')
        else:
            data = df_data


        dates = dates[border1:border2]
        return data[border1:border2], dates

    def _get_obs_scaled(self):
        '''Return the current observation, scaled if required.'''
        assert self.scale, "Scaling is not enabled for this environment."
        return self.unscaled_data[self.index].copy(), self.state.copy()
        
    def _get_obs(self, scaled = True):
        '''Return the current observation.'''

        return self.state.copy()
    
    def simulate_action(self, action, printout = False):
        '''Execute the given action to a state.'''
        prev_net = self.get_net_worth()
        action = action.astype(np.float16)
        state = self.state.astype(np.float16).copy()
        if printout:
            print('before action', state)
            
        #Set all positive actions to zero. Calculate deductive actions first, then calculate increase actions.
        deductive_actions = action.copy()
        deductive_actions[deductive_actions > 0] = 0

        deduct_by = (state[:-1, 1] * (deductive_actions)).astype(np.float16)
        deduct_by = np.round(deduct_by, decimals = 3)
        
        # deduct each holding by  it's current value times action negatives
        # the sum of this is added to the balance
       # assert any(x < 0 for x in deduct_by), f'Negative in deduct by: {deduct_by}'

        state[:-1, 1] += deduct_by
        state[-1, 0] -= np.sum(deduct_by)

        increase_actions = action.copy()
        increase_actions[increase_actions < 0] = 0

        # increase amount is it's percentage of current increases times balance (with a lower barrier)

        increase_by = ( state[-1, 0]*(increase_actions / max(np.sum(increase_actions), self.lower_barrier + 0.01)) ).astype(np.float16)
        
        increase_by = np.round(increase_by, decimals = 3)
        if np.sum(increase_by) > state[-1, 0]:
            print('increase by bigger than balance')
            # print(np.sum(increase_actions/ max(np.sum(increase_actions)), self.lower_barrier))
            # increase_by = np.zeros(state[:-1, 0].shape)

       # assert any(x < 0 for x in increase_by), f'negative in increase by: {increase_by}'

        # add increase amount to each holding
        # deduct increase amount sum from balance
        state[:-1, 1] += increase_by
        state[-1, 0] -= np.sum(increase_by)
        
        if printout:
            print('action',action)
            print('deducitons:',deductive_actions)
            print('increases:', increase_actions)
            print('increase by:', increase_by)
            print('deduct by :', deduct_by)
            print('new state', state)

        assert np.sum(state[:-1, 1]) + state[-1, 0] == prev_net, f"Net worth changed after action! new: {np.sum(state[:-1, 1]) + state[-1, 0]}, prev: {prev_net} "
        
        return state
    
    def get_net_worth(self):
        return np.sum(self.state[:-1, 1]) + self.state[-1, 0]

    def step(self, action):
        '''Take a step in the environment.'''

        action = action.astype(np.float16)
        self.state = self.state.astype(np.float16)

        printout = False if np.sum(action) != 0 else False

        self.state = self.simulate_action(action, printout=printout)

        prev_net = np.sum(self.state[:-1, 1]) + self.state[-1, 0]  # Previous net worth

        self.index += 1

        prev_prices = self.state[:-1, 0].copy()
        self.state[:-1, 0] = self.data[self.index]  # Update prices

        # print(f"prev_prices: {prev_prices}")
        # print(f"new prices: {self.state[:-1, 0]}")
        # print(f"holdings before update: {self.state[:-1, 1]}")

        self.state[:-1, 1] = (self.state[:-1, 1]* (self.state[:-1,0]/ (prev_prices + 1e-8))).astype(np.float16)  # Update holdings value

        reward = np.sum(self.state[:-1, 1]) + self.state[-1, 0] - prev_net  # Reward is the change in net worth

        done = self.index >=self.total_timesteps - 1  # Check if done
        observation = self.state.copy()
        info = None

        assert np.all(self.state >= 0), f"State contains negative values {self.state}"

        return observation, reward, done, info

    def reset(self):
        '''Reset the environment to the initial state.'''
        self.index = 0  # Current index in the data 
        self.state = np.zeros((self.features + 1, 2))  # Initialize state with shape (features + 1, 2)
        self.state[:-1, 0] = self.data[0]  # Set initial prices
        self.state[-1, 0] = self.initial_balance  # Set initial cash balance in the state

        

# if __name__ == "__main__":
    
#     args = dotdict()
#     args.seq_len = 96
#     args.label_len = 48
#     args.pred_len = 24

#     args.root_path = './data/sp100_combined_close.csv'
#     args.data_path = 'sp100_combined_close.csv'
#     args.scale = True
#     args.freq = 'd'
#     args.initial_balance = 10000

#     env =  TradingEnv(args)

#     obs = env._get_obs()
#     print(f"Initial Observation: {obs}")