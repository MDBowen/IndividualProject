
from IndividualProject.utils.tools import StandardScaler
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
    def __init__(self, args, size=None):
        '''Initialize the trading environment.'''
        
        root_path = args.root_path
        data_path = args.data_path

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        flag = 'test'

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]


        self.seq_len = args.seq_len # lenght of the total input sequence
        self.label_len = args.label_len # lenght of steps in the past we use to predict
        self.pred_len = args.pred_len # lenght of steps in the future we predict

        self.scale = args.scale
        self.freq = args.freq

        self.data, self.date_column = self.__read_data__()

        self.total_timesteps = self.data.shape[0]
        features = self.data.shape[1]

        assert timesteps > self.seq_len, "Data length must be larger than sequence length."
        
        self.features = features
        self.action_space = spaces.Box(low=-1, high=1, shape=(features,), dtype=np.float32)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(features + 1, 2), dtype=np.float32) # ([prices,holdings], [balance, total_invested])

        self.index = 0  # Current index in the data 
        self.initial_balance = args.initial_balance  # Initial cash balance
        self.state = np.zeros((features + 1, 2))  # Initialize state with shape (features + 1, 2)
        self.state[-1, 0] = self.initial_balance  # Set initial cash balance in the state

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        np_df = df_raw.to_numpy()

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type] # start depending on train, vali, test 
        border2 = border2s[self.set_type] # end depending on train, vali, test

        if self.select_columns is not None:
            df_raw = df_raw[['date'] + self.select_columns]
            # print(f'select columns: {self.select_columns}')

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

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data.to_numpy()

        dates = dates[border1:border2]
        return data[border1:border2], dates

    
    def _get_obs(self):
        '''Return the current observation.'''
        
        return self.state.copy()

    def step(self, action):
        '''Take a step in the environment.'''

        action = action.astype(np.float32)

    
        '''Set all positive actions to zero.'''
        deductive_actions = action.copy()
        deductive_actions[deductive_actions > 0] = 0

        prev_net = np.sum(self.state[:-1, 1]) + self.state[-1, 0]  # Previous net worth


        invested_amounts = np.sum(self.state[:-1,1])  # Total invested amounts

        self.state[:-1, 1] *= (1 + deductive_actions)  # Update holdings based on actions
        self.state[-1, 0] += invested_amounts - np.sum(self.state[:-1, 1])  # Update cash balance

        increase_actions = action.copy()
        increase_actions[increase_actions < 0] = 0

        self.state[:-1, 1] += increase_actions * self.state[-1, 0] / max(np.sum(increase_actions), self.features * 0.5) 
        # allocates the new balance according to the positive actions proportional to each other
        # we need a lower bound, such that we do not divide by zero, but also do not allways allocate all balance

        self.index += 1
        prev_prices = self.state[:-1, 0].copy()
        self.state[:-1, 0] = self.data[self.index]  # Update prices
        self.state[:-1, 1] = self.state[:-1, 1]* self.state[:-1,0]/ prev_prices  # Update holdings value

        reward = np.sum(self.state[:-1, 1]) + self.state[-1, 0] - prev_net  # Reward is the change in net worth

        done = self.index >=self.total_timesteps - 1  # Check if done
        observation = self.state.copy()
        info = None

        assert np.all(self.state >= 0), "State contains negative values"

        return observation, reward, done, info

    def reset(self):
        '''Reset the environment to the initial state.'''
        self.index = 0  # Current index in the data 
        self.state = np.zeros((self.features + 1, 2))  # Initialize state with shape (features + 1, 2)
        self.state[:-1, 0] = self.data[0]  # Set initial prices
        self.state[-1, 0] = self.initial_balance  # Set initial cash balance in the state
