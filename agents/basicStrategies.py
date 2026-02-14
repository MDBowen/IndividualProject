
import numpy as np
import torch
from collections import deque
import pandas as pd

from utils.timefeatures import time_features

'''For implementing predictive models as agents for the DRL_agent class in 
finrl.agents.stablebaselines3.models'''
''' for DRL_prediciton method each require a .predict method that takes test_obs (a state observation) 
and returns an action (for sure) and and maybe a state?  '''
''' For the StockTradinEnv class in finrl.meta.env_stock_trading.env_stocktrading the state will be of the
form tuple( [balance]  )'''


class Buy_And_Hold:
    def __init__(self, args, hmax):
        self.feature_size = args.enc_in 
        self.index = 0
        self.hmax = hmax
    def get_action(self, state, date):
        prices = state[1:self.feature_size+1]
        balance = state[0]
        self.index += 1
        if self.index == 1:
            return np.ones(self.feature_size, dtype = np.float32) * max(min((balance/(np.sum(prices)*self.hmax)), 1), 0), None
        else:
            return np.zeros(self.feature_size, dtype = np.float32), None

class Autoformer_Buffer:
    def __init__(self,  max_size, args):
        self.prices = deque(maxlen=max_size)
        self.dates = deque(maxlen=max_size)
        self.seq_len = args.seq_len
        self.feature_size = args.enc_in
        self.pred_len = args.pred_len
        self.max_size = max_size
        self.label_len = args.label_len
        self.freq = args.freq
        
    def add(self, x, date):

        self.prices.append(x)
        self.dates.append(date)

    def get_all(self):
        return list(self.prices), list(self.dates)
    
    def get_size(self):
        return len(list(self.prices))

    def get_last(self, device = 'cpu'):

        x = list(self.prices)[-self.seq_len:]
        y_label = list(self.prices)[-self.label_len:]


        x_dates = list(self.dates)[-self.seq_len:]
        y_dates = list(self.dates)[-self.label_len:]

        pred_dates = pd.date_range(y_dates[-1], periods=self.pred_len + 1, freq=self.freq)

        y_stamp = y_dates + list(pred_dates)[1:]
        
        x_mark = time_features(pd.to_datetime(x_dates), freq=self.freq).transpose(1, 0)
        y_mark = time_features(pd.to_datetime(y_stamp), freq=self.freq).transpose(1, 0)

        x = torch.Tensor(x)
        y_label = torch.Tensor(y_label)
        x_mark = torch.Tensor(x_mark)
        y_mark = torch.Tensor(y_mark)

        dec_inp = torch.zeros((self.pred_len, self.feature_size)).float()
        dec_inp = torch.cat([y_label, dec_inp], dim = 0).float().to(device)

        x = x.reshape(self.seq_len, self.feature_size)
        dec_inp = dec_inp.reshape(self.label_len + self.pred_len, self.feature_size)
        x_mark = x_mark.reshape(1, self.seq_len, x_mark.shape[-1])
        y_mark = y_mark.reshape(1, self.label_len + self.pred_len, y_mark.shape[-1])

        return x, x_mark, y_mark


class Buffer:
    def __init__(self,  max_size=1000, seq_len=96, feature_size=98):
        self.states = deque(maxlen=max_size)
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.max_size = max_size
    
    def add(self, item):
        self.states.append(item)
    
    def get_all(self):
        return list(self.states)
    
    def get_size(self):
        return len(list(self.states))

    def get_last(self):
        states = np.array(list(self.states))
        # print(f'Buffer states shape before slicing: {states.shape}')
        prices = states[-self.seq_len:]
        # print(f'Buffer prices shape: {prices.shape}')
        return prices.reshape(self.seq_len, self.feature_size)

class PredictorStrategy: 
    def __init__(self, args, predictor_model, scaler, hmax, buffer = None):
        '''Super class for predictor only strategies as agents for FinRL stocktradin_env'''
        self.scaler = scaler
        self.model = predictor_model

        # Every predictior will require a buffer to get a number of states
        seq_len, features = self.model.input_shape[0], self.model.input_shape[1]
        self.args = args
        self.seq_len = seq_len
        self.features = args.enc_in
        print('Model shapes:', seq_len, features)

        self.buffer = buffer

        if self.buffer is None:
            self.buffer = Buffer(max_size=1000, seq_len=seq_len, feature_size = features)
            

    def state_to_input(self, state):
        '''Turns the state to a valid input to the model'''

        # valid for single asset
        close_data = state[1:self.features + 1]
        holdings = state[self.features+1, self.features*2 +1]
        balance = state[0]


        self.buffer.add(close_data)

        if self.buffer.get_size() >= self.seq_len:
            input = self.buffer.get_last()
        
            x = self.scaler.transform(input)      
            x = torch.from_numpy(x.reshape(1, x.shape[0], x.shape[1])).float()

            return x
        else:
            return None

    def prediciton_to_action(self, prediction, price):
        '''Implements strategy'''
        return self.strategy_func(prediction, price)
    
    def get_action(self, state, date):

        input = self.state_to_input(state)

        # If we cannot get a prediction, return an action that does nothing
        if input == None:
            return np.zeros(self.features)
        self.model.model.eval()

        with torch.no_grad():
            prediction = self.model(input)
            
        prediction = prediction.reshape(self.args.pred_len, self.features)
        prediction = self.scaler.inverse_transform(prediction)
        price = state[1:self.features + 1]
        action = self.prediciton_to_action(prediction, price)

        return action
    
    def strategy_func(self, prediciton, state):
        assert False, 'You are using the superclass, please implement a sub class that implements self.strategy_func'
        return None

class PredictorStrategyAutoformer:
    def __init__(self, args, predictor_model, scaler, hmax):

        self.scaler = scaler
        self.model = predictor_model
        self.hmax = hmax
        self.seq_len = args.seq_len
        self.feature_len = args.enc_in
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.args = args

        self.buffer = Autoformer_Buffer(max_size=1000, args = args)

    def state_to_input(self, state, date):

        close_data = np.array(state[1:self.feature_len+1]).reshape(self.feature_len)

        self.buffer.add(close_data.reshape(self.feature_len), date)

        if self.buffer.get_size() >= self.seq_len:
            x, x_mark, y_mark = self.buffer.get_last()

            x_mark = x_mark.float()
            y_mark = y_mark.float()

            x = torch.tensor(self.scaler.transform(x).reshape(1, self.seq_len, self.feature_len)).float()

            y_label = x[:, -self.label_len:, :]

            dec_inp = torch.zeros((1, self.pred_len, self.feature_len))
            dec_inp = torch.cat([y_label, dec_inp], dim = 1).float()

            return x, dec_inp, x_mark, y_mark
        else:
            return None, None, None, None

    def prediciton_to_action(self, prediction, price, holdings, balance):
        '''Implements strategy'''
        assert prediction.shape == price.shape, f'Price {price.shape} and price prediction {prediction.shape} need to be same shape'
        return self.strategy_func(prediction, price, holdings, balance)
    
    def get_action(self, state, date):
        x, y, x_mark, y_mark = self.state_to_input(state, date)

        if x is None:
            return np.zeros(self.feature_len), np.zeros(self.feature_len)
        
        self.model.model.eval()
        
        with torch.no_grad():

            prediction, _ = self.model._predict(x, y, x_mark, y_mark)

        prediction = self.scaler.inverse_transform(prediction.reshape(self.pred_len, self.feature_len))
        prediction = prediction[0]
        price = np.array(state[1:self.feature_len+1]).reshape(self.feature_len)

        holdings = state[self.feature_len+1:self.feature_len*2+1]
        balance = state[0]

        action = self.prediciton_to_action(prediction, price, holdings, balance)

        assert action.shape == prediction.shape, 'Action and prediction not the same shape'

        return np.array(action.clip(min=-1, max=1).astype(np.float32), dtype=np.float32), prediction

    def strategy_func(self, prediciton, prices):
        assert False, 'You are using the superclass, please implement a sub class that implements self.strategy_func'
        return None
    

class BasicStrategy_auto(PredictorStrategyAutoformer):
    def __init__(self, args, predictor_model, scaler, hmax):
        super().__init__(args, predictor_model, scaler, hmax)

    def strategy_func(self, prediciton, price, holdings, balance):

        price_change = prediciton - price

        action = np.zeros(self.feature_len)

        action[price_change > 0] = 1.0 # buy
          # sell
        action = action*np.dot(action, price)*self.hmax/balance

        action[price_change < 0] = -1.0

        return action

class BasicStrategy(PredictorStrategy):
    def __init__(self, args, predictor_model, scaler):
        super().__init__(args, predictor_model, scaler)

    def strategy_func(self, prediciton, state):
        
        price = state[1]
        prediciton = prediciton[0][0]

        price_change = prediciton - price

        action = np.zeros(self.features)

        action[price_change > 0] = 1  # buy
        action[price_change < 0] = -1  # sell

        return action
