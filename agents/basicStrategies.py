
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
        self.feature_len = args.enc_in 
        self.index = 0
        self.hmax = hmax
    def get_action(self, state, date):
        prices = state[1:self.feature_len+1]
        balance = state[0]
        self.index += 1
        if self.index == 1:
            return np.ones(self.feature_len, dtype = np.float32) * max(min((balance/(np.sum(prices)*self.hmax)), 1), 0), None
        else:
            return np.zeros(self.feature_len, dtype = np.float32), None

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
            prediction = prediction.cpu()

        prediction = self.scaler.inverse_transform(prediction.reshape(self.pred_len, self.feature_len))
        prediction = prediction[0]

        assert prediction.shape == (self.feature_len,), f'Prediction shape {prediction.shape} is not correct'

        price = np.array(state[1:self.feature_len+1]).reshape(self.feature_len)
        holdings = state[self.feature_len+1:self.feature_len*2+1]
        balance = state[0]

        action = self.prediciton_to_action(prediction, price, holdings, balance)

        assert action.shape == prediction.shape, 'Action and prediction not the same shape'

        return np.array(action.clip(min=-1, max=1).astype(np.float32), dtype=np.float32), prediction

    def strategy_func(self, prediciton, prices):
        assert False, 'You are using the superclass, please implement a sub class that implements self.strategy_func'
        return None
    

class BasicStrategy(PredictorStrategyAutoformer):
    def __init__(self, args, predictor_model, scaler, hmax):
        super().__init__(args, predictor_model, scaler, hmax)

    def strategy_func(self, prediciton, price, holdings, balance):

        price_change = prediciton - price

        action = np.zeros(self.feature_len)

        action[price_change > 0] = 1.0
        action = action*np.dot(action, price)*self.hmax/balance

        action[price_change < 0] = -1.0

        return action
