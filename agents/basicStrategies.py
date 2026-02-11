
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

        print(y_label.shape)

        dec_inp = torch.zeros((self.pred_len, self.feature_size)).float()
        dec_inp = torch.cat([y_label, dec_inp], dim = 0).float().to(device)

        x = x.reshape(1, self.seq_len, self.feature_size)
        dec_inp = dec_inp.reshape(1, self.label_len + self.pred_len, self.feature_size)
        x_mark = x_mark.reshape(1, self.seq_len, x_mark.shape[-1])
        y_mark = y_mark.reshape(1, self.label_len + self.pred_len, y_mark.shape[-1])

        return x, dec_inp, x_mark, y_mark


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
    def __init__(self, args, predictor_model, scaler, buffer = None):
        '''Super class for predictor only strategies as agents for FinRL stocktradin_env'''
        self.scaler = scaler
        self.model = predictor_model

        # Every predictior will require a buffer to get a number of states
        seq_len, features = self.model.input_shape[0], self.model.input_shape[1]
        self.args = args
        self.seq_len = seq_len
        self.features = features
        print('Model shapes:', seq_len, features)

        self.buffer = buffer

        if self.buffer is None:
            self.buffer = Buffer(max_size=1000, seq_len=seq_len, feature_size = features)
            

    def state_to_input(self, state):
        '''Turns the state to a valid input to the model'''

        # valid for single asset
        close_data = state[1]

        self.buffer.add(close_data)

        if self.buffer.get_size() >= self.seq_len:
            input = self.buffer.get_last()
        
            x = self.scaler.transform(input)      
            x = torch.from_numpy(x.reshape(1, x.shape[0], x.shape[1])).float()

            return x
        else:
            return None

    def prediciton_to_action(self, prediction, state):
        '''Implements strategy'''
        return self.strategy_func(prediction, state)
    
    def get_action(self, state):

        input = self.state_to_input(state)

        # If we cannot get a prediction, return an action that does nothing
        if input == None:
            return np.zeros(self.features)

        with torch.no_grad():
            prediction = self.model(input)
            
        prediction = prediction.reshape(self.args.pred_len, self.features)
        prediction = self.scaler.inverse_transform(prediction)

        action = self.prediciton_to_action(prediction, state)
        print('action:',action)
        return action
    
    def strategy_func(self, prediciton, state):
        assert False, 'You are using the superclass, please implement a sub class that implements self.strategy_func'
        return None

class PredictorStrategyAutoformer:
    def __init__(self, args, predictor_model, scaler):

        self.scaler = scaler
        self.model = predictor_model

        self.seq_len = args.seq_len
        self.feature_len = args.enc_in
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.args = args

        self.buffer = Autoformer_Buffer(max_size=1000, args = args)

    def state_to_input(self, state, date):

        close_data = state[1].reshape(1, self.feature_len)
        close_data = self.scaler.transform(close_data)

        self.buffer.add(close_data.reshape(self.feature_len), date)

        if self.buffer.get_size() >= self.seq_len:
            x, y, x_mark, y_mark = self.buffer.get_last()

            x = x.float()
            y = y.float()
            x_mark = x_mark.float()
            y_mark = y_mark.float()

            return x, y, x_mark, y_mark
        else:
            return None, None, None, None

    def prediciton_to_action(self, prediction, state):
        '''Implements strategy'''
        return self.strategy_func(prediction, state)
    
    def get_action(self, state, date):
        x, y, x_mark, y_mark = self.state_to_input(state, date)

        if x is None:
            return np.zeros(self.feature_len)
        
        with torch.no_grad():
        
            prediction, _ = self.model._predict(x, y, x_mark, y_mark)
        
        prediction = self.scaler.inverse_transform(prediction.reshape(self.pred_len, self.feature_len))

        prediction = prediction[0]

        action = self.prediciton_to_action(prediction, state)

        return action
    
    def strategy_func(self, prediciton, state):
        assert False, 'You are using the superclass, please implement a sub class that implements self.strategy_func'
        return None
    

class BasicStrategy_auto(PredictorStrategyAutoformer):
    def __init__(self, args, predictor_model, scaler):
        super().__init__(args, predictor_model, scaler)

    def strategy_func(self, prediciton, state):

        price = state[1]

        price_change = prediciton - price

        action = np.zeros(self.feature_len)

        action[price_change > 0] = 1  # buy
        action[price_change < 0] = -1  # sell

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

class basicStrategy:
    def __init__(self, args, predictor_model, scaler):
        self.args = args
        self.scale = args.scale

        self.model = predictor_model
        self.scaler = predictor_model.scaler

        self.action_shape = args.feature_size

    def get_action(self, input, state):

        # is the state scaled?

        x = self.scaler.transform(input)
        x = torch.from_numpy(x.reshape(1, x.shape[0], x.shape[1])).float()
        print(f'x: {x.shape}')

        with torch.no_grad():
            prediciton = self.model(x)
        prediciton = prediciton.numpy()[0]

        prediciton = self.scaler.inverse_transform(prediciton)[0]

        assert prediciton.shape == state[:-1, 0].shape

        price_change = prediciton - state[:-1, 0]

        action = np.zeros(self.action_shape)

        action[price_change > 0] = 1  # buy
        action[price_change < 0] = -1  # sell
        return action

    # def predict():
    
class buyAndHoldStrategy:
    def __init__(self, args):
        self.args = args
        self.scale = args.scale

        self.action_shape = args.features

    def get_action(self, state):

        action = np.ones(self.action_shape)  # buy and hold

        return action
    
class calculatedStrategy:
    def __init__(self, args, predictor_model):
        self.args = args
        self.scaler = predictor_model.scaler

        self.model = predictor_model
        self.action_shape = args.features

    def get_action(self, state):

        action = np.zeros(self.action_shape)  # hold

        return action
    
class naiveProfitMaximizingStrategy:
    def __init__(self, args, predictor_model):
        self.args = args
        self.scaler = predictor_model.scaler

        self.model = predictor_model
        self.action_shape = args.features

    def get_action(self, state):

        action = np.ones(self.action_shape)*-1  # hold

        prediciton = self.model.predict(state)

        if self.scale:
            # inverse scale the prediction
            prediciton = self.scaler.inverse_transform(prediciton)

        price_change = prediciton - state[-1:, 0]

        highest = np.argmax(price_change)

        action[highest] = 1  # buy the asset with the highest predicted price increase with all available funds

        return action
    

class rankedStrategy:
    def __init__(self, args, predictor_model, top_k=3):
        self.args = args
        self.scaler = predictor_model.scaler

        self.model = predictor_model
        self.action_shape = args.features
        self.top_k = top_k

    def get_action(self, state, unscaled_data = None):

        action = np.zeros(self.action_shape)  # hold
        prices = state[-1:, 0]
        x = self.scaler.transform(prices)
        prediciton = self.model.predict(x)

        t = -1

        prediciton = self.scaler.inverse_transform(prediciton)

        price_change = prediciton - state[-1:, 0]

        ranked_indices = np.argsort(price_change)[-self.top_k:]

        action[ranked_indices] = 1  # buy the top_k assets with the highest predicted price increase
        action[price_change < t] = -1  # sell assets with predicted price decrease

        return action 
    
class adjustedRankedStrategy:
    def __init__(self, args, predictor_model, top_k=3):
        self.args = args
        self.scaler = predictor_model.scaler

        self.model = predictor_model
        self.action_shape = args.features
        self.top_k = top_k

    def get_action(self, state):

        action = np.zeros(self.action_shape)  # hold

        prediciton = self.model.predict(state)

        if self.scale:
            # inverse scale the prediction
            prediciton = self.scaler.inverse_transform(prediciton)
            state = self.scaler.inverse_transform(state)

        price_change = prediciton - state[-1:, 0]

        ranked_indices = np.argsort(price_change)[-self.top_k:]

        action[ranked_indices] = 1  # buy the top_k assets with the highest predicted price increase with all available funds
        action[price_change < 0] = -1  # sell assets with predicted price decrease

        # adjust actions based on confidence (magnitude of predicted price change)
        confidence_threshold = np.percentile(np.abs(price_change), 75)  # only act on predictions in the top 25% of confidence
        action[np.abs(price_change) < confidence_threshold] = 0  # hold if confidence is low

        return action