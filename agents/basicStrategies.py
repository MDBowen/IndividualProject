
import numpy as np
import torch
from collections import deque


'''For implementing predictive models as agents for the DRL_agent class in 
finrl.agents.stablebaselines3.models'''
''' for DRL_prediciton method each require a .predict method that takes test_obs (a state observation) 
and returns an action (for sure) and and maybe a state?  '''
''' For the StockTradinEnv class in finrl.meta.env_stock_trading.env_stocktrading the state will be of the
form tuple( [balance]  )'''


class Buffer:
    def __init__(self,  max_size=1000, seq_len =96, feature_size=98):
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

class Predictor_Strategy: 
    def __init__(self, args, predictor_model, scaler):
        '''Super class for predictor only strategies as agents for FinRL stocktradin_env'''


        self.scaler = scaler
        self.model = predictor_model

        self.args = args 
        self.action_shape = args.feature_size

        # Every predictior will require a buffer to get a number of states
        seq_len, features = self.model.input_shape[0], self.model.input_shape[1]
        self.seq_len = seq_len
        self.features = features
        self.buffer = Buffer(max_size= 1000, seq_len=seq_len, feature_size = features)

    def state_to_input(self, state):
        '''Turns the state to a valid input to the model'''

        # valid for single asset
        close_data = state[1]

        x = self.scaler.transform(close_data)      
        x = torch.from_numpy(x.reshape(1, x.shape[0], x.shape[1])).float()

        self.buffer.add(x)

        if self.buffer.get_size() >= self.seq_len:
            input = self.buffer.get_last()
            return input
        else:
            return None

    def prediciton_to_action(self, prediction, strategy_func, state):
        '''Implements strategy'''
        return strategy_func(prediction, state)
    
    def get_action(self, state):

        input = self.state_to_input(state)

        if input == None:
            action = np.zeros(self.features)

        assert input==self.model.input_shape, f'Input shape {input} doesnt match expected shape {self.model.input_shape}'

        with torch.no_grad():
            prediction = self.model(input)[0]

        action = self.prediciton_to_action(prediction, state)

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