
import numpy as np
import torch
from collections import deque
import pandas as pd
import torch as th

from utils.timefeatures import time_features
from utils.custom_buffers import Autoformer_Buffer
'''For implementing predictive models as agents for the DRL_agent class in 
finrl.agents.stablebaselines3.models'''
''' for DRL_prediciton method each require a .predict method that takes test_obs (a state observation) 
and returns an action (for sure) and and maybe a state?  '''
''' For the StockTradinEnv class in finrl.meta.env_stock_trading.env_stocktrading the state will be of the
form tuple( [balance]  )'''

class PredictorStrategy():
    def __init__(self, dynamics_model, hmax, feature_len, scaler = None, pred_idx = 0, device = 'cpu'):
        self.dynamics_model = dynamics_model
        self.scaler = scaler
        self.hmax = hmax
        self.feature_len = feature_len
        self.pred_idx = pred_idx
        self.device = device
        self.window_buffer = Autoformer_Buffer(max_size=100_000, 
                                               feature_size=feature_len, 
                                               seq_len=dynamics_model.args.seq_len, 
                                               pred_len=dynamics_model.args.pred_len, 
                                               label_len=dynamics_model.args.label_len, 
                                               freq = dynamics_model.args.freq)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        date: object | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        
        self.dynamics_model.model.eval()
        price = np.array(observation[:, 1:self.feature_len+1]).reshape(self.feature_len)
        with torch.no_grad():
            pred = self._get_prediction(price, date)

        if len(pred.shape) == 3:
            pred = pred[:, self.pred_idx, :]
        elif len(pred.shape) == 2:
            pred = pred[self.pred_idx]
        else:
            raise ValueError(f'Unexpected prediction shape: {pred.shape}')

        action = self.strategy_func(pred, price)

        return action, state
    
    def strategy_func(self, prediciton: np.ndarray, price: np.ndarray):
        raise NotImplementedError('You are using the superclass, please implement a sub class that implements self.strategy_func')

    def _get_prediction(self, x: np.ndarray | th.Tensor, date: str):
        ''''x needs to be the prices in obs
            date needs to be the date from the env
        '''
        assert date is not None, 'date needs to be passed in for the time features'
        x= x.squeeze()
        assert len(x.shape) == 1, f'x needs to be 1d array of prices, got {x.shape}'
        self.window_buffer.add(x, date) 
        x, y, x_mark, y_mark = self.window_buffer.get_last(device = self.device)
        pred, _ = self._dynamics_model_predict(x, y, x_mark, y_mark)
        return pred.detach()

    def _dynamics_model_predict(self, x, y, x_mark, y_mark, scale = True):
        reshape_and_scale = lambda transform, _x, shape: th.Tensor(transform(_x.reshape(shape[0] * shape[1], shape[2]).detach().numpy())).reshape(shape[0], shape[1], shape[2]).to(_x.device)

        if self.dynamics_model.args.scale and scale:

            batches = x.shape[0]
            window_x = x.shape[1]
            window_y = y.shape[1]
            feature_dim_x = x.shape[2]
            feature_dim_y = y.shape[2]

            x = reshape_and_scale(self.dynamics_model.scaler.transform, x, (batches, window_x, feature_dim_x))
            y = reshape_and_scale(self.dynamics_model.scaler.transform, y, (batches, window_y, feature_dim_y))

        pred, y = self.dynamics_model._predict(x, y, x_mark, y_mark)

        if self.dynamics_model.args.scale and scale:
            pred = reshape_and_scale(self.dynamics_model.scaler.inverse_transform, pred, pred.shape)

        return pred, y
    
class BuyAndHold(PredictorStrategy):
    def __init__(self, dynamics_model, hmax, feature_len, scaler = None, pred_idx = 0, device = 'cpu'):
        super().__init__(dynamics_model, hmax, feature_len, scaler = scaler, pred_idx = pred_idx, device = device)
        self.inx = 0 

    def strategy_func(self, prediciton, price):
        if self.inx == 0:
            action = np.ones(self.feature_len)
            self.inx += 1
        else:
            action = np.zeros(self.feature_len)
        return action
    
    def _get_prediction(self, x, date):
        return np.zeros_like(self.window_buffer.get_last()[1].detach().cpu().numpy())
    

class PredictionSignStrategy(PredictorStrategy):
    def __init__(self, dynamics_model, hmax, feature_len, scaler = None, pred_idx = 0, device = 'cpu'):
        super().__init__(dynamics_model, hmax, feature_len, scaler = scaler, pred_idx = pred_idx, device = device)
    def strategy_func(self, prediciton, price):
        action = np.sign(prediciton - price)
        return action