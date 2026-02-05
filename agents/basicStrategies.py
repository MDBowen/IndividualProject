
import numpy as np
import torch

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