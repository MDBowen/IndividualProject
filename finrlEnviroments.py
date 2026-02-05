import yfinance as yf
import numpy as np
import gymnasium as gym

from sklearn.preprocessing import StandardScaler

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from models.denseModel import train_dense
from data_provider.data_factory import data_provider
from agents.basicStrategies import basicStrategy

from conf import get_train_config

args, setting = get_train_config()

def simulate_strategy(agent, data):

    env = StockTradingEnv(data, 
                          stock_dim = 98, 
                          hmax = 10000,
                          initial_amount = 10000,
                          num_stock_shares = [], 
                          buy_cost_pct = 1e-2,
                          sell_cost_pct = 1e-2,
                          turbulence_threshold = 100
                          print_verbosity(1)
                          )


    state, dict =env.reset()

    done = False

    while not done:

        action = agent.get_action(state)

        state, reward, done, _, dict = env.step()


if __name__ == '__main__':

    train_set, training_loader = data_provider(args, flag = 'train')
    test_set, test_loader = data_provider(args, flag='test')

    print(f'Training model for {len(training_loader)}')
    dense = train_dense(training_loader)
    print(f'Training finshed')

    train_data = train_set.get_data()
    df = test_set.get_data_frame()

    scaler = StandardScaler()
    scaler.fit(train_data)


    print('Data frame:')
    print(df.head(), '\n')

    agent = basicStrategy(args, dense, scaler)

    simulate_strategy(agent, df)




