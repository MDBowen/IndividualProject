"""
Stock Trading Environment using FinRL
This script creates a stock trading environment for reinforcement learning
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl import config_tickers
import warnings
warnings.filterwarnings('ignore')


class StockTradingEnvironment:
    """
    A complete stock trading environment setup using FinRL
    """
    
    def __init__(self, 
                 ticker_list=None,
                 start_date='2020-01-01',
                 end_date='2023-12-31',
                 time_interval='1d',
                 technical_indicators=None,
                 initial_amount=1000000,
                 transaction_cost_pct=0.001,
                 hmax=100):
        """
        Initialize the stock trading environment
        
        Args:
            ticker_list: List of stock tickers (default: DOW 30)
            start_date: Start date for data
            end_date: End date for data
            time_interval: Time interval ('1d', '1h', etc.)
            technical_indicators: List of technical indicators to calculate
            initial_amount: Initial portfolio value
            transaction_cost_pct: Transaction cost percentage
            hmax: Maximum shares to trade per action
        """
        self.ticker_list = ticker_list or config_tickers.DOW_30_TICKER
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.hmax = hmax
        
        # Default technical indicators
        self.technical_indicators = technical_indicators or [
            'macd', 'rsi_30', 'cci_30', 'dx_30'
        ]
        
        self.data = None
        self.processed_data = None
        self.train_data = None
        self.trade_data = None
        self.env_train = None
        self.env_trade = None
        
    def download_data(self):
        """Download stock data using FinRL's YahooDownloader"""
        print("Downloading stock data...")
        df = YahooDownloader(
            start_date=self.start_date,
            end_date=self.end_date,
            ticker_list=self.ticker_list
        ).fetch_data()
        
        print(f"Downloaded data shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Tickers: {df['tic'].nunique()}")
        
        self.data = df
        return df
    
    def add_technical_indicators(self):
        """Add technical indicators to the data"""
        print("Adding technical indicators...")
        
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.technical_indicators,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False
        )
        
        self.processed_data = fe.preprocess_data(self.data)
        
        print(f"Processed data shape: {self.processed_data.shape}")
        print(f"Features added: {list(self.processed_data.columns)}")
        
        return self.processed_data
    
    def split_data(self, train_end_date=None):
        """Split data into training and trading periods"""
        if train_end_date is None:
            # Use 80% for training
            dates = sorted(self.processed_data['date'].unique())
            split_idx = int(len(dates) * 0.8)
            train_end_date = dates[split_idx]
        
        print(f"Splitting data at: {train_end_date}")
        
        self.train_data = data_split(
            self.processed_data,
            self.start_date,
            train_end_date
        )
        
        self.trade_data = data_split(
            self.processed_data,
            train_end_date,
            self.end_date
        )
        
        print(f"Train data: {self.train_data.shape}")
        print(f"Trade data: {self.trade_data.shape}")
        
        return self.train_data, self.trade_data
    
    def create_environments(self):
        """Create training and trading environments"""
        print("Creating environments...")
        
        # Stock dimension
        stock_dimension = len(self.train_data.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(self.technical_indicators) * stock_dimension
        
        print(f"Stock dimension: {stock_dimension}")
        print(f"State space: {state_space}")
        
        # Training environment
        self.env_train = StockTradingEnv(
            df=self.train_data,
            stock_dim=stock_dimension,
            hmax=self.hmax,
            initial_amount=self.initial_amount,
            num_stock_shares=[0] * stock_dimension,
            buy_cost_pct=[self.transaction_cost_pct] * stock_dimension,
            sell_cost_pct=[self.transaction_cost_pct] * stock_dimension,
            reward_scaling=1e-4,
            state_space=state_space,
            action_space=stock_dimension,
            tech_indicator_list=self.technical_indicators,
            turbulence_threshold=None,
            risk_indicator_col='vix',
            make_plots=True,
            print_verbosity=5,
            day=0,
            initial=True,
            previous_state=[],
            model_name='',
            mode='',
            iteration=''
        )
        
        # Trading environment
        self.env_trade = StockTradingEnv(
            df=self.trade_data,
            stock_dim=stock_dimension,
            hmax=self.hmax,
            initial_amount=self.initial_amount,
            num_stock_shares=[0] * stock_dimension,
            buy_cost_pct=[self.transaction_cost_pct] * stock_dimension,
            sell_cost_pct=[self.transaction_cost_pct] * stock_dimension,
            reward_scaling=1e-4,
            state_space=state_space,
            action_space=stock_dimension,
            tech_indicator_list=self.technical_indicators,
            turbulence_threshold=None,
            risk_indicator_col='vix',
            make_plots=True,
            print_verbosity=5,
            day=0,
            initial=True,
            previous_state=[],
            model_name='',
            mode='',
            iteration=''
        )
        
        print("Environments created successfully!")
        return self.env_train, self.env_trade
    
    def setup_complete_environment(self):
        """Complete setup pipeline"""
        print("=" * 60)
        print("Setting up Stock Trading Environment")
        print("=" * 60)
        
        # Download data
        self.download_data()
        
        # Add technical indicators
        self.add_technical_indicators()
        
        # Split data
        self.split_data()
        
        # Create environments
        self.create_environments()
        
        print("=" * 60)
        print("Environment setup complete!")
        print("=" * 60)
        
        return self.env_train, self.env_trade
    
    def test_environment(self, env=None, num_episodes=1):
        """Test the environment with random actions"""
        if env is None:
            env = self.env_train
        
        print(f"\nTesting environment for {num_episodes} episode(s)...")
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}")
            print(f"Initial state shape: {np.array(state).shape}")
            
            while not done and steps < 100:  # Limit steps for testing
                # Random action
                action = np.random.uniform(-1, 1, env.action_space)
                
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if steps % 10 == 0:
                    print(f"Step {steps}: Reward = {reward:.2f}, Total = {total_reward:.2f}")
            
            print(f"Episode {episode + 1} finished!")
            print(f"Total steps: {steps}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final portfolio value: {env.asset_memory[-1]:.2f}")


def main():
    """Main function to demonstrate usage"""
    
    # Example 1: Quick setup with default parameters
    print("\n### Example 1: Quick Setup ###\n")
    env_setup = StockTradingEnvironment(
        ticker_list=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_amount=100000
    )

    assert False, 'breakpoint'
    
    # Complete setup
    env_train, env_trade = env_setup.setup_complete_environment()
    
    # Test the environment
    env_setup.test_environment(env_train, num_episodes=1)
    
    print("\n### Environment Information ###")
    print(f"Action space: {env_train.action_space}")
    print(f"State space: {env_train.state_space}")
    print(f"Stock dimension: {env_train.stock_dim}")
    print(f"Technical indicators: {env_train.tech_indicator_list}")
    
    return env_setup


if __name__ == "__main__":
    # Run the main example
    env_setup = main()
    
    print("\n### Usage Examples ###")
    print("""
    # Access the environments:
    env_train = env_setup.env_train
    env_trade = env_setup.env_trade
    
    # Access the data:
    raw_data = env_setup.data
    processed_data = env_setup.processed_data
    train_data = env_setup.train_data
    trade_data = env_setup.trade_data
    
    # Use with RL algorithms (e.g., Stable Baselines3):
    from stable_baselines3 import PPO
    
    model = PPO("MlpPolicy", env_train, verbose=1)
    model.learn(total_timesteps=50000)
    
    # Test the trained model:
    obs = env_trade.reset()
    for i in range(len(env_trade.df.index.unique())):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_trade.step(action)
        if done:
            break
    
    # Plot results:
    env_trade.render()
    """)

