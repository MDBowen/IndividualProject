"""Adds Technical Indicators To A Pandas DataFrame"""

class AddTechnicalIndicators:
    def __init__(self, data, tickers):
        """Takes in a DataFrame with stock data."""
        self.data = data
        self.tickers = tickers

    def add_sma(self, window=30):

        """Add Simple Moving Average (SMA) to the data."""

        data = self.data

        for name in self.tickers:
            data[name, 'SMA'] = data[name].rolling(window=window).mean()

        self.data = data
        return data

    def add_ema(self, span=30):
        """Add Exponential Moving Average (EMA) to the data."""

        data = self.data

        for name in self.tickers:
            data[name, 'EMA'] = data[name].ewm(span=span, adjust=False).mean()
        self.data = data
        return data
    

    def add_rsi(self, window=30):
        """Add Relative Strength Index (RSI) to the data."""

        data = self.data

        for name in self.tickers:
            delta = data[name].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            data[name, 'RSI'] = 100 - (100 / (1 + rs))

        self.data = data
        return data
    
    def get_rolling_z(self, window=30):
        """Add Rolling Z-Score to the data."""
        """Final Ouput must then me computed as """

        data = self.data
        normalized_data = data.copy()

        for col in self.data.columns:
            rolling_mean = data[col].rolling(window=window).mean()
            rolling_std = data[col].rolling(window=window).std()
            normalized_data[col] = (data[col] - rolling_mean) / rolling_std
        
        return normalized_data