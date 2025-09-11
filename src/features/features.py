import pandas as pd
import numpy as np
import talib as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical analysis indicators for stock data"""
    
    @staticmethod
    def moving_averages(df: pd.DataFrame, 
                       price_col: str = 'close',
                       periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages
        
        Args:
            df: DataFrame with stock data
            price_col: Column name for price data
            periods: List of periods for moving averages
            
        Returns:
            DataFrame with added MA columns
        """
        result_df = df.copy()
        
        for period in periods:
            # Simple Moving Average
            result_df[f'sma_{period}'] = ta.SMA(df[price_col].values, timeperiod=period)
            
            # Exponential Moving Average
            result_df[f'ema_{period}'] = ta.EMA(df[price_col].values, timeperiod=period)
            
        return result_df
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, 
                       price_col: str = 'close',
                       period: int = 20,
                       std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with stock data
            price_col: Column name for price data
            period: Period for calculation
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands columns
        """
        result_df = df.copy()
        
        upper, middle, lower = ta.BBANDS(
            df[price_col].values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0
        )
        
        result_df['bb_upper'] = upper
        result_df['bb_middle'] = middle
        result_df['bb_lower'] = lower
        
        # Calculate Bollinger Band Width and %B
        result_df['bb_width'] = (upper - lower) / middle
        result_df['bb_percent'] = (df[price_col].values - lower) / (upper - lower)
        
        return result_df
    
    @staticmethod
    def macd(df: pd.DataFrame, 
             price_col: str = 'close',
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with stock data
            price_col: Column name for price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD columns
        """
        result_df = df.copy()
        
        macd_line, macd_signal, macd_histogram = ta.MACD(
            df[price_col].values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        result_df['macd'] = macd_line
        result_df['macd_signal'] = macd_signal
        result_df['macd_histogram'] = macd_histogram
        
        return result_df
    
    @staticmethod
    def rsi(df: pd.DataFrame, 
            price_col: str = 'close',
            period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with stock data
            price_col: Column name for price data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with RSI column
        """
        result_df = df.copy()
        result_df['rsi'] = ta.RSI(df[price_col].values, timeperiod=period)
        
        return result_df
    
    @staticmethod
    def stochastic_oscillator(df: pd.DataFrame,
                            high_col: str = 'high',
                            low_col: str = 'low',
                            close_col: str = 'close',
                            k_period: int = 14,
                            d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        
        Args:
            df: DataFrame with stock data
            high_col, low_col, close_col: Column names
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with Stochastic columns
        """
        result_df = df.copy()
        
        slowk, slowd = ta.STOCH(
            df[high_col].values,
            df[low_col].values,
            df[close_col].values,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        
        result_df['stoch_k'] = slowk
        result_df['stoch_d'] = slowd
        
        return result_df
    
    @staticmethod
    def volume_indicators(df: pd.DataFrame,
                         price_col: str = 'close',
                         volume_col: str = 'volume') -> pd.DataFrame:
        """
        Calculate volume-based indicators
        
        Args:
            df: DataFrame with stock data
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            DataFrame with volume indicators
        """
        result_df = df.copy()
        
        # Volume Moving Average
        result_df['volume_sma_20'] = ta.SMA(df[volume_col].values, timeperiod=20)
        
        # On-Balance Volume
        result_df['obv'] = ta.OBV(df[price_col].values, df[volume_col].values)
        
        # Volume Rate of Change
        result_df['volume_roc'] = ta.ROC(df[volume_col].values, timeperiod=10)
        
        return result_df

class FeatureEngineer:
    """Main feature engineering class that combines stock data with sentiment"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.technical_indicators = TechnicalIndicators()
        
    def add_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Adding technical indicators")
        
        result_df = df.copy()
        
        # Moving Averages
        result_df = self.technical_indicators.moving_averages(result_df)
        
        # Bollinger Bands
        result_df = self.technical_indicators.bollinger_bands(result_df)
        
        # MACD
        result_df = self.technical_indicators.macd(result_df)
        
        # RSI
        result_df = self.technical_indicators.rsi(result_df)
        
        # Stochastic Oscillator
        result_df = self.technical_indicators.stochastic_oscillator(result_df)
        
        # Volume Indicators
        if 'volume' in result_df.columns:
            result_df = self.technical_indicators.volume_indicators(result_df)
        
        logger.info(f"Added technical indicators. DataFrame shape: {result_df.shape}")
        return result_df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with price features
        """
        result_df = df.copy()
        
        # Price changes
        result_df['price_change'] = result_df['close'] - result_df['open']
        result_df['price_change_pct'] = (result_df['close'] - result_df['open']) / result_df['open']
        
        # High-Low range
        result_df['high_low_range'] = result_df['high'] - result_df['low']
        result_df['high_low_pct'] = result_df['high_low_range'] / result_df['close']
        
        # Gap analysis
        result_df['gap'] = result_df['open'] - result_df['close'].shift(1)
        result_df['gap_pct'] = result_df['gap'] / result_df['close'].shift(1)
        
        # Price position within day's range
        result_df['close_position'] = (result_df['close'] - result_df['low']) / (result_df['high'] - result_df['low'])
        
        return result_df
    
    def merge_sentiment_data(self, 
                           stock_df: pd.DataFrame, 
                           sentiment_df: pd.DataFrame,
                           stock_time_col: str = 'date',
                           sentiment_time_col: str = 'created_at') -> pd.DataFrame:
        """
        Merge stock data with sentiment data
        
        Args:
            stock_df: DataFrame with stock data
            sentiment_df: DataFrame with sentiment data
            stock_time_col: Stock data timestamp column
            sentiment_time_col: Sentiment data timestamp column
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging stock and sentiment data")
        
        # Ensure datetime columns
        stock_df = stock_df.copy()
        sentiment_df = sentiment_df.copy()
        
        stock_df[stock_time_col] = pd.to_datetime(stock_df[stock_time_col])
        sentiment_df[sentiment_time_col] = pd.to_datetime(sentiment_df[sentiment_time_col])
        
        # Round timestamps to nearest hour for joining
        stock_df['hour'] = stock_df[stock_time_col].dt.floor('H')
        sentiment_df['hour'] = sentiment_df[sentiment_time_col].dt.floor('H')
        
        # Aggregate sentiment data by hour
        sentiment_agg = sentiment_df.groupby('hour').agg({
            'sentiment_mean': 'mean',
            'sentiment_index': 'mean',
            'tweet_count': 'sum',
            'sentiment_momentum': 'mean'
        }).reset_index()
        
        # Merge with stock data
        merged_df = pd.merge(stock_df, sentiment_agg, on='hour', how='left')
        
        # Forward fill sentiment data for missing periods
        sentiment_cols = ['sentiment_mean', 'sentiment_index', 'tweet_count', 'sentiment_momentum']
        merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(method='ffill')
        
        # Fill remaining NaN with neutral values
        merged_df['sentiment_mean'] = merged_df['sentiment_mean'].fillna(0)
        merged_df['sentiment_index'] = merged_df['sentiment_index'].fillna(0)
        merged_df['tweet_count'] = merged_df['tweet_count'].fillna(0)
        merged_df['sentiment_momentum'] = merged_df['sentiment_momentum'].fillna(0)
        
        # Drop the hour column
        merged_df = merged_df.drop('hour', axis=1)
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df
    
    def create_lag_features(self, 
                          df: pd.DataFrame, 
                          columns: List[str],
                          lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: DataFrame to add lags to
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns:
                for lag in lags:
                    result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
        
        return result_df
    
    def create_rolling_features(self, 
                              df: pd.DataFrame,
                              columns: List[str],
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame to add rolling features to
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns:
                for window in windows:
                    result_df[f'{col}_rolling_mean_{window}'] = result_df[col].rolling(window).mean()
                    result_df[f'{col}_rolling_std_{window}'] = result_df[col].rolling(window).std()
                    result_df[f'{col}_rolling_min_{window}'] = result_df[col].rolling(window).min()
                    result_df[f'{col}_rolling_max_{window}'] = result_df[col].rolling(window).max()
        
        return result_df
    
    def normalize_features(self, 
                         df: pd.DataFrame, 
                         method: str = 'standard',
                         exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, object]:
        """
        Normalize features
        
        Args:
            df: DataFrame to normalize
            method: Normalization method ('standard' or 'minmax')
            exclude_cols: Columns to exclude from normalization
            
        Returns:
            Tuple of (normalized DataFrame, fitted scaler)
        """
        if exclude_cols is None:
            exclude_cols = ['date', 'datetime', 'symbol']
        
        result_df = df.copy()
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        result_df[cols_to_normalize] = scaler.fit_transform(result_df[cols_to_normalize])
        
        logger.info(f"Normalized {len(cols_to_normalize)} features using {method} scaling")
        return result_df, scaler
    
    def prepare_sequences(self, 
                        df: pd.DataFrame,
                        sequence_length: int = 60,
                        target_col: str = 'close',
                        feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series modeling
        
        Args:
            df: DataFrame with features
            sequence_length: Length of input sequences
            target_col: Target variable column
            feature_cols: Feature columns to use
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in ['date', 'datetime', target_col]]
        
        # Remove rows with NaN values
        clean_df = df[feature_cols + [target_col]].dropna()
        
        X_data = clean_df[feature_cols].values
        y_data = clean_df[target_col].values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_data)):
            X_sequences.append(X_data[i-sequence_length:i])
            y_sequences.append(y_data[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
        logger.info(f"Feature shape: {X_sequences.shape}, Target shape: {y_sequences.shape}")
        
        return X_sequences, y_sequences

def main():
    """Example usage"""
    # Create sample stock data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_stock_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Create sample sentiment data
    sample_sentiment_data = pd.DataFrame({
        'created_at': dates,
        'sentiment_mean': np.random.normal(0, 0.3, len(dates)),
        'sentiment_index': np.random.normal(0, 0.2, len(dates)),
        'tweet_count': np.random.randint(0, 50, len(dates)),
        'sentiment_momentum': np.random.normal(0, 0.1, len(dates))
    })
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Add technical indicators
    stock_with_indicators = engineer.add_all_technical_indicators(sample_stock_data)
    
    # Add price features
    stock_with_features = engineer.add_price_features(stock_with_indicators)
    
    # Merge with sentiment data
    merged_data = engineer.merge_sentiment_data(stock_with_features, sample_sentiment_data)
    
    # Create lag and rolling features
    important_cols = ['close', 'volume', 'rsi', 'sentiment_index']
    merged_data = engineer.create_lag_features(merged_data, important_cols, lags=[1, 2, 3])
    merged_data = engineer.create_rolling_features(merged_data, important_cols, windows=[5, 10])
    
    # Normalize features
    normalized_data, scaler = engineer.normalize_features(merged_data, exclude_cols=['date'])
    
    print(f"Final dataset shape: {normalized_data.shape}")
    print(f"Number of features: {len([col for col in normalized_data.columns if col != 'date'])}")
    
    # Prepare sequences for modeling
    X, y = engineer.prepare_sequences(normalized_data, sequence_length=20, target_col='close')
    print(f"Sequence data - X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    main()