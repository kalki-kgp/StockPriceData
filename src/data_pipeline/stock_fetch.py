import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, symbol: str = 'AMZN'):
        """
        Initialize stock data fetcher
        
        Args:
            symbol: Stock symbol to fetch data for
        """
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        
    def fetch_historical_data(self, 
                            period: str = '1y',
                            interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {self.symbol} data for period: {period}, interval: {interval}")
            
            data = self.ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.error(f"No data found for {self.symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Rename columns to lowercase
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure datetime column is properly formatted
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            elif 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.rename(columns={'datetime': 'date'}, inplace=True)
                
            logger.info(f"Successfully fetched {len(data)} records for {self.symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def fetch_real_time_data(self) -> Optional[dict]:
        """
        Fetch real-time stock data
        
        Returns:
            Dictionary with current stock information
        """
        try:
            info = self.ticker.info
            
            real_time_data = {
                'symbol': self.symbol,
                'current_price': info.get('currentPrice', None),
                'previous_close': info.get('previousClose', None),
                'open': info.get('open', None),
                'day_high': info.get('dayHigh', None),
                'day_low': info.get('dayLow', None),
                'volume': info.get('volume', None),
                'market_cap': info.get('marketCap', None),
                'timestamp': datetime.now()
            }
            
            logger.info(f"Fetched real-time data for {self.symbol}")
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            return None
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added return columns
        """
        if data.empty or 'close' not in data.columns:
            return data
            
        df = data.copy()
        
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = pd.np.log(df['close'] / df['close'].shift(1))
        
        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # Calculate volatility (rolling 20-day)
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * (252 ** 0.5)
        
        return df
    
    def get_date_range_data(self, 
                           start_date: str, 
                           end_date: str, 
                           interval: str = '1d') -> pd.DataFrame:
        """
        Fetch stock data for specific date range
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            interval: Data interval
            
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Fetching {self.symbol} data from {start_date} to {end_date}")
            
            data = self.ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {self.symbol} in the specified date range")
                return pd.DataFrame()
                
            # Process the data similar to historical data
            data.reset_index(inplace=True)
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            elif 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.rename(columns={'datetime': 'date'}, inplace=True)
                
            logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching date range data: {e}")
            return pd.DataFrame()
    
    def save_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Save stock data to CSV file
        
        Args:
            data: DataFrame to save
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            data.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            return False

def main():
    """Example usage"""
    # Initialize fetcher for Amazon stock
    fetcher = StockDataFetcher('AMZN')
    
    # Fetch 6 months of daily data
    historical_data = fetcher.fetch_historical_data(period='6mo', interval='1d')
    
    if not historical_data.empty:
        print(f"Fetched {len(historical_data)} historical records")
        
        # Calculate returns
        historical_data = fetcher.calculate_returns(historical_data)
        
        # Display summary
        print(f"Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
        print(f"Price range: ${historical_data['close'].min():.2f} - ${historical_data['close'].max():.2f}")
        print(f"Average daily return: {historical_data['daily_return'].mean():.4f}")
        
        # Save to CSV
        fetcher.save_to_csv(historical_data, 'amzn_historical_data.csv')
        
    # Fetch real-time data
    real_time = fetcher.fetch_real_time_data()
    if real_time:
        print(f"Current price: ${real_time['current_price']:.2f}")

if __name__ == "__main__":
    main()