import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
from sklearn.model_selection import train_test_split
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.stock_fetch import StockDataFetcher
from data_pipeline.twitter_fetch import TwitterDataFetcher
from sentiment.sentiment import SentimentAnalyzer
from features.features import FeatureEngineer
from models.gan_model import StockGAN, TimeSeriesGAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockForecastTrainer:
    """Main trainer class for stock forecasting with GAN"""
    
    def __init__(self, config: dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stock_fetcher = StockDataFetcher(config.get('symbol', 'AMZN'))
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        
        # Twitter fetcher (requires API key)
        twitter_token = config.get('twitter_bearer_token')
        if twitter_token:
            self.twitter_fetcher = TwitterDataFetcher(twitter_token)
        else:
            self.twitter_fetcher = None
            logger.warning("No Twitter bearer token provided. Using mock sentiment data.")
    
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        Fetch and prepare all data for training
        
        Returns:
            Prepared dataset
        """
        logger.info("Fetching stock data...")
        
        # Fetch stock data
        stock_data = self.stock_fetcher.fetch_historical_data(
            period=self.config.get('data_period', '1y'),
            interval=self.config.get('data_interval', '1h')
        )
        
        if stock_data.empty:
            raise ValueError("No stock data fetched")
        
        # Fetch sentiment data
        if self.twitter_fetcher:
            logger.info("Fetching Twitter data...")
            tweets = self.twitter_fetcher.fetch_tweets(
                max_results=self.config.get('max_tweets', 1000),
                hours_back=24 * 30  # 30 days
            )
            
            if not tweets.empty:
                tweets = self.twitter_fetcher.preprocess_tweets(tweets)
                sentiment_data = self.sentiment_analyzer.analyze_batch_sentiment(tweets)
                sentiment_index = self.sentiment_analyzer.calculate_sentiment_index(sentiment_data)
            else:
                sentiment_index = self._create_mock_sentiment_data(stock_data)
        else:
            sentiment_index = self._create_mock_sentiment_data(stock_data)
        
        # Add technical indicators
        logger.info("Adding technical indicators...")
        stock_with_indicators = self.feature_engineer.add_all_technical_indicators(stock_data)
        
        # Add price features
        stock_with_features = self.feature_engineer.add_price_features(stock_with_indicators)
        
        # Merge with sentiment data
        logger.info("Merging stock and sentiment data...")
        merged_data = self.feature_engineer.merge_sentiment_data(
            stock_with_features, sentiment_index
        )
        
        # Create lag and rolling features
        important_cols = ['close', 'volume', 'rsi', 'sentiment_index', 'macd']
        merged_data = self.feature_engineer.create_lag_features(
            merged_data, important_cols, lags=[1, 2, 3, 5]
        )
        merged_data = self.feature_engineer.create_rolling_features(
            merged_data, important_cols, windows=[5, 10, 20]
        )
        
        # Remove rows with NaN values
        merged_data = merged_data.dropna().reset_index(drop=True)
        
        logger.info(f"Final dataset shape: {merged_data.shape}")
        return merged_data
    
    def _create_mock_sentiment_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Create mock sentiment data for testing"""
        logger.info("Creating mock sentiment data...")
        
        # Create hourly timestamps matching stock data
        start_time = stock_data['date'].min()
        end_time = stock_data['date'].max()
        hourly_range = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        # Generate synthetic sentiment data with some correlation to price movements
        np.random.seed(42)
        sentiment_data = pd.DataFrame({
            'created_at': hourly_range,
            'sentiment_mean': np.random.normal(0, 0.2, len(hourly_range)),
            'sentiment_index': np.random.normal(0, 0.15, len(hourly_range)),
            'tweet_count': np.random.randint(1, 20, len(hourly_range)),
            'sentiment_momentum': np.random.normal(0, 0.1, len(hourly_range))
        })
        
        return sentiment_data
    
    def prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """
        Prepare data for GAN training
        
        Args:
            data: Prepared dataset
            
        Returns:
            Tuple of training data
        """
        logger.info("Preparing training sequences...")
        
        # Normalize features
        exclude_cols = ['date', 'datetime', 'symbol']
        normalized_data, self.scaler = self.feature_engineer.normalize_features(
            data, exclude_cols=exclude_cols
        )
        
        # Prepare sequences
        sequence_length = self.config.get('sequence_length', 60)
        target_col = self.config.get('target_column', 'close')
        
        # Select feature columns (exclude target and non-numeric columns)
        feature_cols = [col for col in normalized_data.columns 
                       if col not in exclude_cols + [target_col]]
        
        X, y = self.feature_engineer.prepare_sequences(
            normalized_data,
            sequence_length=sequence_length,
            target_col=target_col,
            feature_cols=feature_cols
        )
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )
        
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
        
        # Save feature columns and scaler for later use
        self.feature_columns = feature_cols
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> StockGAN:
        """
        Train the GAN model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained GAN model
        """
        logger.info("Initializing GAN model...")
        
        # Initialize GAN
        gan = StockGAN(
            sequence_length=self.config.get('sequence_length', 60),
            n_features=X_train.shape[2],
            latent_dim=self.config.get('latent_dim', 100),
            generator_lr=self.config.get('generator_lr', 0.0002),
            discriminator_lr=self.config.get('discriminator_lr', 0.0002)
        )
        
        # Train the model
        logger.info("Starting GAN training...")
        history = gan.train(
            X_train, y_train,
            epochs=self.config.get('epochs', 1000),
            batch_size=self.config.get('batch_size', 32),
            save_interval=self.config.get('save_interval', 200),
            model_dir=self.config.get('model_dir', 'trained_models')
        )
        
        # Plot training history
        history_plot_path = os.path.join(
            self.config.get('output_dir', 'outputs'),
            'training_history.png'
        )
        os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
        gan.plot_training_history(history_plot_path)
        
        return gan
    
    def save_training_artifacts(self, gan: StockGAN, data: pd.DataFrame):
        """Save training artifacts"""
        output_dir = self.config.get('output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'final_model')
        gan.save_models(model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        features_path = os.path.join(output_dir, 'feature_columns.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Save sample data for reference
        data_sample_path = os.path.join(output_dir, 'data_sample.csv')
        data.head(100).to_csv(data_sample_path, index=False)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        logger.info(f"Training artifacts saved to {output_dir}")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Fetch and prepare data
            data = self.fetch_and_prepare_data()
            
            # Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(data)
            
            # Train model
            gan = self.train_model(X_train, y_train)
            
            # Save artifacts
            self.save_training_artifacts(gan, data)
            
            logger.info("Training pipeline completed successfully!")
            
            return gan, X_test, y_test
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train Stock Price Forecasting GAN')
    
    parser.add_argument('--symbol', default='AMZN', help='Stock symbol to train on')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    parser.add_argument('--data-period', default='1y', help='Data period to fetch')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--twitter-token', help='Twitter Bearer Token')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'symbol': args.symbol,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'data_period': args.data_period,
        'output_dir': args.output_dir,
        'twitter_bearer_token': args.twitter_token,
        'data_interval': '1h',
        'max_tweets': 1000,
        'test_size': 0.2,
        'latent_dim': 100,
        'generator_lr': 0.0002,
        'discriminator_lr': 0.0002,
        'save_interval': 200,
        'model_dir': os.path.join(args.output_dir, 'models'),
        'target_column': 'close'
    }
    
    # Initialize and run trainer
    trainer = StockForecastTrainer(config)
    gan, X_test, y_test = trainer.run_full_pipeline()
    
    print("Training completed successfully!")
    print(f"Model and artifacts saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()