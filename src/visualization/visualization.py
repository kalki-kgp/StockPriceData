import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockVisualization:
    """Comprehensive visualization tools for stock analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    def plot_stock_data(self, 
                       data: pd.DataFrame,
                       price_cols: List[str] = ['open', 'high', 'low', 'close'],
                       volume_col: str = 'volume',
                       date_col: str = 'date',
                       title: str = "Stock Price Analysis",
                       save_path: Optional[str] = None):
        """
        Plot comprehensive stock data analysis
        
        Args:
            data: Stock data DataFrame
            price_cols: Price columns to plot
            volume_col: Volume column
            date_col: Date column
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # Ensure date column is datetime
        if date_col in data.columns:
            x = pd.to_datetime(data[date_col])
        else:
            x = data.index
        
        # Price plot
        for i, col in enumerate(price_cols):
            if col in data.columns:
                axes[0].plot(x, data[col], label=col.title(), 
                           color=self.colors[i], linewidth=1.5, alpha=0.8)
        
        axes[0].set_title(f'{title} - Price Movement')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume plot
        if volume_col in data.columns:
            axes[1].bar(x, data[volume_col], alpha=0.6, color='steelblue', width=0.8)
            axes[1].set_title('Trading Volume')
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
        
        # Returns plot
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            axes[2].plot(x[1:], returns, color='darkred', alpha=0.7, linewidth=1)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[2].set_title('Daily Returns')
            axes[2].set_ylabel('Return (%)')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stock data plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_technical_indicators(self, 
                                 data: pd.DataFrame,
                                 price_col: str = 'close',
                                 indicators: List[str] = None,
                                 date_col: str = 'date',
                                 title: str = "Technical Indicators",
                                 save_path: Optional[str] = None):
        """
        Plot technical indicators
        
        Args:
            data: DataFrame with stock data and indicators
            price_col: Price column name
            indicators: List of indicator columns to plot
            date_col: Date column
            title: Plot title
            save_path: Path to save plot
        """
        if indicators is None:
            # Default indicators to look for
            indicators = ['sma_20', 'ema_20', 'bb_upper', 'bb_lower', 'rsi', 'macd']
        
        # Filter indicators that exist in data
        available_indicators = [ind for ind in indicators if ind in data.columns]
        
        if not available_indicators:
            logger.warning("No technical indicators found in data")
            return
        
        n_indicators = len(available_indicators)
        fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(self.figsize[0], 4 * (n_indicators + 1)))
        
        if n_indicators == 0:
            axes = [axes]
        
        # Ensure date column is datetime
        if date_col in data.columns:
            x = pd.to_datetime(data[date_col])
        else:
            x = data.index
        
        # Price with moving averages
        axes[0].plot(x, data[price_col], label='Price', color='black', linewidth=1.5)
        
        ma_indicators = [ind for ind in available_indicators if 'ma' in ind.lower() or 'bb' in ind.lower()]
        for i, indicator in enumerate(ma_indicators):
            axes[0].plot(x, data[indicator], label=indicator.upper(), 
                        color=self.colors[i % len(self.colors)], linewidth=1, alpha=0.8)
        
        axes[0].set_title(f'{title} - Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Other indicators
        other_indicators = [ind for ind in available_indicators if ind not in ma_indicators]
        for i, indicator in enumerate(other_indicators, 1):
            if i < len(axes):
                axes[i].plot(x, data[indicator], color=self.colors[i % len(self.colors)], linewidth=1.5)
                axes[i].set_title(f'{indicator.upper()}')
                axes[i].set_ylabel(indicator.upper())
                axes[i].grid(True, alpha=0.3)
                
                # Add reference lines for specific indicators
                if 'rsi' in indicator.lower():
                    axes[i].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
                    axes[i].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
                    axes[i].legend()
                elif 'macd' in indicator.lower() and 'histogram' not in indicator.lower():
                    axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        if len(axes) > 1:
            axes[-1].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Technical indicators plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_sentiment_analysis(self, 
                               sentiment_data: pd.DataFrame,
                               stock_data: pd.DataFrame = None,
                               sentiment_col: str = 'sentiment_index',
                               price_col: str = 'close',
                               date_col: str = 'created_at',
                               title: str = "Sentiment Analysis",
                               save_path: Optional[str] = None):
        """
        Plot sentiment analysis with stock price correlation
        
        Args:
            sentiment_data: DataFrame with sentiment data
            stock_data: Optional DataFrame with stock data
            sentiment_col: Sentiment column name
            price_col: Price column name
            date_col: Date column name
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Ensure date column is datetime
        x_sentiment = pd.to_datetime(sentiment_data[date_col])
        
        # Sentiment plot
        axes[0].plot(x_sentiment, sentiment_data[sentiment_col], 
                    color='blue', linewidth=1.5, alpha=0.8, label='Sentiment Index')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{title} - Sentiment Over Time')
        axes[0].set_ylabel('Sentiment')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Tweet count (if available)
        if 'tweet_count' in sentiment_data.columns:
            axes[1].bar(x_sentiment, sentiment_data['tweet_count'], 
                       alpha=0.6, color='green', width=0.8)
            axes[1].set_title('Tweet Volume')
            axes[1].set_ylabel('Tweet Count')
            axes[1].grid(True, alpha=0.3)
        
        # Stock price (if provided)
        if stock_data is not None and price_col in stock_data.columns:
            stock_date_col = 'date' if 'date' in stock_data.columns else stock_data.index
            x_stock = pd.to_datetime(stock_data[stock_date_col])
            
            axes[2].plot(x_stock, stock_data[price_col], 
                        color='red', linewidth=1.5, alpha=0.8, label='Stock Price')
            axes[2].set_title('Stock Price')
            axes[2].set_ylabel('Price ($)')
            axes[2].set_xlabel('Date')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].set_xlabel('Date')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sentiment analysis plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_gan_training(self, 
                         history: Dict[str, List[float]],
                         title: str = "GAN Training Progress",
                         save_path: Optional[str] = None):
        """
        Plot GAN training history
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        epochs = range(len(history['generator_loss']))
        
        # Loss plot
        axes[0].plot(epochs, history['generator_loss'], 
                    label='Generator Loss', color='blue', linewidth=1.5)
        axes[0].plot(epochs, history['discriminator_loss'], 
                    label='Discriminator Loss', color='red', linewidth=1.5)
        axes[0].set_title(f'{title} - Loss')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Discriminator accuracy
        axes[1].plot(epochs, history['discriminator_accuracy'], 
                    color='green', linewidth=1.5, label='Discriminator Accuracy')
        axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random Guess')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"GAN training plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_correlation_matrix(self, 
                               data: pd.DataFrame,
                               features: List[str] = None,
                               title: str = "Feature Correlation Matrix",
                               save_path: Optional[str] = None):
        """
        Plot correlation matrix heatmap
        
        Args:
            data: DataFrame with features
            features: List of features to include
            title: Plot title
            save_path: Path to save plot
        """
        if features is None:
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[features]
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation'})
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        else:
            plt.show()
    
    def create_interactive_dashboard(self, 
                                   data: pd.DataFrame,
                                   price_col: str = 'close',
                                   volume_col: str = 'volume',
                                   date_col: str = 'date',
                                   title: str = "Interactive Stock Dashboard",
                                   save_path: Optional[str] = None):
        """
        Create interactive Plotly dashboard
        
        Args:
            data: Stock data
            price_col: Price column
            volume_col: Volume column
            date_col: Date column
            title: Dashboard title
            save_path: Path to save HTML file
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Stock Price', 'Volume', 'Technical Indicators'),
            row_width=[0.3, 0.2, 0.5]
        )
        
        # Ensure date column is datetime
        dates = pd.to_datetime(data[date_col])
        
        # Price plot
        fig.add_trace(
            go.Scatter(x=dates, y=data[price_col], 
                      mode='lines', name='Close Price', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Volume plot
        if volume_col in data.columns:
            fig.add_trace(
                go.Bar(x=dates, y=data[volume_col], 
                      name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
        
        # Technical indicators (if available)
        indicators = ['sma_20', 'ema_20', 'rsi']
        for indicator in indicators:
            if indicator in data.columns:
                fig.add_trace(
                    go.Scatter(x=dates, y=data[indicator], 
                              mode='lines', name=indicator.upper(),
                              line=dict(width=1)),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800,
            hovermode='x unified'
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Indicator Value", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        else:
            fig.show()
    
    def plot_prediction_uncertainty(self, 
                                   y_true: np.ndarray,
                                   y_pred_mean: np.ndarray,
                                   y_pred_std: np.ndarray,
                                   dates: Optional[pd.DatetimeIndex] = None,
                                   confidence_level: float = 0.95,
                                   title: str = "Prediction with Uncertainty",
                                   save_path: Optional[str] = None):
        """
        Plot predictions with uncertainty bands
        
        Args:
            y_true: True values
            y_pred_mean: Mean predictions
            y_pred_std: Standard deviation of predictions
            dates: Optional date index
            confidence_level: Confidence level for bands
            title: Plot title
            save_path: Path to save plot
        """
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% confidence
        upper_bound = y_pred_mean + z_score * y_pred_std
        lower_bound = y_pred_mean - z_score * y_pred_std
        
        plt.figure(figsize=self.figsize)
        
        # Create x-axis
        if dates is not None:
            x = dates
        else:
            x = np.arange(len(y_true))
        
        # Plot true values
        plt.plot(x, y_true, label='Actual', color='blue', linewidth=2, alpha=0.8)
        
        # Plot predictions
        plt.plot(x, y_pred_mean, label='Predicted', color='red', linewidth=2, alpha=0.8)
        
        # Plot uncertainty bands
        plt.fill_between(x, lower_bound, upper_bound, 
                        alpha=0.3, color='red', 
                        label=f'{int(confidence_level*100)}% Confidence Interval')
        
        plt.title(title)
        plt.xlabel('Date' if dates is not None else 'Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Uncertainty plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Example usage"""
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    
    # Generate synthetic stock data
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    stock_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'volume': np.random.randint(1000, 10000, n_samples),
        'sma_20': pd.Series(prices).rolling(20).mean(),
        'rsi': np.random.uniform(20, 80, n_samples)
    })
    
    # Generate synthetic sentiment data
    sentiment_data = pd.DataFrame({
        'created_at': dates,
        'sentiment_index': np.random.normal(0, 0.3, n_samples),
        'tweet_count': np.random.randint(1, 50, n_samples)
    })
    
    # Initialize visualizer
    viz = StockVisualization()
    
    # Test various plots
    print("Creating stock data visualization...")
    viz.plot_stock_data(stock_data, title="Sample Stock Analysis")
    
    print("Creating technical indicators plot...")
    viz.plot_technical_indicators(stock_data, indicators=['sma_20', 'rsi'])
    
    print("Creating sentiment analysis plot...")
    viz.plot_sentiment_analysis(sentiment_data, stock_data)
    
    print("Creating correlation matrix...")
    viz.plot_correlation_matrix(stock_data.select_dtypes(include=[np.number]))
    
    print("All visualizations created successfully!")

if __name__ == "__main__":
    main()