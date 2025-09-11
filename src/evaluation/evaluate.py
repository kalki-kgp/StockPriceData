import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import logging
from typing import Tuple, Dict, List, Optional
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive evaluation for stock price forecasting models"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_history = []
        
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
        
        # R-squared
        r2 = r2_score(y_true, y_pred)
        
        # Correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'R2': r2,
            'Correlation': correlation
        }
        
        return metrics
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of directional metrics
        """
        # Calculate price changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Directional accuracy
        correct_direction = np.sum((true_changes > 0) == (pred_changes > 0))
        directional_accuracy = correct_direction / len(true_changes) * 100
        
        # Up movement accuracy
        up_movements = true_changes > 0
        if np.sum(up_movements) > 0:
            up_accuracy = np.sum((true_changes > 0) & (pred_changes > 0)) / np.sum(up_movements) * 100
        else:
            up_accuracy = 0
        
        # Down movement accuracy
        down_movements = true_changes < 0
        if np.sum(down_movements) > 0:
            down_accuracy = np.sum((true_changes < 0) & (pred_changes < 0)) / np.sum(down_movements) * 100
        else:
            down_accuracy = 0
        
        metrics = {
            'Directional_Accuracy': directional_accuracy,
            'Up_Movement_Accuracy': up_accuracy,
            'Down_Movement_Accuracy': down_accuracy
        }
        
        return metrics
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate trading-specific metrics
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            
        Returns:
            Dictionary of trading metrics
        """
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Simple trading strategy: buy if predicted return > 0
        signals = pred_returns > 0
        strategy_returns = true_returns * signals.astype(int)
        
        # Trading metrics
        total_return = np.sum(strategy_returns) * 100
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) * 100
        
        # Sharpe ratio (assuming 252 trading days)
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        metrics = {
            'Total_Return_%': total_return,
            'Win_Rate_%': win_rate,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown_%': max_drawdown
        }
        
        return metrics
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Complete metrics dictionary
        """
        logger.info(f"Evaluating {model_name}")
        
        # Calculate all metrics
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
        directional_metrics = self.calculate_directional_accuracy(y_true, y_pred)
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred)
        
        # Combine all metrics
        all_metrics = {
            'Model': model_name,
            'Evaluation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **regression_metrics,
            **directional_metrics,
            **trading_metrics
        }
        
        # Store in history
        self.metrics_history.append(all_metrics)
        
        # Print summary
        self._print_metrics_summary(all_metrics)
        
        return all_metrics
    
    def _print_metrics_summary(self, metrics: Dict[str, float]):
        """Print formatted metrics summary"""
        print(f"\n=== {metrics['Model']} Evaluation Results ===")
        print(f"Date: {metrics['Evaluation_Date']}")
        print("\nRegression Metrics:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  Correlation: {metrics['Correlation']:.4f}")
        
        print("\nDirectional Metrics:")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        print(f"  Up Movement Accuracy: {metrics['Up_Movement_Accuracy']:.2f}%")
        print(f"  Down Movement Accuracy: {metrics['Down_Movement_Accuracy']:.2f}%")
        
        print("\nTrading Metrics:")
        print(f"  Total Return: {metrics['Total_Return_%']:.2f}%")
        print(f"  Win Rate: {metrics['Win_Rate_%']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['Max_Drawdown_%']:.2f}%")
        print("=" * 50)
    
    def compare_models(self, results: List[Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: List of evaluation results
            
        Returns:
            Comparison DataFrame
        """
        df = pd.DataFrame(results)
        df = df.set_index('Model')
        
        # Select key metrics for comparison
        key_metrics = ['MAPE', 'RMSE', 'R2', 'Directional_Accuracy', 'Total_Return_%', 'Sharpe_Ratio']
        comparison_df = df[key_metrics]
        
        return comparison_df
    
    def plot_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        dates: Optional[pd.DatetimeIndex] = None,
                        title: str = "Predictions vs Actual",
                        save_path: Optional[str] = None):
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dates: Optional date index
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Create x-axis
        if dates is not None:
            x = dates
            xlabel = 'Date'
        else:
            x = np.arange(len(y_true))
            xlabel = 'Time Steps'
        
        # Time series plot
        axes[0, 0].plot(x, y_true, label='Actual', alpha=0.7, linewidth=1)
        axes[0, 0].plot(x, y_pred, label='Predicted', alpha=0.7, linewidth=1)
        axes[0, 0].set_title(f'{title} - Time Series')
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 1].set_title('Predicted vs Actual')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[1, 0].plot(x, residuals, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuals')
        axes[1, 0].set_xlabel(xlabel)
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_model_comparison(self, 
                             comparison_df: pd.DataFrame,
                             save_path: Optional[str] = None):
        """
        Plot model comparison
        
        Args:
            comparison_df: DataFrame with model comparison
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics = comparison_df.columns
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                comparison_df[metric].plot(kind='bar', ax=axes[i], rot=45)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, output_dir: str):
        """
        Save evaluation results to CSV
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.metrics_history:
            results_df = pd.DataFrame(self.metrics_history)
            results_path = os.path.join(output_dir, 'evaluation_results.csv')
            results_df.to_csv(results_path, index=False)
            logger.info(f"Evaluation results saved to {results_path}")

class BaselineModels:
    """Simple baseline models for comparison"""
    
    @staticmethod
    def naive_forecast(y_train: np.ndarray, steps: int) -> np.ndarray:
        """Last value forecast"""
        return np.full(steps, y_train[-1])
    
    @staticmethod
    def moving_average_forecast(y_train: np.ndarray, steps: int, window: int = 5) -> np.ndarray:
        """Moving average forecast"""
        if len(y_train) < window:
            return np.full(steps, np.mean(y_train))
        
        ma_value = np.mean(y_train[-window:])
        return np.full(steps, ma_value)
    
    @staticmethod
    def linear_trend_forecast(y_train: np.ndarray, steps: int) -> np.ndarray:
        """Linear trend forecast"""
        x = np.arange(len(y_train))
        slope, intercept, _, _, _ = stats.linregress(x, y_train)
        
        future_x = np.arange(len(y_train), len(y_train) + steps)
        predictions = slope * future_x + intercept
        
        return predictions

def main():
    """Example usage"""
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic true values with trend
    t = np.linspace(0, 4*np.pi, n_samples)
    y_true = 100 + 10 * np.sin(t) + 0.1 * t + np.random.normal(0, 2, n_samples)
    
    # Create synthetic predictions (with some error)
    y_pred = y_true + np.random.normal(0, 3, n_samples)
    
    # Create date index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_true, y_pred, "Test GAN Model")
    
    # Plot results
    evaluator.plot_predictions(y_true, y_pred, dates, "GAN Model Evaluation")
    
    # Test baseline models
    train_size = int(0.8 * n_samples)
    y_train = y_true[:train_size]
    y_test = y_true[train_size:]
    test_steps = len(y_test)
    
    # Generate baseline predictions
    naive_pred = BaselineModels.naive_forecast(y_train, test_steps)
    ma_pred = BaselineModels.moving_average_forecast(y_train, test_steps)
    trend_pred = BaselineModels.linear_trend_forecast(y_train, test_steps)
    
    # Evaluate baselines
    naive_metrics = evaluator.evaluate_model(y_test, naive_pred, "Naive Baseline")
    ma_metrics = evaluator.evaluate_model(y_test, ma_pred, "Moving Average Baseline")
    trend_metrics = evaluator.evaluate_model(y_test, trend_pred, "Linear Trend Baseline")
    
    # Compare all models
    all_results = [metrics, naive_metrics, ma_metrics, trend_metrics]
    comparison = evaluator.compare_models(all_results)
    print("\nModel Comparison:")
    print(comparison)
    
    # Plot comparison
    evaluator.plot_model_comparison(comparison)

if __name__ == "__main__":
    main()