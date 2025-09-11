import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analysis tools"""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial keywords that might indicate sentiment
        self.positive_keywords = {
            'bullish', 'buy', 'long', 'calls', 'growth', 'profit', 'earnings',
            'revenue', 'beat', 'exceed', 'strong', 'outperform', 'rally'
        }
        
        self.negative_keywords = {
            'bearish', 'sell', 'short', 'puts', 'loss', 'decline', 'drop',
            'fall', 'crash', 'correction', 'underperform', 'weak'
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
            
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}
            
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def calculate_financial_sentiment(self, text: str) -> float:
        """
        Calculate financial-specific sentiment score
        
        Args:
            text: Text to analyze
            
        Returns:
            Financial sentiment score (-1 to 1)
        """
        if not text:
            return 0.0
            
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_count = len(words.intersection(self.positive_keywords))
        negative_count = len(words.intersection(self.negative_keywords))
        
        if positive_count + negative_count == 0:
            return 0.0
            
        financial_score = (positive_count - negative_count) / (positive_count + negative_count)
        return financial_score
    
    def analyze_batch_sentiment(self, df: pd.DataFrame, text_column: str = 'clean_text') -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text to analyze
            
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or missing {text_column} column")
            return df
            
        logger.info(f"Analyzing sentiment for {len(df)} texts")
        
        result_df = df.copy()
        
        # Preprocess text
        result_df['preprocessed_text'] = result_df[text_column].apply(self.preprocess_text)
        
        # VADER sentiment analysis
        vader_results = result_df['preprocessed_text'].apply(self.analyze_sentiment_vader)
        result_df['vader_compound'] = [score['compound'] for score in vader_results]
        result_df['vader_pos'] = [score['pos'] for score in vader_results]
        result_df['vader_neu'] = [score['neu'] for score in vader_results]
        result_df['vader_neg'] = [score['neg'] for score in vader_results]
        
        # TextBlob sentiment analysis
        textblob_results = result_df['preprocessed_text'].apply(self.analyze_sentiment_textblob)
        result_df['textblob_polarity'] = [score['polarity'] for score in textblob_results]
        result_df['textblob_subjectivity'] = [score['subjectivity'] for score in textblob_results]
        
        # Financial sentiment
        result_df['financial_sentiment'] = result_df['preprocessed_text'].apply(
            self.calculate_financial_sentiment
        )
        
        # Combined sentiment score (weighted average)
        result_df['combined_sentiment'] = (
            0.4 * result_df['vader_compound'] +
            0.4 * result_df['textblob_polarity'] +
            0.2 * result_df['financial_sentiment']
        )
        
        logger.info("Sentiment analysis completed")
        return result_df
    
    def calculate_sentiment_index(self, 
                                df: pd.DataFrame, 
                                timestamp_col: str = 'created_at',
                                sentiment_col: str = 'combined_sentiment',
                                window_hours: int = 1) -> pd.DataFrame:
        """
        Calculate rolling sentiment index
        
        Args:
            df: DataFrame with sentiment data
            timestamp_col: Name of timestamp column
            sentiment_col: Name of sentiment column
            window_hours: Rolling window in hours
            
        Returns:
            DataFrame with sentiment index
        """
        if df.empty:
            return pd.DataFrame()
            
        logger.info(f"Calculating sentiment index with {window_hours}h rolling window")
        
        # Ensure datetime column
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index(timestamp_col)
        
        # Resample to hourly intervals and calculate sentiment metrics
        sentiment_index = df_indexed.resample('1H').agg({
            sentiment_col: ['mean', 'std', 'count'],
            'like_count': 'sum' if 'like_count' in df.columns else lambda x: 0,
            'retweet_count': 'sum' if 'retweet_count' in df.columns else lambda x: 0
        }).reset_index()
        
        # Flatten column names
        sentiment_index.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                 for col in sentiment_index.columns]
        
        # Rename columns for clarity
        sentiment_index.rename(columns={
            f'{sentiment_col}_mean': 'sentiment_mean',
            f'{sentiment_col}_std': 'sentiment_std',
            f'{sentiment_col}_count': 'tweet_count',
            'like_count_sum': 'total_likes',
            'retweet_count_sum': 'total_retweets'
        }, inplace=True)
        
        # Calculate rolling sentiment index
        sentiment_index['sentiment_index'] = sentiment_index['sentiment_mean'].rolling(
            window=window_hours, min_periods=1
        ).mean()
        
        # Calculate sentiment momentum (rate of change)
        sentiment_index['sentiment_momentum'] = sentiment_index['sentiment_index'].pct_change()
        
        # Fill NaN values
        sentiment_index = sentiment_index.fillna(0)
        
        logger.info(f"Generated sentiment index with {len(sentiment_index)} time periods")
        return sentiment_index

def main():
    """Example usage"""
    # Sample tweet data for testing
    sample_data = pd.DataFrame({
        'clean_text': [
            'Amazon stock is looking very bullish today with strong earnings',
            'AMZN might drop further, bearish sentiment in the market',
            'Great quarterly results from Amazon, expecting growth',
            'Concerned about Amazon delivery issues, might affect stock',
            'Amazon web services showing excellent performance'
        ],
        'created_at': pd.date_range('2024-01-01 10:00:00', periods=5, freq='1H'),
        'like_count': [10, 5, 15, 8, 12],
        'retweet_count': [3, 2, 7, 1, 4]
    })
    
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    sentiment_results = analyzer.analyze_batch_sentiment(sample_data)
    
    print("Sentiment Analysis Results:")
    print(sentiment_results[['clean_text', 'vader_compound', 'textblob_polarity', 
                           'financial_sentiment', 'combined_sentiment']].round(3))
    
    # Calculate sentiment index
    sentiment_index = analyzer.calculate_sentiment_index(sentiment_results)
    
    print("\nSentiment Index:")
    print(sentiment_index[['created_at', 'sentiment_mean', 'sentiment_index', 
                          'tweet_count', 'sentiment_momentum']].round(3))

if __name__ == "__main__":
    main()