import tweepy
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataFetcher:
    def __init__(self, bearer_token: str, cache_file: str = 'twitter_cache.json'):
        """
        Initialize Twitter API client with v2 API
        
        Args:
            bearer_token: Twitter API Bearer Token
            cache_file: Local cache file for tweets
        """
        self.client = tweepy.Client(bearer_token=bearer_token)
        self.cache_file = cache_file
        
    def fetch_tweets(self, 
                    keywords: List[str] = ['AMZN', '$AMZN', 'Amazon stock'],
                    max_results: int = 100,
                    hours_back: int = 24) -> pd.DataFrame:
        """
        Fetch recent tweets containing specified keywords
        
        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of tweets to fetch
            hours_back: How many hours back to search
            
        Returns:
            DataFrame with tweet data
        """
        try:
            # Build search query
            query = ' OR '.join(keywords)
            query += ' -is:retweet lang:en'  # Exclude retweets, English only
            
            # Calculate start time
            start_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            logger.info(f"Fetching tweets with query: {query}")
            
            # Fetch tweets using API v2
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                max_results=min(max_results, 100),  # API limit per request
                start_time=start_time
            ).flatten(limit=max_results)
            
            tweet_data = []
            for tweet in tweets:
                tweet_dict = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'retweet_count': tweet.public_metrics['retweet_count'] if tweet.public_metrics else 0,
                    'like_count': tweet.public_metrics['like_count'] if tweet.public_metrics else 0,
                    'reply_count': tweet.public_metrics['reply_count'] if tweet.public_metrics else 0,
                }
                tweet_data.append(tweet_dict)
            
            df = pd.DataFrame(tweet_data)
            
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.sort_values('created_at').reset_index(drop=True)
                logger.info(f"Successfully fetched {len(df)} tweets")
                
                # Cache the data
                self._cache_tweets(df)
            else:
                logger.warning("No tweets found for the given criteria")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return self._load_cached_tweets()
    
    def _cache_tweets(self, df: pd.DataFrame) -> None:
        """Cache tweets to local file"""
        try:
            df.to_json(self.cache_file, orient='records', date_format='iso')
            logger.info(f"Cached {len(df)} tweets to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error caching tweets: {e}")
    
    def _load_cached_tweets(self) -> pd.DataFrame:
        """Load cached tweets if API fails"""
        try:
            df = pd.read_json(self.cache_file)
            df['created_at'] = pd.to_datetime(df['created_at'])
            logger.info(f"Loaded {len(df)} cached tweets")
            return df
        except Exception as e:
            logger.warning(f"No cached tweets available: {e}")
            return pd.DataFrame()
    
    def preprocess_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing of tweet text
        
        Args:
            df: DataFrame with tweet data
            
        Returns:
            DataFrame with preprocessed text
        """
        if df.empty:
            return df
            
        # Remove URLs, mentions, hashtags for cleaner sentiment analysis
        df = df.copy()
        df['clean_text'] = df['text'].str.replace(r'http\S+', '', regex=True)
        df['clean_text'] = df['clean_text'].str.replace(r'@\w+', '', regex=True)
        df['clean_text'] = df['clean_text'].str.replace(r'#\w+', '', regex=True)
        df['clean_text'] = df['clean_text'].str.replace(r'\n', ' ', regex=True)
        df['clean_text'] = df['clean_text'].str.strip()
        
        return df

def main():
    """Example usage"""
    # Note: You need to set your Twitter Bearer Token
    bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
    
    fetcher = TwitterDataFetcher(bearer_token)
    tweets_df = fetcher.fetch_tweets(max_results=50, hours_back=12)
    
    if not tweets_df.empty:
        processed_df = fetcher.preprocess_tweets(tweets_df)
        print(f"Fetched and processed {len(processed_df)} tweets")
        print(processed_df[['created_at', 'clean_text', 'like_count']].head())
    else:
        print("No tweets fetched")

if __name__ == "__main__":
    main()