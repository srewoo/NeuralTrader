"""
News & Sentiment Analysis System
Fetches financial news and analyzes sentiment using lexicon-based approach
"""

from .sources import NewsAggregator
from .sentiment import SentimentAnalyzer, get_sentiment_analyzer

__all__ = [
    'NewsAggregator',
    'SentimentAnalyzer',
    'get_sentiment_analyzer',
]
