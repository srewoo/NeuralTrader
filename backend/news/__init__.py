"""
News & Sentiment Analysis System
Fetches financial news and analyzes sentiment
"""

from .sources import NewsAggregator
from .sentiment import SentimentAnalyzer

__all__ = ['NewsAggregator', 'SentimentAnalyzer']

