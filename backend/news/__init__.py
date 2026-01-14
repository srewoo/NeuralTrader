"""
News & Sentiment Analysis System
Fetches financial news and analyzes sentiment
"""

from .sources import NewsAggregator
from .sentiment import SentimentAnalyzer, get_sentiment_analyzer
from .advanced_sentiment import AdvancedSentimentAnalyzer, get_advanced_sentiment_analyzer

__all__ = [
    'NewsAggregator',
    'SentimentAnalyzer',
    'AdvancedSentimentAnalyzer',
    'get_sentiment_analyzer',
    'get_advanced_sentiment_analyzer',
    'get_default_sentiment_analyzer',
]


def get_default_sentiment_analyzer(use_finbert: bool = False):
    """
    Get the default sentiment analyzer.

    Args:
        use_finbert: If True, use FinBERT (advanced) - may crash on some macOS systems.
                     If False (default), use lexicon-based (safer, faster).

    Returns:
        Sentiment analyzer instance
    """
    if use_finbert:
        return get_advanced_sentiment_analyzer()
    else:
        return get_sentiment_analyzer()

