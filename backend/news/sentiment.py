"""
Sentiment Analysis for Financial News
Analyzes sentiment of news articles
"""

from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of financial news articles
    """
    
    # Financial sentiment lexicon
    POSITIVE_WORDS = {
        'gain', 'gains', 'profit', 'profits', 'surge', 'surges', 'rally', 'rallies',
        'rise', 'rises', 'rising', 'rose', 'up', 'bullish', 'bull', 'growth', 'grows',
        'strong', 'strength', 'positive', 'optimistic', 'outperform', 'beat', 'beats',
        'success', 'successful', 'high', 'higher', 'highest', 'record', 'breakthrough',
        'improve', 'improves', 'improved', 'improvement', 'advance', 'advances', 'advancing',
        'boost', 'boosts', 'boosted', 'upgrade', 'upgrades', 'upgraded', 'soar', 'soars',
        'jump', 'jumps', 'jumped', 'climb', 'climbs', 'climbed', 'recovery', 'recover'
    }
    
    NEGATIVE_WORDS = {
        'loss', 'losses', 'lose', 'losing', 'lost', 'fall', 'falls', 'falling', 'fell',
        'drop', 'drops', 'dropped', 'decline', 'declines', 'declining', 'declined', 'down',
        'bearish', 'bear', 'weak', 'weakness', 'negative', 'pessimistic', 'underperform',
        'miss', 'misses', 'missed', 'fail', 'fails', 'failed', 'failure', 'low', 'lower',
        'lowest', 'crash', 'crashes', 'crashed', 'plunge', 'plunges', 'plunged', 'slump',
        'slumps', 'slumped', 'downgrade', 'downgrades', 'downgraded', 'concern', 'concerns',
        'worried', 'worry', 'worries', 'risk', 'risks', 'risky', 'crisis', 'trouble'
    }
    
    NEUTRAL_WORDS = {
        'stable', 'unchanged', 'flat', 'steady', 'maintain', 'maintains', 'maintained',
        'hold', 'holds', 'holding', 'neutral', 'mixed', 'sideways'
    }
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        pass
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        if not text:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "positive_words": [],
                "negative_words": []
            }
        
        # Clean and tokenize text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count sentiment words
        positive_found = [w for w in words if w in self.POSITIVE_WORDS]
        negative_found = [w for w in words if w in self.NEGATIVE_WORDS]
        neutral_found = [w for w in words if w in self.NEUTRAL_WORDS]
        
        positive_count = len(positive_found)
        negative_count = len(negative_found)
        neutral_count = len(neutral_found)
        
        # Calculate sentiment score (-1 to +1)
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            score = 0.0
            sentiment = "neutral"
            confidence = 0.0
        else:
            score = (positive_count - negative_count) / total_sentiment_words
            
            # Determine sentiment category
            if score > 0.2:
                sentiment = "positive"
            elif score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Calculate confidence based on number of sentiment words
            confidence = min(total_sentiment_words / 10.0, 1.0)  # Max at 10 words
        
        return {
            "sentiment": sentiment,
            "score": round(score, 3),
            "confidence": round(confidence, 3),
            "positive_words": positive_found[:5],  # Top 5
            "negative_words": negative_found[:5],
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count
        }
    
    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a news article
        
        Args:
            article: Article dictionary with title and description
            
        Returns:
            Article with sentiment analysis
        """
        # Combine title and description (title weighted more)
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Analyze title (weight 2x)
        title_sentiment = self.analyze_text(title)
        
        # Analyze description
        desc_sentiment = self.analyze_text(description)
        
        # Combined score (title weighted 2x)
        combined_score = (title_sentiment['score'] * 2 + desc_sentiment['score']) / 3
        
        # Determine overall sentiment
        if combined_score > 0.2:
            overall_sentiment = "positive"
        elif combined_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Combined confidence
        combined_confidence = (title_sentiment['confidence'] + desc_sentiment['confidence']) / 2
        
        # Add sentiment to article
        article_with_sentiment = article.copy()
        article_with_sentiment['sentiment'] = {
            "overall": overall_sentiment,
            "score": round(combined_score, 3),
            "confidence": round(combined_confidence, 3),
            "title_sentiment": title_sentiment['sentiment'],
            "description_sentiment": desc_sentiment['sentiment'],
            "positive_words": list(set(
                title_sentiment['positive_words'] + desc_sentiment['positive_words']
            ))[:5],
            "negative_words": list(set(
                title_sentiment['negative_words'] + desc_sentiment['negative_words']
            ))[:5]
        }
        
        return article_with_sentiment
    
    def analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple articles
        
        Args:
            articles: List of articles
            
        Returns:
            Articles with sentiment analysis
        """
        return [self.analyze_article(article) for article in articles]
    
    def get_aggregate_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get aggregate sentiment from multiple articles
        
        Args:
            articles: List of articles with sentiment
            
        Returns:
            Aggregate sentiment analysis
        """
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "sentiment_distribution": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0
                },
                "total_articles": 0
            }
        
        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_score = 0.0
        
        for article in articles:
            sentiment_data = article.get('sentiment', {})
            overall = sentiment_data.get('overall', 'neutral')
            score = sentiment_data.get('score', 0.0)
            
            sentiment_counts[overall] += 1
            total_score += score
        
        avg_score = total_score / len(articles)
        
        # Determine overall sentiment
        if avg_score > 0.2:
            overall_sentiment = "positive"
        elif avg_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": round(avg_score, 3),
            "sentiment_distribution": sentiment_counts,
            "total_articles": len(articles),
            "positive_percentage": round((sentiment_counts['positive'] / len(articles)) * 100, 1),
            "negative_percentage": round((sentiment_counts['negative'] / len(articles)) * 100, 1),
            "neutral_percentage": round((sentiment_counts['neutral'] / len(articles)) * 100, 1)
        }


# Global instance
_sentiment_analyzer_instance = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create global SentimentAnalyzer instance"""
    global _sentiment_analyzer_instance
    if _sentiment_analyzer_instance is None:
        _sentiment_analyzer_instance = SentimentAnalyzer()
    return _sentiment_analyzer_instance

