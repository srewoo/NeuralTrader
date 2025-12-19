"""
Advanced Sentiment Analysis using Transformer Models
Uses FinBERT and other financial domain-specific models
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio

logger = logging.getLogger(__name__)


class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis using transformer models
    Specifically tuned for financial text
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model name (default: FinBERT)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialized = False

        # Try to lazy load the model
        self._load_model()

    def _load_model(self):
        """Load the sentiment model (lazy loading)"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading sentiment model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            self.torch = torch
            self._initialized = True

            logger.info(f"Sentiment model loaded successfully on {self.device}")

        except ImportError:
            logger.warning(
                "Transformers not installed. Install with: pip install transformers torch"
            )
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self._initialized = False

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment scores and label
        """
        if not self._initialized:
            return self._fallback_sentiment(text)

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Run inference (no gradient computation needed)
            with self.torch.no_grad():
                outputs = await asyncio.to_thread(self.model, **inputs)

            # Get probabilities
            probs = self.torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]

            # FinBERT outputs: [positive, negative, neutral]
            sentiment_scores = {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2])
            }

            # Determine dominant sentiment
            max_label = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[max_label]

            return {
                "sentiment": max_label,
                "confidence": confidence,
                "scores": sentiment_scores,
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment(text)

    async def analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple articles

        Args:
            articles: List of article dicts with 'title' and 'description'

        Returns:
            Articles with added 'sentiment' field
        """
        analyzed_articles = []

        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"

            sentiment = await self.analyze_text(text)

            # Add sentiment to article
            article_copy = article.copy()
            article_copy['sentiment'] = sentiment

            analyzed_articles.append(article_copy)

        return analyzed_articles

    async def aggregate_sentiment(
        self,
        articles: List[Dict[str, Any]],
        method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Aggregate sentiment across multiple articles

        Args:
            articles: List of articles (already analyzed)
            method: 'weighted' (by confidence) or 'simple' (average)

        Returns:
            Aggregated sentiment scores
        """
        if not articles:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "article_count": 0
            }

        if method == "weighted":
            # Weight by confidence
            total_weight = 0
            weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            for article in articles:
                sentiment = article.get('sentiment', {})
                confidence = sentiment.get('confidence', 0.0)
                scores = sentiment.get('scores', {})

                total_weight += confidence

                for key in weighted_scores:
                    weighted_scores[key] += scores.get(key, 0.0) * confidence

            # Normalize
            if total_weight > 0:
                for key in weighted_scores:
                    weighted_scores[key] /= total_weight
            else:
                weighted_scores = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

        else:  # simple average
            weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            for article in articles:
                scores = article.get('sentiment', {}).get('scores', {})
                for key in weighted_scores:
                    weighted_scores[key] += scores.get(key, 0.0)

            # Average
            for key in weighted_scores:
                weighted_scores[key] /= len(articles)

        # Determine dominant sentiment
        max_label = max(weighted_scores, key=weighted_scores.get)
        confidence = weighted_scores[max_label]

        return {
            "sentiment": max_label,
            "confidence": confidence,
            "scores": weighted_scores,
            "article_count": len(articles),
            "method": method
        }

    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple fallback sentiment analysis using keyword matching
        Used when transformer model is not available
        """
        text_lower = text.lower()

        # Simple keyword-based sentiment
        positive_words = [
            'bullish', 'growth', 'profit', 'gain', 'surge', 'rally', 'up',
            'positive', 'strong', 'outperform', 'beat', 'exceed', 'success',
            'increase', 'rise', 'high', 'boost', 'improve', 'win'
        ]

        negative_words = [
            'bearish', 'loss', 'decline', 'drop', 'fall', 'down', 'negative',
            'weak', 'underperform', 'miss', 'fail', 'decrease', 'low', 'risk',
            'concern', 'warning', 'cut', 'reduce', 'sell', 'crash'
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count

        if total == 0:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "model": "fallback_keyword"
            }

        positive_score = positive_count / total if total > 0 else 0.33
        negative_score = negative_count / total if total > 0 else 0.33
        neutral_score = 1.0 - positive_score - negative_score

        scores = {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }

        max_label = max(scores, key=scores.get)

        return {
            "sentiment": max_label,
            "confidence": scores[max_label],
            "scores": scores,
            "model": "fallback_keyword"
        }

    async def analyze_with_aspects(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment with aspect-based analysis
        Identifies sentiment for different aspects (price, management, product, etc.)

        Args:
            text: Text to analyze

        Returns:
            Dict with overall and aspect-level sentiment
        """
        # Get overall sentiment
        overall = await self.analyze_text(text)

        # Extract aspects (simplified version)
        aspects = {
            "price": ["price", "valuation", "expensive", "cheap", "cost"],
            "management": ["ceo", "management", "leadership", "executive"],
            "product": ["product", "service", "innovation", "quality"],
            "earnings": ["earnings", "revenue", "profit", "sales"],
            "outlook": ["outlook", "forecast", "guidance", "future"]
        }

        aspect_sentiments = {}
        text_lower = text.lower()

        for aspect_name, keywords in aspects.items():
            # Check if aspect is mentioned
            if any(keyword in text_lower for keyword in keywords):
                # Extract sentences containing aspect keywords
                sentences = text.split('.')
                aspect_sentences = [
                    s for s in sentences
                    if any(keyword in s.lower() for keyword in keywords)
                ]

                if aspect_sentences:
                    # Analyze sentiment of aspect-specific text
                    aspect_text = '. '.join(aspect_sentences)
                    aspect_sentiment = await self.analyze_text(aspect_text)
                    aspect_sentiments[aspect_name] = aspect_sentiment

        return {
            "overall": overall,
            "aspects": aspect_sentiments,
            "text_length": len(text)
        }


# Singleton instance
_analyzer_instance: Optional[AdvancedSentimentAnalyzer] = None


def get_advanced_sentiment_analyzer(model_name: str = "ProsusAI/finbert") -> AdvancedSentimentAnalyzer:
    """Get or create advanced sentiment analyzer instance"""
    global _analyzer_instance

    if _analyzer_instance is None:
        _analyzer_instance = AdvancedSentimentAnalyzer(model_name)

    return _analyzer_instance
