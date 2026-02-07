"""
FinBERT Sentiment Analysis for Financial News
Uses ProsusAI/finbert transformer model for accurate financial sentiment.
Falls back gracefully if torch/transformers not installed.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded model components
_model = None
_tokenizer = None
_model_loaded = False
_model_load_failed = False


def _load_model():
    """Lazy-load FinBERT model and tokenizer on first use."""
    global _model, _tokenizer, _model_loaded, _model_load_failed

    if _model_loaded or _model_load_failed:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Loading FinBERT model (ProsusAI/finbert)...")
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _model.eval()
        _model_loaded = True
        logger.info("FinBERT model loaded successfully")
    except ImportError:
        _model_load_failed = True
        logger.warning("torch/transformers not installed — FinBERT unavailable")
    except Exception as e:
        _model_load_failed = True
        logger.warning(f"FinBERT model loading failed: {e}")


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analyzer using ProsusAI/finbert.
    Same interface as the lexicon-based SentimentAnalyzer.
    """

    # FinBERT label mapping: index 0=positive, 1=negative, 2=neutral
    LABELS = ["positive", "negative", "neutral"]

    def __init__(self):
        """Initialize — model is lazy-loaded on first analyze_text call."""
        pass

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using FinBERT.
        Returns dict matching SentimentAnalyzer interface.
        """
        if not text or not text.strip():
            return self._empty_result()

        _load_model()

        if not _model_loaded:
            return self._empty_result()

        try:
            import torch

            inputs = _tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs_list = probs[0].tolist()
            pos_prob = probs_list[0]
            neg_prob = probs_list[1]
            neu_prob = probs_list[2]

            # Score: positive maps to +1, negative to -1, weighted by probability
            score = pos_prob - neg_prob  # Range: -1 to +1

            # Label based on highest probability
            max_idx = probs_list.index(max(probs_list))
            label = self.LABELS[max_idx]

            # Confidence: how dominant the winning class is
            confidence = max(probs_list)

            return {
                "sentiment": label,
                "label": label,
                "score": round(score, 3),
                "confidence": round(confidence, 3),
                "positive_count": 1 if label == "positive" else 0,
                "negative_count": 1 if label == "negative" else 0,
                "neutral_count": 1 if label == "neutral" else 0,
                "positive_words": [],  # FinBERT doesn't identify individual words
                "negative_words": [],
                "probabilities": {
                    "positive": round(pos_prob, 3),
                    "negative": round(neg_prob, 3),
                    "neutral": round(neu_prob, 3)
                }
            }

        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            return self._empty_result()

    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a news article.
        Title is weighted 2x relative to description.
        """
        title = article.get("title", "")
        description = article.get("description", article.get("content", ""))

        title_result = self.analyze_text(title)
        desc_result = self.analyze_text(description)

        # Combined score: title weighted 2x
        combined_score = (title_result["score"] * 2 + desc_result["score"]) / 3

        if combined_score > 0.2:
            overall = "positive"
        elif combined_score < -0.2:
            overall = "negative"
        else:
            overall = "neutral"

        combined_confidence = (title_result["confidence"] + desc_result["confidence"]) / 2

        article_copy = article.copy()
        article_copy["sentiment"] = {
            "overall": overall,
            "score": round(combined_score, 3),
            "confidence": round(combined_confidence, 3),
            "title_sentiment": title_result["label"],
            "description_sentiment": desc_result["label"],
            "positive_words": [],
            "negative_words": [],
        }
        return article_copy

    def analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of articles."""
        return [self.analyze_article(article) for article in articles]

    def get_aggregate_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate sentiment across analyzed articles."""
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "total_articles": 0,
                "positive_percentage": 0.0,
                "negative_percentage": 0.0,
                "neutral_percentage": 0.0,
            }

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_score = 0.0

        for article in articles:
            sent = article.get("sentiment", {})
            overall = sent.get("overall", "neutral")
            score = sent.get("score", 0.0)
            counts[overall] += 1
            total_score += score

        avg_score = total_score / len(articles)

        if avg_score > 0.2:
            overall_sentiment = "positive"
        elif avg_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        n = len(articles)
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": round(avg_score, 3),
            "sentiment_distribution": counts,
            "total_articles": n,
            "positive_percentage": round((counts["positive"] / n) * 100, 1),
            "negative_percentage": round((counts["negative"] / n) * 100, 1),
            "neutral_percentage": round((counts["neutral"] / n) * 100, 1),
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "sentiment": "neutral",
            "label": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_words": [],
            "negative_words": [],
        }


# Singleton
_finbert_analyzer: Optional[FinBERTSentimentAnalyzer] = None


def get_finbert_analyzer() -> FinBERTSentimentAnalyzer:
    """Get singleton FinBERT analyzer instance."""
    global _finbert_analyzer
    if _finbert_analyzer is None:
        _finbert_analyzer = FinBERTSentimentAnalyzer()
    return _finbert_analyzer
