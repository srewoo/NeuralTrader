"""
Sentiment Analysis for Financial News
Advanced weighted lexicon-based analyzer with negation handling,
bigram detection, and intensity modifiers.
"""

from typing import Dict, Any, List
import logging
import math
import re

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Advanced financial sentiment analyzer.

    Features:
    - 300+ weighted lexicon words organized by category
    - Negation handling with a 3-word look-ahead window
    - Bigram detection (scanned before unigrams to avoid double-counting)
    - Intensity modifiers that multiply adjacent sentiment weights
    - Logarithmic confidence formula
    - Weighted scoring (pos-neg) / (pos+neg)
    """

    # ------------------------------------------------------------------
    # POSITIVE WORDS  (word -> weight)
    # Categories: earnings, technical, corporate, analyst, macro
    # Weights: 0.5 mild, 1.0 standard, 1.5 strong, 2.0 extreme
    # ------------------------------------------------------------------
    POSITIVE_WORDS: Dict[str, float] = {
        # --- earnings ---
        "beat": 1.5,
        "beats": 1.5,
        "beaten": 1.5,
        "revenue_growth": 1.5,
        "profit": 1.0,
        "profits": 1.0,
        "profitable": 1.0,
        "profitability": 1.0,
        "profit_surge": 2.0,
        "eps_beat": 2.0,
        "margin_expansion": 1.5,
        "record_revenue": 2.0,
        "earnings": 0.5,
        "outperformed": 1.5,
        "exceeded": 1.5,
        "exceeds": 1.5,
        "surpassed": 1.5,
        "topped": 1.0,
        "top_line": 0.5,
        "blowout": 2.0,
        "windfall": 1.5,
        "dividend": 0.5,
        "dividends": 0.5,
        "yield": 0.5,

        # --- technical ---
        "breakout": 1.5,
        "accumulation": 1.0,
        "golden_cross": 2.0,
        "support_held": 1.0,
        "new_highs": 1.5,
        "momentum": 1.0,
        "recovery": 1.0,
        "recover": 1.0,
        "recovered": 1.0,
        "rally": 1.5,
        "rallies": 1.5,
        "rallied": 1.5,
        "rebound": 1.0,
        "rebounds": 1.0,
        "rebounded": 1.0,
        "uptrend": 1.0,
        "bullish": 1.5,
        "bull": 1.0,
        "breakaway": 1.0,
        "uptick": 0.5,
        "surge": 1.5,
        "surges": 1.5,
        "surging": 1.5,
        "soar": 1.5,
        "soars": 1.5,
        "soaring": 1.5,
        "spike": 1.0,
        "spiking": 1.0,
        "jump": 1.0,
        "jumps": 1.0,
        "jumped": 1.0,
        "climb": 1.0,
        "climbs": 1.0,
        "climbed": 1.0,
        "gain": 1.0,
        "gains": 1.0,
        "gained": 1.0,
        "rise": 1.0,
        "rises": 1.0,
        "rising": 1.0,
        "rose": 1.0,
        "advance": 1.0,
        "advances": 1.0,
        "advancing": 1.0,
        "higher": 0.5,
        "highest": 1.0,
        "high": 0.5,
        "up": 0.5,
        "record": 1.0,
        "outpace": 1.0,
        "outpacing": 1.0,

        # --- corporate ---
        "acquisition": 1.0,
        "acquire": 1.0,
        "acquired": 1.0,
        "merger": 1.0,
        "buyback": 1.0,
        "repurchase": 1.0,
        "expansion": 1.0,
        "expanding": 1.0,
        "partnership": 1.0,
        "collaboration": 0.5,
        "launch": 1.0,
        "launched": 1.0,
        "launches": 1.0,
        "innovation": 1.0,
        "innovative": 1.0,
        "patent": 0.5,
        "approval": 1.0,
        "approved": 1.0,
        "fda_approval": 2.0,
        "ipo": 0.5,
        "spinoff": 0.5,
        "restructuring": 0.5,
        "turnaround": 1.5,
        "boost": 1.0,
        "boosts": 1.0,
        "boosted": 1.0,
        "breakthrough": 1.5,
        "success": 1.0,
        "successful": 1.0,
        "improve": 1.0,
        "improves": 1.0,
        "improved": 1.0,
        "improvement": 1.0,
        "growth": 1.0,
        "grows": 1.0,
        "growing": 1.0,
        "grew": 1.0,
        "strong": 1.0,
        "strength": 1.0,
        "strengthen": 1.0,
        "strengthened": 1.0,
        "positive": 0.5,
        "optimistic": 1.0,
        "optimism": 1.0,
        "confidence": 0.5,
        "confident": 0.5,

        # --- analyst ---
        "upgrade": 1.5,
        "upgrades": 1.5,
        "upgraded": 1.5,
        "outperform": 1.5,
        "overweight": 1.0,
        "price_target_raised": 1.5,
        "strong_buy": 2.0,
        "buy": 1.0,
        "accumulate": 1.0,
        "reiterate_buy": 1.0,
        "initiates_buy": 1.0,
        "above_consensus": 1.0,
        "recommend": 0.5,
        "recommended": 0.5,
        "favorable": 1.0,

        # --- macro ---
        "rate_cut": 1.5,
        "stimulus": 1.0,
        "gdp_growth": 1.0,
        "reform": 0.5,
        "inflow": 1.0,
        "inflows": 1.0,
        "easing": 1.0,
        "dovish": 1.0,
        "accommodative": 1.0,
        "employment": 0.5,
        "hiring": 0.5,
        "surplus": 0.5,
        "prosperity": 1.0,
        "boom": 1.5,
        "booming": 1.5,
        "stable": 0.5,
        "stability": 0.5,
    }

    # ------------------------------------------------------------------
    # NEGATIVE WORDS  (word -> weight)
    # ------------------------------------------------------------------
    NEGATIVE_WORDS: Dict[str, float] = {
        # --- earnings ---
        "miss": 1.5,
        "misses": 1.5,
        "missed": 1.5,
        "revenue_decline": 1.5,
        "profit_warning": 2.0,
        "eps_miss": 2.0,
        "margin_compression": 1.5,
        "loss": 1.0,
        "losses": 1.0,
        "losing": 1.0,
        "lost": 1.0,
        "shortfall": 1.5,
        "underperformed": 1.0,
        "disappointing": 1.0,
        "disappointed": 1.0,
        "disappointment": 1.0,
        "below_expectations": 1.5,
        "write_down": 1.0,
        "writeoff": 1.0,
        "impairment": 1.0,
        "deficit": 1.0,

        # --- technical ---
        "breakdown": 1.5,
        "distribution": 1.0,
        "death_cross": 2.0,
        "support_broken": 1.5,
        "crash": 2.0,
        "crashes": 2.0,
        "crashed": 2.0,
        "plunge": 2.0,
        "plunges": 2.0,
        "plunged": 2.0,
        "plummets": 2.0,
        "plummeted": 2.0,
        "selloff": 1.5,
        "sell_off": 1.5,
        "correction": 1.0,
        "downtrend": 1.0,
        "bearish": 1.5,
        "bear": 1.0,
        "slump": 1.5,
        "slumps": 1.5,
        "slumped": 1.5,
        "tumble": 1.5,
        "tumbles": 1.5,
        "tumbled": 1.5,
        "tank": 1.5,
        "tanked": 1.5,
        "tanking": 1.5,
        "collapse": 2.0,
        "collapsed": 2.0,
        "decline": 1.0,
        "declines": 1.0,
        "declining": 1.0,
        "declined": 1.0,
        "drop": 1.0,
        "drops": 1.0,
        "dropped": 1.0,
        "dropping": 1.0,
        "fall": 1.0,
        "falls": 1.0,
        "falling": 1.0,
        "fell": 1.0,
        "lower": 0.5,
        "lowest": 1.0,
        "low": 0.5,
        "down": 0.5,
        "sink": 1.0,
        "sinks": 1.0,
        "sinking": 1.0,
        "sank": 1.0,
        "erode": 1.0,
        "eroded": 1.0,
        "erosion": 1.0,
        "volatile": 0.5,
        "volatility": 0.5,
        "rout": 1.5,

        # --- corporate ---
        "layoff": 1.5,
        "layoffs": 1.5,
        "lawsuit": 1.0,
        "litigation": 1.0,
        "fraud": 2.0,
        "fraudulent": 2.0,
        "scandal": 2.0,
        "bankruptcy": 2.0,
        "bankrupt": 2.0,
        "insolvent": 2.0,
        "insolvency": 2.0,
        "default": 2.0,
        "defaults": 2.0,
        "defaulted": 2.0,
        "debt_crisis": 2.0,
        "delisted": 1.5,
        "delisting": 1.5,
        "recall": 1.0,
        "recalled": 1.0,
        "investigation": 1.0,
        "investigated": 1.0,
        "penalty": 1.0,
        "fine": 1.0,
        "fined": 1.0,
        "violation": 1.0,
        "breach": 1.0,
        "shutdown": 1.5,
        "closure": 1.0,
        "restructuring_charges": 1.0,
        "fail": 1.5,
        "fails": 1.5,
        "failed": 1.5,
        "failure": 1.5,
        "trouble": 1.0,
        "troubled": 1.0,
        "warning": 1.0,
        "warns": 1.0,
        "warned": 1.0,

        # --- analyst ---
        "downgrade": 1.5,
        "downgrades": 1.5,
        "downgraded": 1.5,
        "underperform": 1.5,
        "underweight": 1.0,
        "price_target_cut": 1.5,
        "sell": 1.0,
        "reduce": 0.5,
        "below_consensus": 1.0,
        "overvalued": 1.0,
        "negative_outlook": 1.5,
        "caution": 0.5,
        "cautious": 0.5,
        "skeptical": 0.5,

        # --- macro ---
        "rate_hike": 0.5,
        "recession": 2.0,
        "recessionary": 2.0,
        "inflation": 1.0,
        "inflationary": 1.0,
        "stagflation": 1.5,
        "outflow": 1.0,
        "outflows": 1.0,
        "sanctions": 1.0,
        "tariff": 1.0,
        "tariffs": 1.0,
        "hawkish": 0.5,
        "tightening": 0.5,
        "unemployment": 1.0,
        "debt": 0.5,
        "bubble": 1.0,
        "contagion": 1.5,
        "panic": 1.5,
        "fear": 1.0,
        "fears": 1.0,
        "concern": 0.5,
        "concerns": 0.5,
        "worried": 0.5,
        "worry": 0.5,
        "worries": 0.5,
        "risk": 0.5,
        "risks": 0.5,
        "risky": 0.5,
        "crisis": 1.5,
        "weak": 1.0,
        "weakness": 1.0,
        "weaken": 1.0,
        "weakened": 1.0,
        "pessimistic": 1.0,
        "pessimism": 1.0,
        "negative": 0.5,
        "uncertainty": 0.5,
        "uncertain": 0.5,
        "downturn": 1.5,
        "stagnation": 1.0,
        "headwinds": 0.5,
    }

    # ------------------------------------------------------------------
    # NEUTRAL WORDS  (word -> weight, weight is unused but kept for
    #                 consistency; counted for reporting)
    # ------------------------------------------------------------------
    NEUTRAL_WORDS: Dict[str, float] = {
        "unchanged": 0.0,
        "flat": 0.0,
        "steady": 0.0,
        "maintain": 0.0,
        "maintains": 0.0,
        "maintained": 0.0,
        "hold": 0.0,
        "holds": 0.0,
        "holding": 0.0,
        "neutral": 0.0,
        "mixed": 0.0,
        "sideways": 0.0,
        "consolidation": 0.0,
        "consolidating": 0.0,
        "range_bound": 0.0,
        "rangebound": 0.0,
        "inline": 0.0,
        "expected": 0.0,
        "as_expected": 0.0,
        "consensus": 0.0,
        "unchanged": 0.0,
        "balanced": 0.0,
        "moderate": 0.0,
    }

    # ------------------------------------------------------------------
    # NEGATION
    # ------------------------------------------------------------------
    NEGATION_WORDS = {
        "not", "no", "never", "don't", "doesn't", "didn't", "won't",
        "can't", "cannot", "couldn't", "hardly", "barely", "scarcely",
        "neither", "nor", "without", "isn't", "aren't", "wasn't",
        "weren't", "shouldn't", "wouldn't",
    }
    NEGATION_WINDOW = 3

    # ------------------------------------------------------------------
    # BIGRAMS  (tuple of two lowercased words) -> sentiment weight
    # Scanned BEFORE single-word pass. Consumed indices are skipped.
    # ------------------------------------------------------------------
    BIGRAMS: Dict[tuple, float] = {
        ("strong", "buy"): 2.0,
        ("market", "crash"): -2.0,
        ("earnings", "beat"): 1.5,
        ("earnings", "miss"): -1.5,
        ("earnings", "surprise"): 1.0,
        ("profit", "warning"): -1.5,
        ("profit", "surge"): 2.0,
        ("profit", "growth"): 1.0,
        ("debt", "crisis"): -1.5,
        ("debt", "default"): -2.0,
        ("short", "squeeze"): 1.0,
        ("short", "selling"): -0.5,
        ("bear", "market"): -1.5,
        ("bull", "run"): 1.5,
        ("bull", "market"): 1.5,
        ("rate", "hike"): -0.5,
        ("rate", "cut"): 1.0,
        ("rate", "increase"): -0.5,
        ("price", "target"): 0.5,
        ("price", "cut"): -1.0,
        ("golden", "cross"): 1.5,
        ("death", "cross"): -1.5,
        ("all", "time"): 1.0,
        ("record", "high"): 1.5,
        ("record", "low"): -1.5,
        ("record", "revenue"): 1.5,
        ("margin", "expansion"): 1.0,
        ("margin", "compression"): -1.0,
        ("revenue", "growth"): 1.0,
        ("revenue", "decline"): -1.0,
        ("revenue", "miss"): -1.5,
        ("guidance", "raised"): 1.5,
        ("guidance", "lowered"): -1.5,
        ("guidance", "cut"): -1.5,
        ("share", "buyback"): 1.0,
        ("stock", "split"): 0.5,
        ("hostile", "takeover"): -0.5,
        ("credit", "downgrade"): -1.5,
        ("credit", "upgrade"): 1.5,
        ("supply", "shortage"): -1.0,
        ("supply", "chain"): -0.5,
        ("insider", "buying"): 1.0,
        ("insider", "selling"): -1.0,
        ("cash", "flow"): 0.5,
        ("going", "concern"): -2.0,
        ("profit", "taking"): -0.5,
        ("market", "rally"): 1.5,
        ("market", "correction"): -1.0,
        ("new", "highs"): 1.5,
        ("new", "lows"): -1.5,
        ("strong", "results"): 1.5,
        ("weak", "results"): -1.5,
        ("better", "expected"): 1.0,
        ("worse", "expected"): -1.0,
        ("above", "expectations"): 1.0,
        ("below", "expectations"): -1.0,
        ("top", "pick"): 1.5,
    }

    # ------------------------------------------------------------------
    # INTENSITY MODIFIERS  (modifier_word -> multiplier)
    # When one of these immediately precedes a sentiment word, the
    # sentiment word's weight is multiplied by the modifier value.
    # ------------------------------------------------------------------
    INTENSITY_MODIFIERS: Dict[str, float] = {
        "extremely": 2.0,
        "very": 1.5,
        "highly": 1.5,
        "significantly": 1.5,
        "substantially": 1.5,
        "dramatically": 1.5,
        "sharply": 1.5,
        "massively": 2.0,
        "tremendously": 2.0,
        "remarkably": 1.5,
        "exceptionally": 1.5,
        "incredibly": 1.5,
        "notably": 1.3,
        "particularly": 1.3,
        "especially": 1.3,
        "slightly": 0.5,
        "marginally": 0.5,
        "somewhat": 0.7,
        "moderately": 0.8,
        "mildly": 0.5,
        "barely": 0.5,
        "a_bit": 0.5,
        "relatively": 0.7,
    }

    def __init__(self):
        """Initialize sentiment analyzer."""
        pass

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using weighted lexicon, bigrams,
        negation handling, and intensity modifiers.

        Args:
            text: Text to analyze.

        Returns:
            Dict with keys: sentiment, score, label, confidence,
            positive_count, negative_count, neutral_count,
            positive_words, negative_words.
        """
        if not text:
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

        # Tokenize
        text_lower = text.lower()
        words = re.findall(r"\b[\w']+\b", text_lower)
        n = len(words)

        # Tracking
        consumed: set = set()          # indices consumed by bigrams
        weighted_positive: float = 0.0
        weighted_negative: float = 0.0
        positive_count: int = 0
        negative_count: int = 0
        neutral_count: int = 0
        positive_found: List[str] = []
        negative_found: List[str] = []

        # ---- Phase 1: bigram scan ----
        for i in range(n - 1):
            pair = (words[i], words[i + 1])
            if pair in self.BIGRAMS:
                weight = self.BIGRAMS[pair]

                # Check for negation in the two words before the bigram
                negated = False
                for j in range(max(0, i - self.NEGATION_WINDOW), i):
                    if words[j] in self.NEGATION_WORDS:
                        negated = True
                        break

                # Check for intensity modifier immediately before
                modifier = 1.0
                if i > 0 and words[i - 1] in self.INTENSITY_MODIFIERS:
                    modifier = self.INTENSITY_MODIFIERS[words[i - 1]]

                effective_weight = weight * modifier
                if negated:
                    effective_weight = -effective_weight

                bigram_str = f"{words[i]}_{words[i + 1]}"
                if effective_weight > 0:
                    weighted_positive += effective_weight
                    positive_count += 1
                    positive_found.append(bigram_str)
                elif effective_weight < 0:
                    weighted_negative += abs(effective_weight)
                    negative_count += 1
                    negative_found.append(bigram_str)

                consumed.add(i)
                consumed.add(i + 1)

        # ---- Phase 2: build negation map for single-word pass ----
        negation_active: List[bool] = [False] * n
        for i, w in enumerate(words):
            if w in self.NEGATION_WORDS:
                for j in range(i + 1, min(i + 1 + self.NEGATION_WINDOW, n)):
                    negation_active[j] = True

        # ---- Phase 3: single-word pass ----
        for i, w in enumerate(words):
            if i in consumed:
                continue

            # Determine intensity modifier from preceding word
            modifier = 1.0
            if i > 0 and words[i - 1] in self.INTENSITY_MODIFIERS and (i - 1) not in consumed:
                modifier = self.INTENSITY_MODIFIERS[words[i - 1]]

            negated = negation_active[i]

            if w in self.POSITIVE_WORDS:
                weight = self.POSITIVE_WORDS[w] * modifier
                if negated:
                    # Positive word negated -> treat as negative
                    weighted_negative += weight
                    negative_count += 1
                    negative_found.append(w)
                else:
                    weighted_positive += weight
                    positive_count += 1
                    positive_found.append(w)

            elif w in self.NEGATIVE_WORDS:
                weight = self.NEGATIVE_WORDS[w] * modifier
                if negated:
                    # Negative word negated -> treat as positive
                    weighted_positive += weight
                    positive_count += 1
                    positive_found.append(w)
                else:
                    weighted_negative += weight
                    negative_count += 1
                    negative_found.append(w)

            elif w in self.NEUTRAL_WORDS:
                neutral_count += 1

        # ---- Scoring ----
        total_weight = weighted_positive + weighted_negative
        if total_weight > 0:
            score = (weighted_positive - weighted_negative) / total_weight
        else:
            score = 0.0

        # Clamp to [-1, 1] (should already be, but guard)
        score = max(-1.0, min(1.0, score))

        # ---- Confidence (log-based) ----
        total_sentiment_words = positive_count + negative_count + neutral_count
        if total_sentiment_words > 0:
            confidence = min(math.log(1 + total_sentiment_words) / math.log(21), 1.0)
        else:
            confidence = 0.0

        # ---- Label ----
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        return {
            "sentiment": label,
            "label": label,
            "score": round(score, 3),
            "confidence": round(confidence, 3),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_words": positive_found[:10],
            "negative_words": negative_found[:10],
        }

    # ------------------------------------------------------------------
    # Article-level analysis
    # ------------------------------------------------------------------
    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a news article.
        Title is weighted 2x relative to description.

        Args:
            article: Article dict with 'title' and optionally 'description'.

        Returns:
            Copy of article with 'sentiment' key added.
        """
        title = article.get("title", "")
        description = article.get("description", "")

        title_sentiment = self.analyze_text(title)
        desc_sentiment = self.analyze_text(description)

        # Combined score: title weighted 2x
        combined_score = (title_sentiment["score"] * 2 + desc_sentiment["score"]) / 3

        if combined_score > 0.2:
            overall_sentiment = "positive"
        elif combined_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        combined_confidence = (title_sentiment["confidence"] + desc_sentiment["confidence"]) / 2

        article_with_sentiment = article.copy()
        article_with_sentiment["sentiment"] = {
            "overall": overall_sentiment,
            "score": round(combined_score, 3),
            "confidence": round(combined_confidence, 3),
            "title_sentiment": title_sentiment["sentiment"],
            "description_sentiment": desc_sentiment["sentiment"],
            "positive_words": list(set(
                title_sentiment["positive_words"] + desc_sentiment["positive_words"]
            ))[:5],
            "negative_words": list(set(
                title_sentiment["negative_words"] + desc_sentiment["negative_words"]
            ))[:5],
        }

        return article_with_sentiment

    def analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a list of articles.

        Args:
            articles: List of article dicts.

        Returns:
            List of articles with sentiment analysis attached.
        """
        return [self.analyze_article(article) for article in articles]

    def get_aggregate_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate sentiment across multiple analyzed articles.

        Args:
            articles: List of articles that already have 'sentiment' key.

        Returns:
            Aggregate sentiment summary dict.
        """
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "sentiment_distribution": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                },
                "total_articles": 0,
            }

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_score = 0.0

        for article in articles:
            sentiment_data = article.get("sentiment", {})
            overall = sentiment_data.get("overall", "neutral")
            score = sentiment_data.get("score", 0.0)

            sentiment_counts[overall] += 1
            total_score += score

        avg_score = total_score / len(articles)

        if avg_score > 0.2:
            overall_sentiment = "positive"
        elif avg_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        count = len(articles)
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": round(avg_score, 3),
            "sentiment_distribution": sentiment_counts,
            "total_articles": count,
            "positive_percentage": round((sentiment_counts["positive"] / count) * 100, 1),
            "negative_percentage": round((sentiment_counts["negative"] / count) * 100, 1),
            "neutral_percentage": round((sentiment_counts["neutral"] / count) * 100, 1),
        }


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------
_sentiment_analyzer_instance = None


def get_sentiment_analyzer():
    """Get or create global sentiment analyzer instance. Prefers FinBERT if available."""
    global _sentiment_analyzer_instance
    if _sentiment_analyzer_instance is None:
        try:
            from news.finbert_sentiment import get_finbert_analyzer
            _sentiment_analyzer_instance = get_finbert_analyzer()
            logger.info("Using FinBERT sentiment analyzer")
        except (ImportError, Exception):
            _sentiment_analyzer_instance = SentimentAnalyzer()
            logger.info("Using lexicon-based sentiment analyzer")
    return _sentiment_analyzer_instance
