"""
Unit Tests for News Sentiment Analysis
Tests for SentimentAnalyzer
"""

import pytest


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class"""

    def test_analyzer_initialization(self):
        """Test sentiment analyzer initializes correctly"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        assert analyzer is not None

    def test_positive_words_exist(self):
        """Test positive words dictionary"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        assert len(analyzer.POSITIVE_WORDS) > 0
        assert "gain" in analyzer.POSITIVE_WORDS
        assert "profit" in analyzer.POSITIVE_WORDS
        assert "bullish" in analyzer.POSITIVE_WORDS

    def test_negative_words_exist(self):
        """Test negative words dictionary"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        assert len(analyzer.NEGATIVE_WORDS) > 0
        assert "loss" in analyzer.NEGATIVE_WORDS
        assert "crash" in analyzer.NEGATIVE_WORDS
        assert "bearish" in analyzer.NEGATIVE_WORDS

    def test_neutral_words_exist(self):
        """Test neutral words dictionary"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        assert len(analyzer.NEUTRAL_WORDS) > 0
        assert "stable" in analyzer.NEUTRAL_WORDS
        assert "unchanged" in analyzer.NEUTRAL_WORDS


class TestAnalyzeText:
    """Tests for analyze_text method"""

    def test_analyze_empty_text(self):
        """Test analysis of empty text"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_text("")

        assert result["sentiment"] == "neutral"
        assert result["score"] == 0.0
        assert result["confidence"] == 0.0

    def test_analyze_positive_text(self, sample_positive_article):
        """Test analysis of positive text"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        text = sample_positive_article["title"] + " " + sample_positive_article["description"]
        result = analyzer.analyze_text(text)

        assert result["sentiment"] == "positive"
        assert result["score"] > 0
        assert len(result["positive_words"]) > 0

    def test_analyze_negative_text(self, sample_negative_article):
        """Test analysis of negative text"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        text = sample_negative_article["title"] + " " + sample_negative_article["description"]
        result = analyzer.analyze_text(text)

        assert result["sentiment"] == "negative"
        assert result["score"] < 0
        assert len(result["negative_words"]) > 0

    def test_analyze_neutral_text(self, sample_neutral_article):
        """Test analysis of neutral text"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        text = sample_neutral_article["title"] + " " + sample_neutral_article["description"]
        result = analyzer.analyze_text(text)

        assert result["sentiment"] == "neutral"
        assert -0.2 <= result["score"] <= 0.2

    def test_analyze_text_structure(self):
        """Test result structure"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_text("Stock gains profit record high")

        assert "sentiment" in result
        assert "score" in result
        assert "confidence" in result
        assert "positive_words" in result
        assert "negative_words" in result
        assert "positive_count" in result
        assert "negative_count" in result

    def test_score_range(self):
        """Test score is within expected range"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Test various texts
        texts = [
            "gain profit surge rally bullish",  # Very positive
            "loss crash plunge bearish decline",  # Very negative
            "stable unchanged steady flat"  # Neutral
        ]

        for text in texts:
            result = analyzer.analyze_text(text)
            assert -1 <= result["score"] <= 1

    def test_confidence_calculation(self):
        """Test confidence increases with more sentiment words"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Few sentiment words
        result1 = analyzer.analyze_text("Stock gains")

        # Many sentiment words
        result2 = analyzer.analyze_text("Stock gains profit surge rally high bullish strong record growth")

        assert result2["confidence"] > result1["confidence"]

    def test_case_insensitivity(self):
        """Test analysis is case-insensitive"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        result_lower = analyzer.analyze_text("gain profit")
        result_upper = analyzer.analyze_text("GAIN PROFIT")
        result_mixed = analyzer.analyze_text("Gain Profit")

        assert result_lower["sentiment"] == result_upper["sentiment"]
        assert result_lower["sentiment"] == result_mixed["sentiment"]


class TestAnalyzeArticle:
    """Tests for analyze_article method"""

    def test_analyze_positive_article(self, sample_positive_article):
        """Test analyzing positive article"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_article(sample_positive_article)

        assert "sentiment" in result
        assert result["sentiment"]["overall"] == "positive"
        assert result["sentiment"]["score"] > 0

    def test_analyze_negative_article(self, sample_negative_article):
        """Test analyzing negative article"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_article(sample_negative_article)

        assert result["sentiment"]["overall"] == "negative"
        assert result["sentiment"]["score"] < 0

    def test_article_preserves_original_data(self, sample_positive_article):
        """Test that original article data is preserved"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_article(sample_positive_article)

        assert result["title"] == sample_positive_article["title"]
        assert result["description"] == sample_positive_article["description"]
        assert result["source"] == sample_positive_article["source"]

    def test_article_sentiment_structure(self, sample_positive_article):
        """Test sentiment result structure in article"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_article(sample_positive_article)

        sentiment = result["sentiment"]
        assert "overall" in sentiment
        assert "score" in sentiment
        assert "confidence" in sentiment
        assert "title_sentiment" in sentiment
        assert "description_sentiment" in sentiment
        assert "positive_words" in sentiment
        assert "negative_words" in sentiment

    def test_title_weighted_more(self):
        """Test that title is weighted more than description"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Positive title, negative description
        article = {
            "title": "Stock Surges Rally Gains Profit",
            "description": "However some concerns about losses remain."
        }
        result = analyzer.analyze_article(article)

        # Title should pull score positive
        assert result["sentiment"]["overall"] == "positive" or result["sentiment"]["score"] > -0.2


class TestAnalyzeArticles:
    """Tests for analyze_articles method"""

    def test_analyze_multiple_articles(
        self, sample_positive_article, sample_negative_article, sample_neutral_article
    ):
        """Test analyzing multiple articles"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        articles = [sample_positive_article, sample_negative_article, sample_neutral_article]
        results = analyzer.analyze_articles(articles)

        assert len(results) == 3
        assert all("sentiment" in r for r in results)

    def test_analyze_empty_list(self):
        """Test analyzing empty list"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        results = analyzer.analyze_articles([])

        assert results == []


class TestAggregateSentiment:
    """Tests for get_aggregate_sentiment method"""

    def test_aggregate_empty_articles(self):
        """Test aggregate sentiment with no articles"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.get_aggregate_sentiment([])

        assert result["overall_sentiment"] == "neutral"
        assert result["average_score"] == 0.0
        assert result["total_articles"] == 0

    def test_aggregate_positive_articles(self, sample_positive_article):
        """Test aggregate sentiment with positive articles"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Create multiple positive articles with sentiment
        articles = [
            analyzer.analyze_article(sample_positive_article),
            analyzer.analyze_article(sample_positive_article),
            analyzer.analyze_article(sample_positive_article)
        ]

        result = analyzer.get_aggregate_sentiment(articles)

        assert result["overall_sentiment"] == "positive"
        assert result["average_score"] > 0
        assert result["total_articles"] == 3

    def test_aggregate_mixed_articles(
        self, sample_positive_article, sample_negative_article, sample_neutral_article
    ):
        """Test aggregate sentiment with mixed articles"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        articles = [
            analyzer.analyze_article(sample_positive_article),
            analyzer.analyze_article(sample_negative_article),
            analyzer.analyze_article(sample_neutral_article)
        ]

        result = analyzer.get_aggregate_sentiment(articles)

        assert result["total_articles"] == 3
        assert "sentiment_distribution" in result
        assert result["sentiment_distribution"]["positive"] >= 1
        assert result["sentiment_distribution"]["negative"] >= 1

    def test_aggregate_result_structure(self, sample_positive_article):
        """Test aggregate result structure"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        articles = [analyzer.analyze_article(sample_positive_article)]

        result = analyzer.get_aggregate_sentiment(articles)

        assert "overall_sentiment" in result
        assert "average_score" in result
        assert "sentiment_distribution" in result
        assert "total_articles" in result
        assert "positive_percentage" in result
        assert "negative_percentage" in result
        assert "neutral_percentage" in result

    def test_aggregate_percentages_sum_to_100(
        self, sample_positive_article, sample_negative_article, sample_neutral_article
    ):
        """Test that percentages sum to approximately 100"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        articles = [
            analyzer.analyze_article(sample_positive_article),
            analyzer.analyze_article(sample_negative_article),
            analyzer.analyze_article(sample_neutral_article)
        ]

        result = analyzer.get_aggregate_sentiment(articles)

        total = (
            result["positive_percentage"] +
            result["negative_percentage"] +
            result["neutral_percentage"]
        )

        assert abs(total - 100.0) < 0.5  # Allow small floating point error


class TestGlobalSentimentAnalyzer:
    """Tests for global sentiment analyzer instance"""

    def test_get_sentiment_analyzer_singleton(self):
        """Test singleton pattern for analyzer"""
        from news.sentiment import get_sentiment_analyzer

        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()

        assert analyzer1 is analyzer2

    def test_get_sentiment_analyzer_creates_instance(self):
        """Test that get_sentiment_analyzer creates instance"""
        from news.sentiment import get_sentiment_analyzer, SentimentAnalyzer

        analyzer = get_sentiment_analyzer()
        assert isinstance(analyzer, SentimentAnalyzer)


class TestEdgeCases:
    """Tests for edge cases in sentiment analysis"""

    def test_text_with_no_sentiment_words(self):
        """Test text with no sentiment words"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        # Use text with absolutely no financial sentiment words
        result = analyzer.analyze_text("Lorem ipsum dolor sit amet consectetur")

        assert result["sentiment"] == "neutral"
        assert result["score"] == 0.0
        assert result["confidence"] == 0.0

    def test_text_with_special_characters(self):
        """Test text with special characters"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_text("Stock gains!!! Profit??? $$$ surge @#$%")

        # Should still detect sentiment words
        assert len(result["positive_words"]) > 0

    def test_text_with_numbers(self):
        """Test text with numbers"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_text("Stock gains 10% profit rises by 25%")

        assert result["sentiment"] == "positive"

    def test_very_long_text(self):
        """Test handling of very long text"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        long_text = " ".join(["gain profit"] * 100)
        result = analyzer.analyze_text(long_text)

        assert result["sentiment"] == "positive"
        # Confidence should be maxed out
        assert result["confidence"] == 1.0

    def test_article_with_missing_fields(self):
        """Test article with missing fields"""
        from news.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        article = {"title": "Stock gains profit"}  # Missing description

        result = analyzer.analyze_article(article)

        assert "sentiment" in result
        assert result["sentiment"]["overall"] in ["positive", "negative", "neutral"]
