"""
Integrated News System
Combines multiple news sources with advanced sentiment analysis
Includes NewsAPI and Alpha Vantage news integration
"""

import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentCategory(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class EnhancedNewsArticle:
    """Enhanced news article with sentiment"""
    title: str
    description: str
    url: str
    source: str
    published_at: str
    symbol: Optional[str]
    sentiment_score: float  # -1 to +1
    sentiment_category: SentimentCategory
    sentiment_confidence: float
    relevance_score: float  # 0 to 1
    keywords: List[str]


class NewsAPISource:
    """
    NewsAPI.org integration for global news

    Free tier: 100 requests/day
    Get API key at: https://newsapi.org/
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.enabled = bool(api_key)

    async def get_articles(
        self,
        query: str,
        days: int = 7,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles from NewsAPI

        Args:
            query: Search query (stock symbol, company name, etc.)
            days: Number of days to look back
            language: Article language
            sort_by: Sort order (publishedAt, relevancy, popularity)
            page_size: Number of results

        Returns:
            List of news articles
        """
        if not self.enabled:
            logger.warning("NewsAPI not configured")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            async with aiohttp.ClientSession() as session:
                params = {
                    "q": query,
                    "from": from_date,
                    "language": language,
                    "sortBy": sort_by,
                    "pageSize": page_size,
                    "apiKey": self.api_key
                }

                async with session.get(f"{self.BASE_URL}/everything", params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        articles = []
                        for article in data.get("articles", []):
                            articles.append({
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "source": f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}",
                                "published_at": article.get("publishedAt", ""),
                                "author": article.get("author"),
                                "image_url": article.get("urlToImage"),
                                "content": article.get("content", "")[:500] if article.get("content") else None
                            })

                        logger.info(f"NewsAPI: Fetched {len(articles)} articles for '{query}'")
                        return articles

                    else:
                        error_data = await response.json()
                        logger.error(f"NewsAPI error: {error_data}")
                        return []

        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

    async def get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch top headlines"""
        if not self.enabled:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "category": category,
                    "country": country,
                    "pageSize": page_size,
                    "apiKey": self.api_key
                }

                async with session.get(f"{self.BASE_URL}/top-headlines", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "url": article.get("url", ""),
                                "source": f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}",
                                "published_at": article.get("publishedAt", ""),
                            }
                            for article in data.get("articles", [])
                        ]
                    return []

        except Exception as e:
            logger.error(f"NewsAPI headlines failed: {e}")
            return []


class AlphaVantageNewsSource:
    """
    Alpha Vantage News Sentiment API

    Free tier: 5 API calls/minute
    Premium gives more data and sentiment scores
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.enabled = bool(api_key)

    async def get_news_sentiment(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Fetch news with sentiment from Alpha Vantage

        Args:
            tickers: List of stock tickers (e.g., ["AAPL", "MSFT"])
            topics: Topics to filter (earnings, technology, finance, etc.)
            time_from: Start time in YYYYMMDDTHHMM format
            limit: Max results (up to 1000)

        Returns:
            Dict with feed and sentiment data
        """
        if not self.enabled:
            logger.warning("Alpha Vantage API not configured")
            return {"feed": [], "sentiment": None}

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": limit
            }

            if tickers:
                params["tickers"] = ",".join(tickers)
            if topics:
                params["topics"] = ",".join(topics)
            if time_from:
                params["time_from"] = time_from

            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Check for API errors
                        if "Error Message" in data or "Note" in data:
                            logger.error(f"Alpha Vantage error: {data}")
                            return {"feed": [], "sentiment": None}

                        feed = data.get("feed", [])
                        articles = []

                        for item in feed:
                            # Extract ticker sentiments
                            ticker_sentiments = {}
                            for ts in item.get("ticker_sentiment", []):
                                ticker_sentiments[ts.get("ticker")] = {
                                    "relevance": float(ts.get("relevance_score", 0)),
                                    "sentiment_score": float(ts.get("ticker_sentiment_score", 0)),
                                    "sentiment_label": ts.get("ticker_sentiment_label", "Neutral")
                                }

                            articles.append({
                                "title": item.get("title", ""),
                                "description": item.get("summary", ""),
                                "url": item.get("url", ""),
                                "source": f"Alpha Vantage - {item.get('source', 'Unknown')}",
                                "published_at": item.get("time_published", ""),
                                "authors": item.get("authors", []),
                                "overall_sentiment_score": float(item.get("overall_sentiment_score", 0)),
                                "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
                                "ticker_sentiments": ticker_sentiments,
                                "topics": [t.get("topic") for t in item.get("topics", [])],
                                "banner_image": item.get("banner_image")
                            })

                        # Calculate aggregate sentiment
                        if articles:
                            avg_sentiment = sum(a["overall_sentiment_score"] for a in articles) / len(articles)
                            sentiment_summary = {
                                "average_score": round(avg_sentiment, 3),
                                "total_articles": len(articles),
                                "positive_count": sum(1 for a in articles if a["overall_sentiment_score"] > 0.15),
                                "negative_count": sum(1 for a in articles if a["overall_sentiment_score"] < -0.15),
                                "neutral_count": sum(1 for a in articles if -0.15 <= a["overall_sentiment_score"] <= 0.15)
                            }
                        else:
                            sentiment_summary = None

                        logger.info(f"Alpha Vantage: Fetched {len(articles)} articles with sentiment")
                        return {"feed": articles, "sentiment": sentiment_summary}

                    return {"feed": [], "sentiment": None}

        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return {"feed": [], "sentiment": None}


class LLMSentimentAnalyzer:
    """
    LLM-powered sentiment analysis for deeper understanding
    Uses OpenAI or Gemini for nuanced financial sentiment
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key

    async def analyze_sentiment(
        self,
        articles: List[Dict[str, Any]],
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze sentiment of news articles

        Args:
            articles: List of news articles
            symbol: Stock symbol for context

        Returns:
            Sentiment analysis result
        """
        if not articles:
            return {
                "overall_sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "summary": "No articles to analyze",
                "key_themes": [],
                "market_impact": "neutral"
            }

        # Combine article titles and descriptions
        articles_text = "\n".join([
            f"- {a.get('title', '')} | {a.get('description', '')[:200]}"
            for a in articles[:10]  # Limit to 10 articles
        ])

        prompt = f"""Analyze the following financial news articles{f' about {symbol}' if symbol else ''} and provide sentiment analysis.

NEWS ARTICLES:
{articles_text}

Provide your analysis in the following JSON format:
{{
  "overall_sentiment": "very_positive" or "positive" or "neutral" or "negative" or "very_negative",
  "score": -1.0 to +1.0 (float),
  "confidence": 0.0 to 1.0 (float),
  "summary": "Brief summary of the news sentiment and key themes",
  "key_themes": ["theme1", "theme2", "theme3"],
  "market_impact": "bullish", "bearish", or "neutral",
  "key_risks": ["risk1", "risk2"],
  "key_opportunities": ["opportunity1", "opportunity2"],
  "recommended_action": "BUY", "SELL", or "HOLD" based on news sentiment
}}

Be specific and objective in your analysis."""

        try:
            if self.openai_api_key:
                return await self._call_openai(prompt)
            elif self.gemini_api_key:
                return await self._call_gemini(prompt)
            else:
                # Fallback to rule-based sentiment
                return self._fallback_sentiment(articles)

        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            return self._fallback_sentiment(articles)

    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI for sentiment analysis"""
        import openai
        import json

        client = openai.AsyncOpenAI(api_key=self.openai_api_key)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial news analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    async def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini for sentiment analysis"""
        from google import genai
        from google.genai import types
        import json

        client = genai.Client(api_key=self.gemini_api_key)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.0-flash",
            contents=types.Content(parts=[types.Part(text=prompt)]),
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )

        return json.loads(response.text)

    def _fallback_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rule-based sentiment fallback"""
        from news.sentiment import get_sentiment_analyzer

        analyzer = get_sentiment_analyzer()
        analyzed = analyzer.analyze_articles(articles)
        aggregate = analyzer.get_aggregate_sentiment(analyzed)

        return {
            "overall_sentiment": aggregate.get("overall_sentiment", "neutral"),
            "score": aggregate.get("average_score", 0.0),
            "confidence": 0.6,  # Lower confidence for rule-based
            "summary": f"Analysis of {aggregate.get('total_articles', 0)} articles",
            "key_themes": [],
            "market_impact": aggregate.get("overall_sentiment", "neutral"),
            "key_risks": [],
            "key_opportunities": [],
            "recommended_action": "HOLD"
        }


class IntegratedNewsService:
    """
    Unified news service combining all sources with enhanced sentiment
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        self.newsapi = NewsAPISource(newsapi_key)
        self.alphavantage = AlphaVantageNewsSource(alphavantage_key)
        self.llm_analyzer = LLMSentimentAnalyzer(openai_api_key, gemini_api_key)

        # Import other sources
        from news.sources import get_news_aggregator
        from news.advanced_sources import get_advanced_news_aggregator

        self.rss_aggregator = get_news_aggregator()
        self.advanced_aggregator = get_advanced_news_aggregator()

    async def get_comprehensive_news(
        self,
        symbol: str,
        days: int = 7,
        include_sentiment: bool = True,
        use_llm_sentiment: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive news from all sources with sentiment analysis

        Args:
            symbol: Stock symbol
            days: Days to look back
            include_sentiment: Whether to analyze sentiment
            use_llm_sentiment: Use LLM for sentiment (vs rule-based)

        Returns:
            Comprehensive news data with sentiment
        """
        # Fetch from all sources concurrently
        symbol_clean = symbol.replace('.NS', '').replace('.BO', '')

        tasks = [
            self.rss_aggregator.fetch_latest_news(symbol_clean, limit=20),
            self.advanced_aggregator.get_all_news(symbol_clean, days),
        ]

        # Add API sources if configured
        if self.newsapi.enabled:
            tasks.append(self.newsapi.get_articles(symbol_clean, days))

        if self.alphavantage.enabled:
            tasks.append(self.alphavantage.get_news_sentiment([symbol_clean]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine all articles
        all_articles = []
        alphavantage_sentiment = None

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"News source error: {result}")
                continue

            if isinstance(result, dict) and "feed" in result:
                # Alpha Vantage response
                all_articles.extend(result.get("feed", []))
                alphavantage_sentiment = result.get("sentiment")
            elif isinstance(result, list):
                all_articles.extend(result)

        # Deduplicate by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        # Sort by date
        unique_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)

        # Analyze sentiment
        sentiment_result = None
        if include_sentiment and unique_articles:
            if use_llm_sentiment:
                sentiment_result = await self.llm_analyzer.analyze_sentiment(
                    unique_articles[:15],  # Limit for LLM
                    symbol_clean
                )
            else:
                from news.sentiment import get_sentiment_analyzer
                analyzer = get_sentiment_analyzer()
                analyzed = analyzer.analyze_articles(unique_articles)
                sentiment_result = analyzer.get_aggregate_sentiment(analyzed)
                unique_articles = analyzed

        # Build response
        return {
            "symbol": symbol,
            "total_articles": len(unique_articles),
            "articles": unique_articles[:50],  # Limit response size
            "sentiment": sentiment_result,
            "alphavantage_sentiment": alphavantage_sentiment,
            "sources": {
                "rss": True,
                "newsapi": self.newsapi.enabled,
                "alphavantage": self.alphavantage.enabled
            },
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }

    async def get_market_news(
        self,
        category: str = "business",
        limit: int = 20,
        include_sentiment: bool = True
    ) -> Dict[str, Any]:
        """Get general market news"""
        tasks = [
            self.rss_aggregator.fetch_latest_news(limit=limit),
            self.advanced_aggregator.get_all_news(days=3),
        ]

        if self.newsapi.enabled:
            tasks.append(self.newsapi.get_top_headlines(category, page_size=limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        # Deduplicate and sort
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        unique_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)

        # Analyze sentiment
        sentiment_result = None
        if include_sentiment and unique_articles:
            sentiment_result = await self.llm_analyzer.analyze_sentiment(unique_articles[:10])

        return {
            "category": category,
            "total_articles": len(unique_articles),
            "articles": unique_articles[:limit],
            "sentiment": sentiment_result,
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }


# Singleton instance
_integrated_news_service: Optional[IntegratedNewsService] = None


def get_integrated_news_service(
    newsapi_key: Optional[str] = None,
    alphavantage_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None
) -> IntegratedNewsService:
    """Get or create integrated news service"""
    global _integrated_news_service

    # Always create new if keys provided (allows config updates)
    if newsapi_key or alphavantage_key or openai_api_key or gemini_api_key:
        return IntegratedNewsService(
            newsapi_key=newsapi_key,
            alphavantage_key=alphavantage_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )

    if _integrated_news_service is None:
        _integrated_news_service = IntegratedNewsService()

    return _integrated_news_service
