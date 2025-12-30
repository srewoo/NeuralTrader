"""
Web Search Service for Real-Time News

Supports multiple search providers:
- DuckDuckGo (free, no API key required)
- SerpAPI (paid, higher quality results)

Integrates with existing news sentiment analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import re

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS = None

import aiohttp
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result"""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    relevance_score: float = 0.0


class WebSearchService:
    """
    Web search service for real-time news and information.

    Features:
    - Multiple provider support (DuckDuckGo, SerpAPI)
    - Automatic fallback between providers
    - Result deduplication
    - Relevance scoring
    - Date filtering
    """

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        default_provider: str = "duckduckgo"
    ):
        """
        Initialize web search service.

        Args:
            serpapi_key: API key for SerpAPI (optional)
            default_provider: "duckduckgo" or "serpapi"
        """
        self.serpapi_key = serpapi_key
        self.default_provider = default_provider
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        days_back: int = 7,
        region: str = "in-en",  # India English
        provider: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for news articles.

        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: Look back this many days
            region: Region code (in-en for India)
            provider: Override default provider

        Returns:
            List of SearchResult objects
        """
        provider = provider or self.default_provider

        try:
            if provider == "serpapi" and self.serpapi_key:
                results = await self._search_serpapi(
                    query, max_results, days_back, region
                )
            else:
                results = await self._search_duckduckgo(
                    query, max_results, days_back, region
                )

            return results

        except Exception as e:
            logger.error(f"Search failed with {provider}: {e}")
            # Fallback to alternative provider
            if provider == "serpapi":
                logger.info("Falling back to DuckDuckGo")
                return await self._search_duckduckgo(
                    query, max_results, days_back, region
                )
            return []

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        days_back: int,
        region: str
    ) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        if not DDGS_AVAILABLE:
            logger.warning("duckduckgo-search not installed")
            return []

        results = []

        try:
            # Run blocking DDGS call in executor
            loop = asyncio.get_event_loop()
            ddgs_results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().news(
                    keywords=query,
                    region=region,
                    max_results=max_results
                ))
            )

            for item in ddgs_results:
                # Parse date
                published_date = None
                if 'date' in item:
                    try:
                        # DuckDuckGo returns dates in various formats
                        date_str = item['date']
                        # Try parsing common formats
                        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d", "%d/%m/%Y"]:
                            try:
                                published_date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

                # Filter by date if we have it
                if published_date:
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    if published_date.replace(tzinfo=None) < cutoff_date:
                        continue

                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    snippet=item.get('body', ''),
                    source=item.get('source', 'Unknown'),
                    published_date=published_date,
                    relevance_score=1.0  # DDG doesn't provide scores
                ))

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")

        return results

    async def _search_serpapi(
        self,
        query: str,
        max_results: int,
        days_back: int,
        region: str
    ) -> List[SearchResult]:
        """Search using SerpAPI Google News"""
        if not self.serpapi_key:
            logger.warning("SerpAPI key not configured")
            return []

        session = await self._get_session()

        # Map region to Google News region code
        region_map = {
            "in-en": "IN",
            "us-en": "US",
            "uk-en": "GB"
        }
        gl = region_map.get(region, "IN")

        params = {
            "engine": "google_news",
            "q": query,
            "gl": gl,
            "hl": "en",
            "api_key": self.serpapi_key,
            "num": max_results
        }

        try:
            url = "https://serpapi.com/search"
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.error(f"SerpAPI returned status {resp.status}")
                    return []

                data = await resp.json()
                results = []

                for item in data.get("news_results", [])[:max_results]:
                    # Parse date
                    published_date = None
                    if 'date' in item:
                        try:
                            date_str = item['date']
                            # SerpAPI returns relative dates like "2 hours ago"
                            published_date = self._parse_relative_date(date_str)
                        except Exception:
                            pass

                    # Filter by date
                    if published_date:
                        cutoff_date = datetime.now() - timedelta(days=days_back)
                        if published_date < cutoff_date:
                            continue

                    results.append(SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source=item.get('source', {}).get('name', 'Unknown'),
                        published_date=published_date,
                        relevance_score=1.0
                    ))

                return results

        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []

    def _parse_relative_date(self, date_str: str) -> Optional[datetime]:
        """Parse relative date strings like '2 hours ago', '3 days ago'"""
        now = datetime.now()

        # Extract number and unit
        match = re.search(r'(\d+)\s*(hour|day|minute|week)s?\s*ago', date_str.lower())
        if not match:
            return now  # Assume recent if can't parse

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 'minute':
            return now - timedelta(minutes=value)
        elif unit == 'hour':
            return now - timedelta(hours=value)
        elif unit == 'day':
            return now - timedelta(days=value)
        elif unit == 'week':
            return now - timedelta(weeks=value)

        return now

    async def search_stock_news(
        self,
        symbol: str,
        company_name: Optional[str] = None,
        max_results: int = 10,
        days_back: int = 7
    ) -> List[SearchResult]:
        """
        Search for news about a specific stock.

        Args:
            symbol: Stock symbol
            company_name: Optional company name for better results
            max_results: Maximum results
            days_back: Days to look back

        Returns:
            List of SearchResult objects
        """
        # Build query
        if company_name:
            query = f"{company_name} {symbol} stock news"
        else:
            query = f"{symbol} stock news"

        # Add India-specific terms for NSE stocks
        if symbol in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]:
            query += " India NSE"

        return await self.search_news(
            query=query,
            max_results=max_results,
            days_back=days_back,
            region="in-en"
        )

    async def search_market_news(
        self,
        topic: str = "Indian stock market",
        max_results: int = 10,
        days_back: int = 3
    ) -> List[SearchResult]:
        """
        Search for general market news.

        Args:
            topic: Market topic to search
            max_results: Maximum results
            days_back: Days to look back

        Returns:
            List of SearchResult objects
        """
        return await self.search_news(
            query=topic,
            max_results=max_results,
            days_back=days_back,
            region="in-en"
        )

    async def search_sector_news(
        self,
        sector: str,
        max_results: int = 10,
        days_back: int = 7
    ) -> List[SearchResult]:
        """
        Search for sector-specific news.

        Args:
            sector: Sector name (e.g., "banking", "IT", "pharma")
            max_results: Maximum results
            days_back: Days to look back

        Returns:
            List of SearchResult objects
        """
        query = f"{sector} sector India stock market news"
        return await self.search_news(
            query=query,
            max_results=max_results,
            days_back=days_back,
            region="in-en"
        )

    def deduplicate_results(
        self,
        results: List[SearchResult],
        similarity_threshold: float = 0.8
    ) -> List[SearchResult]:
        """
        Remove duplicate/very similar results.

        Args:
            results: List of search results
            similarity_threshold: Title similarity threshold

        Returns:
            Deduplicated list
        """
        if not results:
            return []

        unique_results = []
        seen_urls = set()

        for result in results:
            # Skip if exact URL match
            if result.url in seen_urls:
                continue

            # Check title similarity with existing results
            is_duplicate = False
            for existing in unique_results:
                if self._titles_similar(result.title, existing.title, similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_urls.add(result.url)

        return unique_results

    def _titles_similar(self, title1: str, title2: str, threshold: float) -> bool:
        """Check if two titles are similar using simple word overlap"""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard = len(intersection) / len(union) if union else 0
        return jaccard >= threshold


# Singleton instance
_search_service: Optional[WebSearchService] = None


def get_web_search_service(
    serpapi_key: Optional[str] = None
) -> WebSearchService:
    """Get singleton web search service instance"""
    global _search_service
    if _search_service is None:
        _search_service = WebSearchService(serpapi_key=serpapi_key)
    return _search_service


async def search_stock_news(
    symbol: str,
    company_name: Optional[str] = None,
    max_results: int = 10,
    days_back: int = 7
) -> List[Dict[str, Any]]:
    """
    Convenience function to search stock news.

    Returns:
        List of news dictionaries
    """
    service = get_web_search_service()
    results = await service.search_stock_news(
        symbol=symbol,
        company_name=company_name,
        max_results=max_results,
        days_back=days_back
    )

    # Deduplicate
    results = service.deduplicate_results(results)

    # Convert to dicts
    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "source": r.source,
            "published_date": r.published_date.isoformat() if r.published_date else None,
            "relevance_score": r.relevance_score
        }
        for r in results
    ]
